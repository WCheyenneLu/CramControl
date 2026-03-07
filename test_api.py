import argparse
from datetime import date, datetime
import json
import os
import re
from typing_extensions import TypedDict
from pathlib import Path
from optimizer import optimize_weekly_schedule

import google.generativeai as genai
import pdfplumber
import pandas as pd
import joblib

# Assuming your parser.py is inside the src folder or same directory
from src.parser import parse_date_to_iso 

# Path to your trained XGBoost model
MODEL_PATH = Path("data/model_hours_xgb.joblib")

PROMPT = """
Extract structured course information from this syllabus text.

Return JSON only using this structure:
{
  "course_name": "string",
  "course_code": "string",
  "schedule_items": [
    {
      "assignment_type": "reading|essay_or_presentation|weekly_short_task|weekly_long_task|larger_project|capstone_project|midterm|final|other",
      "assignment_name": "string",
      "due_date": "string",
      "weighting": "string" or null
    }
  ]
}

Classification rules:
- Use "midterm" when text says "midterm", "MT1/MT2", or "midterm exam".
- Use "final" when text says "final", "final exam", or "comprehensive final".
- Use "reading" for required readings or reading responses.
- Use "essay_or_presentation" for essays and presentations.
- Use "weekly_short_task" for reflections, discussion posts, journals, quizzes, and labs.
- Use "weekly_long_task" for weekly longer assignments such as problem sets and longer homework sets.
- Use "larger_project" for coding assignments or longer bi-weekly style project work.
- Use "capstone_project" for capstone/research/final projects spanning a large portion of the term.
- Use "other" only when no category clearly applies.

Date rules:
- Extract and clean the due dates into a standardized "Month Day" format (e.g., convert "3.4 We at 11:59 pm" to "March 4").
- Remove all times (e.g., "11:59 pm") and days of the week (e.g., "Tuesday", "We") to ensure strict compatibility with date parsers.
- If an assignment description lacks dates, heavily scrutinize the Course Calendar grid at the end of the syllabus.
- If a date is just a week (e.g., "Week 3"), output "Week 3".
- If no clear due date exists, use "unknown".

Weighting rules:
- Extract the percentage of the final course grade this specific item represents (e.g., if it is worth 25%, output "25%").
- If the weight is given in points rather than a percentage, or if the weight cannot be determined, output null.

Output rules:
- Include all dated course events you can find.
- One schedule item per event (e.g., if there are 4 essays, create 4 separate items).
- Do not add keys not listed in the schema.
- Return valid JSON only, without any markdown formatting blocks.
"""

class CourseItem(TypedDict):
    assignment_type: str
    assignment_name: str
    due_date: str
    weighting: str | None
    
    # Injected fields
    due_date_iso: str | None
    due_date_parse_status: str
    is_recurring: bool
    task_type: str
    weight_numeric: float
    estimated_hours: float

class Schedule(TypedDict):
    course_name: str
    course_code: str
    weekly_schedule: dict

# ---------------------------------------------------------
# Helper Functions (Merged from App 2)
# ---------------------------------------------------------

def normalize_items(items: list[dict], quarter_start: date, year_fallback: int) -> list[dict]:
    for it in items:
        iso, status, is_rec = parse_date_to_iso(
            it.get("due_date", ""),
            quarter_start,
            year_fallback
        )
        it["due_date_iso"] = iso
        it["due_date_parse_status"] = status
        it["is_recurring"] = is_rec

    # sort parsed first, then unparsed
    def sort_key(it):
        iso = it.get("due_date_iso")
        return (0, iso) if iso else (1, it.get("due_date", ""))

    return sorted(items, key=sort_key)

def parse_weight_to_float(weighting) -> float:
    if not weighting:
        return 0.0
    s = str(weighting).strip().lower()
    if s in {"unknown", "n/a", "na", "none"}:
        return 0.0
    m = re.search(r"(\d+(\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+(\.\d+)?)\s*percent", s)
    if m2:
        return float(m2.group(1))
    # Fallback: if they just provided "25", assume it means 25%
    m3 = re.search(r"^(\d+(\.\d+)?)$", s)
    if m3:
        return float(m3.group(1))
    return 0.0

def map_assignment_type_to_task_type(assignment_type: str, assignment_name: str) -> str:
    """ Maps App 1's detailed Prompt buckets to App 2's Model buckets. """
    t = (assignment_type or "").lower().strip()
    name = (assignment_name or "").lower()

    if "midterm" in name: return "midterm_easy"
    if "final" in name: return "final_hard"
    if "capstone" in name or "research project" in name: return "capstone"
    if "essay" in name or "paper" in name or "presentation" in name: return "essay"

    mapping = {
        "reading": "reading",
        "essay_or_presentation": "essay",
        "weekly_short_task": "weekly_short",
        "weekly_long_task": "weekly_long",
        "larger_project": "project",
        "capstone_project": "capstone",
        "midterm": "midterm_easy",
        "final": "final_hard",
        "other": "other"
    }
    return mapping.get(t, "other")

def days_until_due(due_iso: str) -> int:
    try:
        d = datetime.fromisoformat(str(due_iso)).date()
        return max(0, (d - date.today()).days)
    except Exception:
        return 30

def predict_hours_for_item(model, light_week_max, heavy_week_max, task_type, weight, days_until) -> float:
    heavy_light_ratio = (heavy_week_max + 1e-6) / (light_week_max + 1e-6)
    X = pd.DataFrame([{
        "student_id": 0,
        "task_id": 0,
        "light_week_max": float(light_week_max),
        "heavy_week_max": float(heavy_week_max),
        "heavy_light_ratio": float(heavy_light_ratio),
        "task_type": str(task_type),
        "weight": float(weight),
        "days_until_due": int(days_until),
    }])
    return round(float(model.predict(X)[0]), 2)

# ---------------------------------------------------------
# Main Extraction Logic
# ---------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("No extractable text found in PDF.")
    return text

def parse_syllabus_text(syllabus_text: str, api_key: str, quarter_start: date, year_fallback: int, light_week_max: float, heavy_week_max: float) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    response = model.generate_content(
        f"{PROMPT}\n\nSYLLABUS TEXT:\n{syllabus_text}",
        generation_config={"response_mime_type": "application/json"},
    )
    data = json.loads(response.text)
    
    # 1. Normalize dates
    items = normalize_items(data.get("schedule_items", []), quarter_start, year_fallback)
    
    # 2. Load ML model safely
    ml_model = None
    if MODEL_PATH.exists():
        ml_model = joblib.load(MODEL_PATH)

    # 3. Calculate model features & predict
    for item in items:
        item["task_type"] = map_assignment_type_to_task_type(item.get("assignment_type", ""), item.get("assignment_name", ""))
        item["weight_numeric"] = parse_weight_to_float(item.get("weighting"))
        
        d_until = days_until_due(item.get("due_date_iso"))
        if item.get("is_recurring"): d_until = 7
            
        if ml_model:
            item["estimated_hours"] = predict_hours_for_item(
                ml_model, light_week_max, heavy_week_max, 
                item["task_type"], item["weight_numeric"], d_until
            )
        else:
            item["estimated_hours"] = 0.0
    
    # Build final JSON object replacing schedule_items with weekly_schedule
    return {
        "course_name": data.get("course_name", "Unknown Course"),
        "course_code": data.get("course_code", "Unknown Code"),
        "quarter_start": str(quarter_start),
        "schedule_items": items # Return the flat list!
    }

    return final_output

def parse_syllabus_pdf(pdf_path: str, api_key: str, quarter_start: date, year_fallback: int, light_week_max: float, heavy_week_max: float) -> dict:
    syllabus_text = extract_pdf_text(pdf_path)
    return parse_syllabus_text(syllabus_text, api_key, quarter_start, year_fallback, light_week_max, heavy_week_max)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract schedule data from a syllabus PDF and estimate workloads."
    )
    parser.add_argument("pdf_path", help="Path to the syllabus PDF file")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    quarter_start = date(2026, 1, 6) 
    year_fallback = 2026

    # For CLI, we pass standard defaults for light/heavy maxes. 
    data = parse_syllabus_pdf(args.pdf_path, api_key, quarter_start, year_fallback, light_week_max=12.0, heavy_week_max=25.0)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()