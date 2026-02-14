import pdfplumber
from google import genai
import typing_extensions as typing
import json
import datetime

# 1. SETUP API & MODEL
api_key = "AIzaSyC1fQquGOe2_SCpBRiJaGhn2neeiQ0qGTE"
client = genai.Client(api_key=api_key)
target_model = 'gemini-3-flash-preview'

# 2. DEFINE THE DATA SHAPE (The "Blueprint")
class CourseItem(typing.TypedDict):
    event_type: str  # Exam, Assignment, or Quiz
    title: str
    due_date: str

class Schedule(typing.TypedDict):
    course_name: str
    schedule_items: list[CourseItem]

# 3. EXTRACT TEXT FROM PDF
print("Reading PDF...")
pdf_path = "syllabus_gs.pdf"  # Make sure your file is named this or change it here
full_text = ""

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

# 4. SEND TO AI
print(f"Parsing with {target_model}...")

prompt = f"""
Extract all assignments, quizzes, and exams from the following syllabus text.
Format the dates clearly (e.g., 'Oct 5').
Syllabus Text:
{full_text}
"""

response = client.models.generate_content(
    model=target_model,
    contents=prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Schedule,
    }
)

# 5. SAVE AND VIEW RESULT
data = json.loads(response.text)

with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)

def structured_prior(item, current_date="2026-02-13"):
    """
    Inputs: item (dict from your JSON), current_date
    Output: estimated_hours (float)
    """
    from datetime import datetime
    
    # Factor 1: Time Proximity (t)
    due = datetime.strptime(item['due_date'], "%b %d") # Adjust format if needed
    due = due.replace(year=2026)
    today = datetime.strptime(current_date, "%Y-%m-%d")
    days_left = max((due - today).days, 1)
    
    # Factor 2: Type (Heuristic weight)
    type_weights = {"Exam": 15, "Quiz": 3, "Assignment": 5}
    base_hours = type_weights.get(item['event_type'], 4)
    
    # Factor 3: Complexity (Scaling the workload)
    # The closer it is, the more 'work' it occupies in your immediate horizon
    estimated_workload = (base_hours / days_left) * 1.5 
    
    return round(estimated_workload, 2)

print("--- Extraction Complete! ---")
print(json.dumps(data, indent=2))
