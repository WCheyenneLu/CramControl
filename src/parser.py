from datetime import datetime, timedelta, date
import re

def parse_date_to_iso(s: str, quarter_start: date, year_fallback: int):
    """
    Convert due_date strings to ISO date (YYYY-MM-DD) where possible.
    """
    if not s:
        return None, "missing", False

    s_clean = s.strip()

    # 1. Recurring phrases
    if re.search(r"\bweekly\b|\bthroughout\b|\bevery\b|\beach\b", s_clean, re.I):
        return None, "recurring_or_range", True

    # 2. Week N pattern (e.g., "Week 10")
    m1 = re.search(r"Week\s*(\d+)", s_clean, re.I)
    if m1:
        week_num = int(m1.group(1))
        iso_date = quarter_start + timedelta(weeks=week_num - 1)
        return iso_date.isoformat(), "academic_week", False

    # 3. ISO date already
    try:
        # Just grab the first 10 chars in case Gemini appended time
        dt = datetime.fromisoformat(s_clean[:10]) 
        return dt.date().isoformat(), "ok", False
    except Exception:
        pass

    # 4. Relaxed Month Day pattern
    # Added abbreviations so "Oct 7", "Sept. 15th", etc., will all work.
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, 
        "march": 3, "mar": 3, "april": 4, "apr": 4, 
        "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, 
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, 
        "october": 10, "oct": 10, "november": 11, "nov": 11, 
        "december": 12, "dec": 12
    }
    
    # This regex searches for ANY month word, ignores punctuation, and grabs the next 1-2 digits
    month_keys = "|".join(months.keys())
    m2 = re.search(fr"(?i)({month_keys})[\s\.\,]*(\d{{1,2}})", s_clean)
    
    if m2:
        month_str = m2.group(1).lower()
        day = int(m2.group(2))
        month = months[month_str]
        
        try:
            iso = datetime(year_fallback, month, day).date().isoformat()
            return iso, "ok", False
        except Exception:
            return None, "unparsed", False

    return None, "unparsed", False