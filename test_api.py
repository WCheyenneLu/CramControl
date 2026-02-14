from google import genai
import typing_extensions as typing
import json

# 1. SETUP API 
api_key = "AIzaSyC1fQquGOe2_SCpBRiJaGhn2neeiQ0qGTE"
client = genai.Client(api_key=api_key)

# 2. DEFINE THE DATA SHAPE
class CourseItem(typing.TypedDict):
    event_type: str
    title: str
    due_date: str

class Schedule(typing.TypedDict):
    course_name: str
    schedule_items: list[CourseItem]

# 3. INITIALIZE AI + STRUCTURED OUTPUT
print("Sending request to AI...")

response = client.models.generate_content(
    model='gemini-3-flash-preview', 
    contents="The course is Biology 101. Midterm is on Oct 5th. Final is Dec 12th.",
    config={
        'response_mime_type': 'application/json', # <-- Tells AI to output ONLY JSON
        'response_schema': Schedule,             # <-- The "Blueprint" it must follow
    }
)

# 4. PRINT OUT THE RESULT
# response.text is now 100% clean JSON, no "Sure!" or "Here is..."
data = json.loads(response.text)
print(json.dumps(data, indent=2))