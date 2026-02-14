import streamlit as st
import pdfplumber
import json
import pandas as pd
from google import genai
import typing_extensions as typing

# --- 1. APP CONFIG ---
st.set_page_config(page_title="CramControl", page_icon="📚")
st.title("📚 CramControl: Syllabus Parser")
st.write("Upload your syllabus to extract all important dates.")

# --- 2. API SETUP ---
# It's better to put your key in st.secrets later, but for now, paste it here:
API_KEY = "AIzaSyC1fQquGOe2_SCpBRiJaGhn2neeiQ0qGTE"
client = genai.Client(api_key=API_KEY)

# Define the JSON Blueprint
class CourseItem(typing.TypedDict):
    event_type: str
    title: str
    due_date: str

class Schedule(typing.TypedDict):
    course_name: str
    schedule_items: list[CourseItem]

# --- 3. UI: FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a syllabus PDF", type="pdf")

if uploaded_file is not None:
    # 4. EXTRACT TEXT
    with st.spinner("Reading PDF..."):
        full_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
        
    st.success("PDF Uploaded Successfully!")

    # 5. BUTTON TO TRIGGER AI
    if st.button("Extract Schedule ✨"):
        with st.spinner("Gemini is thinking..."):
            try:
                response = client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=f"Extract assignments/exams from this syllabus: {full_text}",
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': Schedule,
                    }
                )
                
                # 6. PARSE & DISPLAY
                raw_data = json.loads(response.text)
                
                # Show Course Name
                st.subheader(f"Course: {raw_data.get('course_name', 'Unknown')}")
                
                # Convert list to a Table (DataFrame)
                df = pd.DataFrame(raw_data['schedule_items'])
                
                # Display the Table
                st.dataframe(df, use_container_width=True)
                
                # Optional: Let user download the JSON
                st.download_button("Download JSON", response.text, "schedule.json")

            except Exception as e:
                st.error(f"Error: {e}")




