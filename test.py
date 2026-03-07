import os
from google import genai
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Hello"
)
print(response.text)

import pdfplumber as pp
path = "ARCFundBudget-SBLakers.pdf"
with pp.open(path) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        print(f"\n---Page {i}---\n")
        if text:
            print(text)
        else:
            print("[No text found on this page]")