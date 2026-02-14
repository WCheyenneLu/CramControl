import pdfplumber

full_text = ""

with pdfplumber.open("/Users/sanjitha_vaidyanathan/Documents/syllabus_gs.pdf") as pdf:
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

print(full_text[:500])

