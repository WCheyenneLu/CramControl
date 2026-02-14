from google import genai

client = genai.Client(api_key="AIzaSyC1fQquGOe2_SCpBRiJaGhn2neeiQ0qGTE")

response = client.models.generate_content(
    model="models/gemini-3-flash-preview",
    contents="Hello!"
)

print(response.text)
