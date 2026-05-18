import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available chat models:")
for model in client.models.list():
    if "gemini" in model.name.lower() and "embed" not in model.name.lower():
        print(f"  {model.name}")