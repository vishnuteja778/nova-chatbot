"""Step 0: check that your API key works and see which models it can use."""
from dotenv import load_dotenv
from google import genai

load_dotenv()              # copies the values in .env into environment variables
client = genai.Client()    # finds GEMINI_API_KEY in the environment by itself

print("Models that can generate text:\n")
for model in client.models.list():
    # Skip models that can't chat, such as embedding-only models.
    if "generateContent" in (model.supported_actions or []):
        print(f"  {model.name.removeprefix('models/'):40} {model.display_name}")
