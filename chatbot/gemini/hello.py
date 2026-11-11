"""Step 1: send ONE message to Gemini and print the answer and token usage."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client()  # reads GEMINI_API_KEY from the environment

response = client.models.generate_content(
    model=MODEL,
    # "contents" is what the user says. A plain string is fine for one message.
    contents="What is an API? Explain it to a beginner.",
    config=types.GenerateContentConfig(
        # Gemini takes the system prompt here, not as a message in the list.
        system_instruction="Answer in at most two sentences.",
        temperature=0.5,  # 0 = predictable, higher = more creative
    ),
)

print(response.text)

usage = response.usage_metadata
print("\n--- tokens ---")
print("input (prompt):", usage.prompt_token_count)
print("output (answer):", usage.candidates_token_count)
print("total:", usage.total_token_count)
