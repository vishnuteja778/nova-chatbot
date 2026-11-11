"""Step 1: send ONE message to Groq and print the answer and token usage."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

client = Groq()  # reads GROQ_API_KEY from the environment

response = client.chat.completions.create(
    model=MODEL,
    # The conversation: a list of messages, each with a role and content.
    messages=[
        {"role": "system", "content": "Answer in at most two sentences."},
        {"role": "user", "content": "What is an API? Explain it to a beginner."},
    ],
    temperature=0.5,  # 0 = predictable, higher = more creative
)

# "choices" is a list because you can ask for several answers; we asked for one.
print(response.choices[0].message.content)

usage = response.usage
print("\n--- tokens ---")
print("input (prompt):", usage.prompt_tokens)
print("output (answer):", usage.completion_tokens)
print("total:", usage.total_tokens)
