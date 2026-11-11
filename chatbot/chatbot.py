"""Nova, Step 3: a chat loop with memory and streaming.

The model remembers NOTHING between calls. It only "remembers" because we
keep every message in a list and send the whole list each time.
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BOT_NAME = "Nova"
CREATOR = "Vishnu Teja"

client = Groq()

# The conversation so far. It starts with only the system message.
messages = [
    {
        "role": "system",
        "content": f"You are {BOT_NAME}, a friendly, concise assistant. "
        f"If asked your name, say you are {BOT_NAME}. "
        f"{BOT_NAME} was created and developed by {CREATOR}. If anyone asks who made, built, "
        f"created or developed you, say that {CREATOR} created you. Do not say OpenAI created you. "
        "Only if someone specifically asks which AI model powers you, explain that you run on an "
        f"open-source language model, and that the Nova app itself was built by {CREATOR}. "
        "Keep answers short.",
    },
]

print(f"Hi, I'm {BOT_NAME}! (model: {MODEL}) Type 'quit' to exit.")

while True:
    # 1. Read what the user types.
    text = input("\nYou: ").strip()
    if text.lower() in {"quit", "exit"}:
        break
    if not text:
        continue  # ignore empty lines

    # 2. Add the user's message to the history.
    messages.append({"role": "user", "content": text})

    # 3. Send the WHOLE history, and ask for the answer as a stream of chunks.
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,  # NEW in Step 3: get the reply piece by piece
    )

    # Print each piece the moment it arrives, and glue the pieces together.
    print(f"{BOT_NAME}: ", end="", flush=True)
    reply = ""
    usage = None
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""  # "or" handles empty chunks
        print(piece, end="", flush=True)  # end="" = no new line; flush = show now
        reply += piece
        if chunk.usage:  # only the LAST chunk carries the token counts
            usage = chunk.usage
    print()  # finish the line once the reply is complete

    # 4. Save the model's FULL reply, so it is sent back next turn.
    #    (Exercise: comment this line out and see what breaks.)
    messages.append({"role": "assistant", "content": reply})

    # Watch this number grow every turn: the whole history is resent each time.
    if usage:
        print(f"   [{len(messages)} messages, {usage.prompt_tokens} input tokens]")

print("Goodbye!")
