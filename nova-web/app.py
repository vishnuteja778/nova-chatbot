"""Nova on the web: a FastAPI backend that streams Groq replies to the browser.

Vercel finds the FastAPI instance named `app` in this file and runs it as a
serverless function. The chat page lives in static/ and is served from the CDN.

Key idea: a serverless function keeps NO memory between requests (just like
the model). So the BROWSER keeps the conversation and sends the whole history
with every message, exactly like messages[] in chatbot.py.
"""
import io
import os
import secrets
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from xml.etree import ElementTree

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()  # local only; on Vercel the variables come from the dashboard

# .strip() removes spaces and line breaks that sneak in when values are copied
# and pasted into a dashboard. A key with a trailing line break can't be sent
# in an HTTP header, which the Groq library reports as "could not connect".
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()  # optional password for your site
BOT_NAME = "Nova"
CREATOR = "Vishnu Teja"
SYSTEM_PROMPT = (
    f"You are {BOT_NAME}, a friendly, concise assistant. "
    f"If asked your name, say you are {BOT_NAME}. "
    f"{BOT_NAME} was created and developed by {CREATOR}. If anyone asks who made, built, "
    f"created or developed you, say that {CREATOR} created you. Do not say OpenAI created you. "
    "Only if someone specifically asks which AI model powers you, explain that you run on an "
    "open-source language model, and that the Nova app itself was built by "
    f"{CREATOR}. Keep answers short unless asked for detail. "
    "The user may attach files. Their contents appear inside <file name=\"...\"> tags in the "
    "user's message. Use them to answer, and mention the file name when you quote from one."
)

# Limits that protect your free Groq quota from a single visitor.
MAX_MESSAGES = 20          # only the last 20 messages are sent to the model
MAX_CHARS = 50_000         # longest message accepted (attached file text included)

# Upload limits. Vercel accepts request bodies up to 4.5 MB.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_FILE_CHARS = 15_000    # about 4,000 tokens per file, to stay inside free-tier limits
MAX_PDF_PAGES = 50

app = FastAPI(title="Nova")
client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class Message(BaseModel):
    # Only "user" and "assistant" are allowed from the browser. The system
    # prompt stays on the server so visitors can't rewrite Nova's instructions.
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=MAX_CHARS)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


# ---------- Rate limiting ----------
# Stops one visitor from burning through your Groq quota.
#
# Note: this memory lives inside ONE running copy of the server. Vercel may
# run several copies at once and restarts them now and then, so these limits
# are a strong speed bump, not an exact guarantee. For exact limits across
# all copies you'd store the counts in a shared database such as Redis.

class RateLimiter:
    """Allows at most `limit` hits per visitor within the last `window` seconds."""

    MAX_VISITORS = 10_000  # forget the oldest visitors so memory can't grow forever

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()  # sync routes run on several threads

    def _recent(self, key: str, now: float) -> deque:
        q = self.hits.get(key)
        if q is None:
            if len(self.hits) >= self.MAX_VISITORS:
                self.hits.pop(next(iter(self.hits)))  # drop the oldest visitor
            q = self.hits[key] = deque()
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key: str) -> int:
        """0 if the visitor may continue, otherwise seconds until they may."""
        with self.lock:
            now = time.monotonic()
            q = self._recent(key, now)
            if len(q) < self.limit:
                return 0
            return max(1, int(self.window - (now - q[0])) + 1)

    def hit(self, key: str) -> None:
        with self.lock:
            now = time.monotonic()
            self._recent(key, now).append(now)

    def take(self, key: str) -> int:
        """Count one hit if allowed. Returns 0, or seconds to wait if not."""
        with self.lock:
            now = time.monotonic()
            q = self._recent(key, now)
            if len(q) >= self.limit:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return 0


chat_limiter = RateLimiter(limit=20, window=60)          # 20 messages per minute
extract_limiter = RateLimiter(limit=10, window=60)       # 10 file uploads per minute
bad_code_limiter = RateLimiter(limit=5, window=15 * 60)  # 5 wrong codes per 15 minutes


def client_ip(request: Request) -> str:
    # Vercel puts the visitor's real IP in x-real-ip (it overwrites any value
    # the visitor tries to send). Locally we fall back to the socket address.
    return (request.headers.get("x-real-ip", "").strip()
            or (request.client.host if request.client else "unknown"))


def too_many(seconds: int, message: str) -> HTTPException:
    return HTTPException(status_code=429, detail=message, headers={"Retry-After": str(seconds)})


def check_access(request: Request, limiter: RateLimiter) -> None:
    """Check the access code (if one is set), then the visitor's rate limit."""
    ip = client_ip(request)
    if ACCESS_CODE:
        # Visitors who keep guessing wrong codes are locked out for a while.
        wait = bad_code_limiter.retry_after(ip)
        if wait:
            raise too_many(wait, f"Too many wrong access codes. Try again in {wait // 60 + 1} min.")
        sent = request.headers.get("x-access-code", "")
        # compare_digest avoids leaking the code through timing differences
        if not secrets.compare_digest(sent.encode(), ACCESS_CODE.encode()):
            if sent:  # an empty code is just "not entered yet", not a guess
                bad_code_limiter.hit(ip)
            raise HTTPException(status_code=401, detail="Wrong or missing access code.")

    wait = limiter.take(ip)
    if wait:
        raise too_many(wait, f"You're sending messages too fast. Try again in {wait} s.")


@app.get("/api/health")
def health():
    """Quick check that the server is up and configured."""
    return {
        "ok": client is not None,
        "model": MODEL,
        "bot": BOT_NAME,
        "access_code_required": bool(ACCESS_CODE),
    }


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    check_access(request, chat_limiter)
    if client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on the server.")

    # Same as chatbot.py: system prompt first, then the (trimmed) history.
    history = [m.model_dump() for m in body.messages][-MAX_MESSAGES:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Start the stream BEFORE returning, so key/rate-limit errors become
    # proper HTTP errors instead of a half-finished reply.
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.7, stream=True,
        )
    except groq.AuthenticationError:
        raise HTTPException(status_code=500, detail="The server's Groq API key is invalid.")
    except groq.RateLimitError:
        raise HTTPException(status_code=429, detail="Nova is busy (rate limit). Try again in a minute.")
    except groq.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach Groq.")
    except groq.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Groq error {e.status_code}.")

    def generate():
        # Each piece is sent to the browser as soon as Groq produces it.
        try:
            for chunk in stream:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    yield piece
        except groq.APIError:
            yield "\n\n[Connection to Groq was interrupted. Please try again.]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- File uploads ----------
# Plain text and code files are read by the browser. PDF and Word files are
# binary formats, so the browser sends their raw bytes here and gets text back.

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise HTTPException(status_code=422, detail="This PDF is password-protected.")
    pages = reader.pages[:MAX_PDF_PAGES]
    return "\n\n".join((page.extract_text() or "").strip() for page in pages)


def read_docx(data: bytes) -> str:
    # A .docx file is a ZIP archive; the text lives in word/document.xml.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > 20 * 1024 * 1024:  # guard against "zip bombs"
            raise HTTPException(status_code=413, detail="This document is too large to read.")
        root = ElementTree.fromstring(archive.read(info))

    paragraphs = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{WORD_NS}t":
                parts.append(node.text or "")
            elif node.tag == f"{WORD_NS}tab":
                parts.append("\t")
            elif node.tag in (f"{WORD_NS}br", f"{WORD_NS}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


READERS = {".pdf": read_pdf, ".docx": read_docx}


@app.post("/api/extract")
async def extract(request: Request, name: str = Query(min_length=1, max_length=255)):
    """Receive one file's raw bytes and return the text inside it."""
    check_access(request, extract_limiter)

    declared = int(request.headers.get("content-length") or 0)
    if declared > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Too big (max 4 MB)")
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Too big (max 4 MB)")

    reader = READERS.get(Path(name).suffix.lower())
    if reader is None:
        raise HTTPException(status_code=415, detail="Only PDF and .docx are read here")

    try:
        text = reader(data).strip()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail="Couldn't read this file")

    if not text:
        raise HTTPException(status_code=422, detail="No text found (scanned PDF?)")

    return {
        "name": name,
        "text": text[:MAX_FILE_CHARS],
        "chars": len(text),
        "truncated": len(text) > MAX_FILE_CHARS,
    }


# The chat page. Declared AFTER the /api routes so those routes win.
# On Vercel these files are promoted to the CDN automatically.
STATIC_DIR = Path(__file__).parent / "static"  # works from any working folder
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
