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
from concurrent.futures import ThreadPoolExecutor
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

import stock_tools  # noqa: E402  (imported after load_dotenv so it can read TAVILY_API_KEY)

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

STOCK_PROMPT = """

## Stock research
You are also a stock market research assistant. When the user asks about a company or stock:
1. If you are not sure of the exact ticker, call search_symbol first. Indian stocks use .NS (NSE) or .BO (BSE).
2. For a general analysis or outlook, call analyze_stock ONCE: it returns quote, price trend, financials and news together. Use the single tools (get_stock_quote, get_price_history, get_financials, get_stock_news, web_search if available) only for narrow follow-up questions. Keep tool calls to a minimum.
3. Base every number on tool results. Never invent figures. If a tool fails, say what is missing.
4. Structure the answer with short headings:
   - Snapshot (price, valuation, market cap)
   - Trend & technicals (returns, moving averages, RSI, volatility)
   - Financial health (growth, margins, debt, cash flow)
   - News & sentiment
   - Bull case / Bear case (2-3 bullets each)
   - Outlook: a short-term lean (Bullish / Neutral / Bearish) with a confidence level (Low / Medium) and the 2-3 signals that drive it. Never claim High confidence or certainty.
   - Key things to watch
5. End with this exact line: "*This is research, not financial advice. Stock prices are unpredictable; do your own research or consult a registered advisor before investing.*"
Do not tell the user to buy or sell, and do not give position sizes or personalized portfolio advice. Mention the date of the data you used."""

MAX_TOOL_ROUNDS = 6   # how many times the model may call tools for one question
RATE_LIMIT_MAX_WAIT = 25  # seconds; Vercel stops a request after 60 s


def system_prompt() -> str:
    # Built per request so the date is always right.
    return SYSTEM_PROMPT + STOCK_PROMPT + f"\n\nToday's date is {stock_tools.today()}."

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


def start_stream(messages: list, tools_allowed: bool):
    """Ask Groq for a streamed reply, turning its errors into HTTP errors."""
    try:
        return client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.5, stream=True,
            tools=stock_tools.TOOLS,
            tool_choice="auto" if tools_allowed else "none",
        )
    except groq.AuthenticationError:
        raise HTTPException(status_code=500, detail="The server's Groq API key is invalid.")
    except groq.RateLimitError as e:
        retry = e.response.headers.get("retry-after", "") if e.response is not None else ""
        try:
            seconds = max(1, int(float(retry) + 0.999))
        except ValueError:
            seconds = 60
        raise HTTPException(status_code=429, headers={"Retry-After": str(seconds)},
                            detail=f"Nova is busy (Groq free-tier limit). Try again in {seconds} s.")
    except groq.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach Groq.")
    except groq.BadRequestError as e:
        # e.g. the model wrote a malformed tool call ("tool_use_failed")
        raise HTTPException(status_code=502, detail=f"Nova got confused calling a tool. Try rephrasing. ({e.status_code})")
    except groq.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Groq error {e.status_code}.")


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request):
    check_access(request, chat_limiter)
    if client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on the server.")

    # Same as chatbot.py: system prompt first, then the (trimmed) history.
    history = [m.model_dump() for m in body.messages][-MAX_MESSAGES:]
    messages = [{"role": "system", "content": system_prompt()}] + history

    # Start the stream BEFORE returning, so key/rate-limit errors become
    # proper HTTP errors instead of a half-finished reply.
    stream = start_stream(messages, tools_allowed=True)

    def generate():
        # The tool loop:
        #   1. Stream the model's reply. Text goes straight to the browser.
        #   2. If the model asked for tools instead, run them, add the results
        #      to `messages`, and ask the model again with that new data.
        #   3. Repeat until it answers in plain text (or we hit the limit).
        # Tool activity is sent as "\x02label\x03" markers; the page shows them
        # as small status chips and leaves them out of the reply text.
        nonlocal stream
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            calls: dict[int, dict] = {}
            text = ""
            reasoning = ""
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # gpt-oss thinks before calling tools. We don't show this, but
                    # it must be sent back with the tool calls, or the model can
                    # lose its train of thought and return an empty answer.
                    reasoning += getattr(delta, "reasoning", None) or ""
                    if delta.content:
                        text += delta.content
                        yield delta.content
                    # Tool calls can arrive in pieces; glue them together by index.
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.id or slot["id"]
                        if tc.function:
                            slot["name"] += tc.function.name or ""
                            slot["args"] += tc.function.arguments or ""
            except groq.APIError:
                yield "\n\n[Connection to Groq was interrupted. Please try again.]"
                return

            if not calls:
                return  # a normal text answer: done
            if round_no >= MAX_TOOL_ROUNDS:
                yield "\n\n[Nova needed too many lookups for this question. Try asking something narrower.]"
                return

            ordered = [calls[i] for i in sorted(calls)]
            for c in ordered:
                yield f"\x02{stock_tools.label_for(c['name'], c['args'])}\x03"
            assistant_msg = {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                               for c in ordered],
            }
            if reasoning:
                assistant_msg["reasoning"] = reasoning
            messages.append(assistant_msg)
            # Independent tools run at the same time to save waiting.
            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(lambda c: stock_tools.run_tool(c["name"], c["args"]), ordered))
            for c, result in zip(ordered, results):
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

            # On the last round the model must answer with what it has.
            last = round_no >= MAX_TOOL_ROUNDS - 1
            try:
                stream = start_stream(messages, tools_allowed=not last)
            except HTTPException as e:
                # Groq's free tier allows ~8,000 tokens per minute. If the limit
                # frees up soon, wait once instead of throwing the research away.
                wait = int(e.headers.get("Retry-After", "0")) if e.headers else 0
                if e.status_code != 429 or not 0 < wait <= RATE_LIMIT_MAX_WAIT:
                    yield f"\n\n[{e.detail}]"
                    return
                yield f"\x02Waiting {wait}s for Groq's free-tier limit\x03"
                time.sleep(wait)
                try:
                    stream = start_stream(messages, tools_allowed=not last)
                except HTTPException as e2:
                    yield f"\n\n[{e2.detail}]"
                    return

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
