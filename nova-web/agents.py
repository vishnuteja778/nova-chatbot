"""Nova's multi-agent team. It can work on any task: questions, research,
analysis, comparisons, plans, reports and code.

Each agent is a separate LLM call with its own role, instructions, tools and
Groq model:

    request
       │
       ▼
    PLANNER ─── "direct" ──► the normal single Nova (quick replies)
       │ "team"
       ▼
    ┌────────────────────────┬─────────────────────────────┐  run at the same
    │ ANALYST (if needed)    │ RESEARCHER (if needed)      │  time
    │ calculator, live data  │ search_documents (RAG),     │
    │ tools, web search      │ web search                  │
    └───────────┬────────────┴──────────────┬──────────────┘
                └────────── notes ──────────┘
                            ▼
                         WRITER   ──► draft (streamed to the user)
                            ▼
                         REVIEWER ──► approve, or list problems
                            ▼ (only if problems)
                         WRITER   ──► revised answer replaces the draft

Why different models? Groq's free tier gives EACH model its own budget of
about 8,000 tokens per minute. Spreading agents across models lets a
multi-step run fit inside the free tier.

The browser receives the answer as plain text plus small markers:
  label  a status chip ("Researcher → searching documents: ...")
             "clear the reply": the revised answer replaces the draft
"""
import ast
import json
import math
import operator
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import groq

import rag
import stock_tools


def _model(env: str, default: str) -> str:
    return os.getenv(env, "").strip() or default


MODELS = {
    "planner": _model("PLANNER_MODEL", "qwen/qwen3.8-27b"),
    "analyst": _model("ANALYST_MODEL", "openai/gpt-oss-120b"),
    "researcher": _model("RESEARCHER_MODEL", "qwen/qwen3.8-27b"),
    "reviewer": _model("REVIEWER_MODEL", "openai/gpt-oss-120b"),
    # the writer uses GROQ_MODEL, the same model as normal Nova
}
NAMES = {"planner": "Planner", "analyst": "Analyst", "researcher": "Researcher",
         "writer": "Writer", "reviewer": "Reviewer"}

# Longest reply each agent may write. IMPORTANT on Groq's free tier: the
# per-minute token limit counts prompt + max_tokens BEFORE the call runs. If
# max_tokens is left at a model's default (which can be huge), a request can be
# refused instantly even when almost no tokens have been used.
MAX_TOKENS = {"planner": 500, "analyst": 1500, "researcher": 1500, "writer": 2000, "reviewer": 1500}
# gpt-oss models "think" before answering, and that hidden reasoning uses the
# same max_tokens allowance. For short JSON replies we ask for little thinking,
# or the reply can be cut off before the JSON is written.
LOW_REASONING = {"planner", "reviewer"}

TIME_BUDGET = 48        # seconds; Vercel stops the request at 60
MAX_RATE_WAIT = 20      # longest single wait for Groq's per-minute limit
# Tool-call rounds per specialist. Prompts ask for few searches, but a hard
# limit in code is what actually guarantees it.
SPECIALIST_ROUNDS = {"analyst": 4, "researcher": 2}
SPECIALIST_RESERVE = 24 # stop gathering when fewer seconds remain (Writer + Reviewer need them)
CALL_TIMEOUT = 20       # seconds for any single Groq call


class Stop(Exception):
    """Raised to abandon the run with a message for the user."""


class Run:
    """Shared state for one question: the event queue, the deadline, sources."""

    def __init__(self, client: groq.Groq, uploads: list[rag.Chunk]):
        self.client = client
        self.uploads = uploads
        self.events: queue.Queue = queue.Queue()
        self.deadline = time.monotonic() + TIME_BUDGET
        self.sources: dict[str, dict] = {}   # "D1" -> chunk info, shared by all agents
        self.lock = threading.Lock()

    def status(self, text: str) -> None:
        self.events.put(("status", text))

    def text(self, piece: str) -> None:
        self.events.put(("text", piece))

    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    # --- one Groq call, with rate-limit handling ---
    def call(self, agent: str, model: str, **kwargs):
        kwargs.setdefault("max_tokens", MAX_TOKENS[agent])
        kwargs.setdefault("timeout", CALL_TIMEOUT)
        if self.time_left() < 4:
            raise Stop("Nova ran out of time on this question. Try asking something narrower.")
        if agent in LOW_REASONING and model.startswith("openai/gpt-oss"):
            kwargs.setdefault("reasoning_effort", "low")
        for attempt in range(2):
            try:
                return self.client.chat.completions.create(model=model, **kwargs)
            except groq.RateLimitError as e:
                print(f"[agents] {agent} ({model}) rate limited: {str(e)[:300]}")
                retry = e.response.headers.get("retry-after", "") if e.response is not None else ""
                try:
                    wait = max(1.0, float(retry))
                except ValueError:
                    wait = 30.0
                if attempt or wait > MAX_RATE_WAIT or wait > self.time_left() - 8:
                    raise Stop("Nova is busy (Groq free-tier limit). Try again in a minute.")
                self.status(f"{NAMES[agent]}: waiting {wait:.0f}s for Groq's limit")
                time.sleep(wait)
            except groq.BadRequestError as e:
                print(f"[agents] {agent} ({model}) bad request: {str(e)[:300]}")
                if attempt:
                    raise
                # Usually a malformed tool call by the model; one retry often fixes it.
            except groq.APIConnectionError:
                raise Stop("Could not reach Groq.")
        raise Stop("Nova couldn't complete this request. Try rephrasing.")

    # --- the RAG tool, bound to this request's uploads ---
    def search_documents(self, query: str, scope: str = "all") -> dict:
        query = str(query or "").strip()[:300]
        if not query:
            return {"error": "Give a search query."}
        scope = scope if scope in ("all", "library", "uploads") else "all"
        results = rag.search(query, self.uploads, scope=scope)
        out = []
        with self.lock:
            for chunk, _ in results:
                key = f"{chunk.source}|{chunk.section}|{chunk.text[:60]}"
                ref = next((r for r, s in self.sources.items() if s["key"] == key), None)
                if ref is None:
                    ref = f"D{len(self.sources) + 1}"
                    self.sources[ref] = {"key": key, "source": chunk.source,
                                         "section": chunk.section, "origin": chunk.origin}
                out.append({"ref": ref, "source": chunk.source, "section": chunk.section,
                            "from": "user's document" if chunk.origin == "upload" else "Nova library",
                            "text": chunk.text})
        if not out:
            return {"query": query, "results": [], "note": "Nothing relevant found."}
        return {"query": query, "results": out}


# ---------- tools per agent ----------

SEARCH_DOCS_TOOL = {"type": "function", "function": {
    "name": "search_documents",
    "description": ("Search Nova's knowledge library and the user's uploaded documents (reports, "
                    "notes, manuals, papers - anything they added). Returns numbered passages "
                    "(D1, D2...) to cite. Use short, specific queries; search several times "
                    "for different sub-questions."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["all", "library", "uploads"],
                  "description": "'uploads' = only the user's documents. Default 'all'."},
    }, "required": ["query"]},
}}

CALCULATE_TOOL = {"type": "function", "function": {
    "name": "calculate",
    "description": ("Evaluate an arithmetic expression exactly, e.g. '(1240 - 946) / 946 * 100' or "
                    "'5000 * (1 + 0.08) ** 10'. Supports + - * / // % ** and parentheses, plus "
                    "round, abs, min, max, sqrt, log. Use it for every calculation instead of "
                    "doing maths in your head."),
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string"},
    }, "required": ["expression"]},
}}

WEB_TOOL = [t for t in stock_tools.TOOLS if t["function"]["name"] == "web_search"]
MARKET_DATA_TOOLS = [t for t in stock_tools.TOOLS if t["function"]["name"] != "web_search"]


def tools_for(agent: str) -> list[dict]:
    if agent == "analyst":
        return [CALCULATE_TOOL] + MARKET_DATA_TOOLS + WEB_TOOL
    if agent == "researcher":
        return [SEARCH_DOCS_TOOL] + WEB_TOOL
    return []


# A safe calculator: we parse the expression into a syntax tree and allow only
# numbers, arithmetic operators and a few maths functions. Never use eval() on
# text a model (or user) wrote: it could run any Python code.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
          "log": math.log, "log10": math.log10, "exp": math.exp}


def calculate(expression: str) -> dict:
    expression = str(expression or "").replace("^", "**").replace(",", "")[:200]

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("only numbers, + - * / // % ** ( ) and round/abs/min/max/sqrt/log are allowed")

    try:
        value = ev(ast.parse(expression, mode="eval"))
        if isinstance(value, float):
            value = round(value, 6)
        return {"expression": expression, "result": value}
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return {"expression": expression, "error": str(e) or type(e).__name__}


def run_tool(run: Run, name: str, args_text: str) -> str:
    if name == "calculate":
        try:
            return json.dumps(calculate(json.loads(args_text or "{}").get("expression", "")))
        except (json.JSONDecodeError, AttributeError):
            return json.dumps({"error": "Arguments must be JSON with an 'expression'."})
    if name == "search_documents":
        try:
            args = json.loads(args_text or "{}")
            result = run.search_documents(**{k: v for k, v in args.items() if k in ("query", "scope")})
        except (json.JSONDecodeError, TypeError, AttributeError):
            result = {"error": "Arguments must be JSON with a 'query'."}
        text = json.dumps(result, ensure_ascii=False)
        return text[:stock_tools.MAX_RESULT_CHARS]
    return stock_tools.run_tool(name, args_text)


# ---------- the specialists ----------

SPECIALIST_PROMPTS = {
    "analyst": (
        "You are the Analyst in a team of AI agents. You handle anything that needs live data or "
        "exact numbers, and report facts; another agent writes the final answer. Use 'calculate' "
        "for every calculation (percentages, growth, comparisons, budgets, conversions) instead of "
        "mental maths. For live market data on a company, call analyze_stock ONCE per company "
        "(it already includes quote, trend, financials and news) and, when comparing, call it for "
        "all of them in the SAME turn; use search_symbol only when unsure of a ticker (Indian "
        "stocks: .NS / .BO). Use web_search, if available, for other current facts. "
        "Never invent numbers; if no tool can get something, say so."
    ),
    "researcher": (
        "You are the Researcher in a team of AI agents. You find evidence and background on any "
        "topic in documents; another agent writes the final answer. Use search_documents with "
        "short, specific queries: at most 3 searches, ideally sent together in one turn. If the "
        "user has uploaded documents and the task is about them, search scope 'uploads'. Nova's "
        "built-in library holds general reference material (concepts, rules, metrics), not news or "
        "facts about specific companies; if searches return nothing relevant, stop and say so. "
        "Use web_search, if available, for topics the documents don't cover. Only report what "
        "the sources say."
    ),
}
NOTES_FORMAT = {
    "analyst": (
        "\n\nWhen you have enough, reply with RESEARCH NOTES only: at most 180 words of bullet points, "
        "the key numbers with their dates and the results of your calculations. End every bullet "
        "that uses fetched data with [Live data]. Never use D1/D2-style labels: those are reserved "
        "for document passages. If something could not be found, say so."
    ),
    "researcher": (
        "\n\nWhen you have enough, reply with RESEARCH NOTES only: at most 180 words of bullet points. "
        "End every bullet with the ref of the passage it comes from, exactly as given in the search "
        "results, e.g. [D3]. If something could not be found, say so in one bullet."
    ),
}


def run_specialist(run: Run, agent: str, task: str) -> str:
    """A tool-using agent: think, call tools, read results, repeat, then report."""
    name = NAMES[agent]
    model = MODELS[agent]
    tools = tools_for(agent)
    messages = [
        {"role": "system", "content": SPECIALIST_PROMPTS[agent] + NOTES_FORMAT[agent]
         + f"\nToday's date is {stock_tools.today()}."},
        {"role": "user", "content": task},
    ]
    gathered: list[str] = []   # raw tool results, kept in case the final summary fails
    max_rounds = SPECIALIST_ROUNDS[agent]
    for round_no in range(max_rounds + 1):
        # Leave enough time for the Writer and Reviewer after the specialists.
        final_round = round_no == max_rounds or run.time_left() < SPECIALIST_RESERVE
        if final_round:
            # Don't offer tools at all: some models call a tool even when told
            # tool_choice="none", and Groq then rejects the whole request.
            return write_notes(run, agent, model, messages, gathered)
        try:
            resp = run.call(agent, model, messages=messages, temperature=0.2, tools=tools)
        except groq.BadRequestError:
            return write_notes(run, agent, model, messages, gathered)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip() or "(no findings)"
        calls = msg.tool_calls
        assistant = {"role": "assistant", "content": msg.content or None, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
            for c in calls]}
        # gpt-oss models need their reasoning back alongside their tool calls.
        if getattr(msg, "reasoning", None):
            assistant["reasoning"] = msg.reasoning
        messages.append(assistant)
        for c in calls:
            run.status(f"{name} → {tool_label(c.function.name, c.function.arguments)}")
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda c: run_tool(run, c.function.name, c.function.arguments), calls))
        for c, result in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            gathered.append(f"{c.function.name}: {result}")
    return write_notes(run, agent, model, messages, gathered)


def write_notes(run: Run, agent: str, model: str, messages: list, gathered: list[str]) -> str:
    """Ask the specialist to summarise what it has, with no tools offered.

    If even that fails, hand the raw tool results to the Writer, trimmed:
    data that was already fetched should never be thrown away."""
    if not gathered:
        return "(no findings)"
    try:
        resp = run.call(agent, model, temperature=0.2, messages=messages + [{
            "role": "user",
            "content": "Stop gathering. Write your RESEARCH NOTES now from the results above, "
                       "in the format you were given. Do not call any tools."}])
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except groq.APIError as e:
        print(f"[agents] {agent} notes fallback: {type(e).__name__}")
    raw = "\n".join(gathered)[:3000]
    tag = "[Live data]" if agent == "analyst" else "the refs shown"
    return f"(raw results; cite them as {tag})\n{raw}"


def tool_label(name: str, args_text: str) -> str:
    if name == "search_documents":
        return f"searching documents: {_arg(args_text, 'query')}"
    if name == "calculate":
        return f"calculating {_arg(args_text, 'expression')}"
    return stock_tools.label_for(name, args_text)


def _arg(args_text: str, key: str) -> str:
    try:
        return str(json.loads(args_text or "{}").get(key, ""))[:50]
    except json.JSONDecodeError:
        return ""


# ---------- the planner ----------

PLANNER_PROMPT = """You are the Planner of Nova, a team of AI agents that can help with any task: questions, explanations, research, analysis, comparisons, planning, writing and code. Decide how to handle the user's latest message.

The team:
- "analyst": gets live data and exact numbers with tools - a calculator for any maths, live stock-market data for named companies or indices, and web search when available.
- "researcher": finds evidence in documents - Nova's knowledge library (currently strongest on business and markets: valuation, financial statements, technical indicators, Indian market rules, earnings, risk, sector metrics) and the user's uploaded documents{uploads}.
- The Writer then produces the answer (explanation, report, plan, code...) and the Reviewer checks it for errors and gaps. They are always included in "team" mode, so you never assign them tasks.

Modes:
- "direct": quick replies that need no checking - greetings, small talk, short factual answers, simple rewrites, or questions fully answered by text pasted in the message.
- "team": anything substantial - multi-part questions, analysis, comparisons, plans, reports, non-trivial code or debugging, calculations, anything needing live data or documents. Assign the analyst and/or researcher only if their tools help; "team" with no tasks means the Writer answers and the Reviewer checks. Company prices, financials and news come from the analyst's tools, so don't ask the researcher for them; use the researcher for concepts, explanations, rules and the user's documents.

Reply with JSON only:
{{"mode": "direct" | "team", "tasks": [{{"agent": "analyst" | "researcher", "task": "<clear, specific instruction: what to find or compute>"}}]}}
Use at most one task per agent; one task can cover several items. Use no tasks in "direct" mode."""


def plan(run: Run, question: str, history: list[dict], upload_names: list[str]) -> dict:
    uploads = (f". The user has uploaded: {', '.join(upload_names[:10])}" if upload_names
               else ". The user has not uploaded any documents")
    recent = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-4:])
    resp = run.call("planner", MODELS["planner"], temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": PLANNER_PROMPT.format(uploads=uploads)},
                              {"role": "user", "content": f"Recent conversation:\n{recent}\n\n"
                                                          f"Latest message:\n{question[:3000]}"}])
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"mode": "direct", "tasks": []}
    tasks = []
    for t in data.get("tasks") or []:
        if isinstance(t, dict) and t.get("agent") in ("analyst", "researcher") and t.get("task"):
            if all(x["agent"] != t["agent"] for x in tasks):
                tasks.append({"agent": t["agent"], "task": str(t["task"])[:1000]})
    mode = "team" if data.get("mode") in ("team", "research") else "direct"
    return {"mode": mode, "tasks": tasks}


# ---------- the Writer and the Reviewer ----------

WRITER_RULES = """

## You are the Writer of a team of AI agents
Below the user's request you get the TASK and, when teammates gathered any, their NOTES. Produce the best final answer to the user - an explanation, report, comparison, plan, code or whatever the task needs:
- When notes are given, use them for every number, date and fact about data or documents. If the notes lack something, say it wasn't available rather than guessing.
- Keep citations exactly as the notes give them, in plain square brackets, right after the claim they support: [D1], [D2] for document passages and [Live data] for fetched live data. Never label live data with a D number.
- Without notes, answer from your own knowledge, carefully; for code, give complete, working code.
- Only for an analysis of a specific stock or company share, follow the Stock research section (snapshot, trend, financial health, news, bull/bear case, outlook with Low/Medium confidence, things to watch, and its disclaimer). Do not add that stock disclaimer to any other kind of answer.
- Do not mention the agents, the notes or this process."""

REVIEWER_PROMPT = """You are the Reviewer in a team of AI agents. Check the DRAFT answer to the user's TASK (and against the NOTES, if any).
Look only for real problems:
1. Factual errors, wrong calculations, or numbers and facts not supported by the notes.
2. Bugs or missing pieces in code.
3. Parts of the user's request that were not answered.
4. Claims about documents without a [D#] citation, or citations pointing to the wrong passage.
5. Overconfident predictions (certainty, guaranteed results) or direct buy/sell instructions.
Minor style issues do not count.
Reply with JSON only: {"verdict": "approve" | "revise", "issues": ["<specific problem and fix>", ...]}"""


def notes_block(question: str, findings: dict[str, str]) -> str:
    parts = [f"TASK (the user's request):\n{question[:4000]}"]
    if findings:
        parts.append("NOTES from teammates:")
        for agent, text in findings.items():
            parts.append(f"--- from {NAMES[agent]} ---\n{text}")
    else:
        parts.append("NOTES: none - no live data or documents were needed for this task.")
    return "\n\n".join(parts)


def stream_writer(run: Run, model: str, system: str, history: list[dict], content: str) -> str:
    """Stream the writer's answer to the browser, and return the full text."""
    messages = [{"role": "system", "content": system}] + history[:-1] + [{"role": "user", "content": content}]
    stream = run.call("writer", model, messages=messages, temperature=0.4, stream=True)
    text = ""
    try:
        for chunk in stream:
            piece = chunk.choices[0].delta.content if chunk.choices else None
            if piece:
                # Some models write citations as 【D1】 or hide zero-width
                # spaces inside them; normalise to a plain [D1].
                piece = piece.replace("【", "[").replace("】", "]").replace("​", "")
                text += piece
                run.text(piece)
    except groq.APIError:
        run.text("\n\n[Connection to Groq was interrupted.]")
    return text


def review_draft(run: Run, notes: str, draft: str) -> dict:
    """Returns {"verdict": "approve" | "revise" | "skipped", "issues": [...]}.

    The Reviewer is a safety net, not a gate: if it fails, the draft stands."""
    try:
        resp = run.call("reviewer", MODELS["reviewer"], temperature=0, response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": REVIEWER_PROMPT},
                                  {"role": "user", "content": f"{notes}\n\nDRAFT ANSWER:\n{draft}"}])
        data = json.loads(resp.choices[0].message.content or "{}")
    except Stop:
        return {"verdict": "skipped", "issues": []}
    except (groq.APIError, json.JSONDecodeError) as e:
        print(f"[agents] reviewer skipped: {type(e).__name__}")
        return {"verdict": "skipped", "issues": []}
    issues = [str(i)[:300] for i in (data.get("issues") or []) if str(i).strip()][:5]
    verdict = "revise" if data.get("verdict") == "revise" and issues else "approve"
    return {"verdict": verdict, "issues": issues}


def sources_footer(run: Run, text: str) -> str:
    """List the documents the answer actually cited."""
    used = [r for r in run.sources if f"[{r}]" in text]
    if not used:
        return ""
    lines = []
    for r in used:
        s = run.sources[r]
        where = "your document" if s["origin"] == "upload" else "Nova library"
        lines.append(f"- [{r}] {s['source']} > {s['section']} ({where})")
    return "\n\n**Sources**\n" + "\n".join(lines)


# ---------- the whole team pipeline ----------

def team(run: Run, tasks: list[dict], question: str, history: list[dict],
         writer_model: str, writer_system: str) -> None:
    run.status("Planner: " + " + ".join([NAMES[t["agent"]] for t in tasks] + ["Writer", "Reviewer"]))

    # 1. Specialists (if any) work in parallel (different models, different budgets).
    findings: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        futures = {t["agent"]: pool.submit(run_specialist, run, t["agent"], t["task"]) for t in tasks}
        for agent, future in futures.items():
            try:
                findings[agent] = future.result()
            except Stop as e:
                findings[agent] = f"(this agent could not finish: {e})"
            except Exception as e:
                print(f"[agents] {agent} failed: {type(e).__name__}: {str(e)[:300]}")
                findings[agent] = f"(this agent failed: {type(e).__name__})"
    run.status("Writer: drafting the answer")

    # 2. The writer streams a draft.
    notes = notes_block(question, findings)
    system = writer_system + WRITER_RULES
    draft = stream_writer(run, writer_model, system, history, notes)

    # 3. The Reviewer checks it (skipped if we're short on time).
    if run.time_left() < 15 or not draft.strip():
        run.text(sources_footer(run, draft))
        return
    run.status("Reviewer: checking the answer")
    review = review_draft(run, notes, draft)
    if review["verdict"] != "revise":
        run.status("Reviewer: approved" if review["verdict"] == "approve" else "Reviewer: skipped")
        run.text(sources_footer(run, draft))
        return

    # 4. One revision, which replaces the draft on screen.
    run.status(f"Reviewer: {len(review['issues'])} fix{'es' if len(review['issues']) > 1 else ''} requested")
    if run.time_left() < 12:
        run.text(sources_footer(run, draft))
        return
    run.events.put(("reset", None))
    fixes = "\n".join(f"- {i}" for i in review["issues"])
    revised = stream_writer(run, writer_model, system, history,
                            f"{notes}\n\nYOUR PREVIOUS DRAFT:\n{draft}\n\n"
                            f"A reviewer found these problems. Rewrite the full answer fixing them:\n{fixes}")
    run.text(sources_footer(run, revised))


def stream_events(run: Run, work) -> "Iterator[str]":
    """Run `work` in a thread and turn its events into the text stream for the browser."""
    done = object()

    def target():
        try:
            work()
        except Stop as e:
            run.text(f"\n\n[{e}]")
        except Exception as e:
            run.text(f"\n\n[Something went wrong in the agents ({type(e).__name__}). Please try again.]")
        finally:
            run.events.put(done)

    threading.Thread(target=target, daemon=True).start()
    while True:
        item = run.events.get()
        if item is done:
            return
        kind, value = item
        if kind == "status":
            yield f"\x02{value}\x03"
        elif kind == "reset":
            yield "\x04"
        elif value:
            yield value
