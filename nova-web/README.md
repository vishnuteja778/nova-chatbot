# Nova (web)

A streaming AI chat app with **tool calling, RAG and a general-purpose multi-agent team**: FastAPI backend + plain HTML/CSS/JS frontend, using Groq's free tier.

📘 **Learning guide (PDF):** [docs/guides/nova-agents-learning-guide.pdf](../docs/guides/nova-agents-learning-guide.pdf)

## Agents, tools and RAG

The team works on any substantial task: questions, research, analysis, comparisons, plans, reports and code.

```
request ─► PLANNER ── "direct" ──► single Nova (quick replies)
               │ "team"
               ▼
   ANALYST (live data, calculator)  +  RESEARCHER (RAG over library + your docs)   in parallel, only if needed
               └──────────────── notes ────────────────┘
                                 ▼
                 WRITER (draft, streamed) ─► REVIEWER (approve / revise once)
```

| Agent | Groq model | Role and tools |
| --- | --- | --- |
| Planner | `qwen/qwen3.8-27b` | Breaks the task down and picks the team (JSON decision) |
| Analyst | `openai/gpt-oss-120b` | Live data and exact numbers: `calculate`, market-data tools (`analyze_stock`, quote, history, financials, news, symbol search), optional `web_search` |
| Researcher | `qwen/qwen3.8-27b` | Evidence from documents: `search_documents` (hybrid BM25 + vector RAG), optional `web_search` |
| Writer | `GROQ_MODEL` (`openai/gpt-oss-20b`) | Writes the answer: explanation, report, plan or code |
| Reviewer | `openai/gpt-oss-120b` | Checks for errors, bugs, gaps and unsupported claims (JSON verdict) |

Each Groq model has its own free 8,000 tokens/minute budget, so spreading agents across three models lets a team run fit the free tier. The chat shows each step as a chip ("Analyst → calculating 12000 * 12"). Answers cite `[D1]` for document passages and `[Live data]` for fetched data, with a Sources list.

**RAG:** a built-in knowledge library in [knowledge/](knowledge/) (7 reference documents, 60 chunks, currently on business and markets) plus your own documents on any topic from the **Library** panel. Documents are chunked, embedded locally with BGE-small (fastembed, free), stored as int8 vectors, and searched with BM25 + vectors fused by Reciprocal Rank Fusion. Retrieval on 16 test questions: BM25 88%, vector 88%, **hybrid 94% Hit@4** (`python eval_rag.py`).

```
nova-web/
├── app.py            FastAPI routes: /api/chat, /api/library, /api/extract, /api/health
├── agents.py         Planner, Analyst, Researcher, Writer, Reviewer; calculator tool; team pipeline
├── stock_tools.py    market-data tools (Yahoo Finance via yfinance)
├── rag.py            chunking, BM25, embeddings, hybrid search
├── build_index.py    builds knowledge_index.json from knowledge/
├── eval_rag.py       retrieval evaluation (Hit@4, MRR)
├── knowledge/        the built-in finance library (add your own .md / .pdf)
├── static/
│   ├── index.html    the chat page
│   ├── style.css     light + dark theme, mobile layout
│   └── app.js        keeps the history, streams replies, Stop / New chat
├── requirements.txt  Python packages Vercel installs
├── vercel.json       lets a reply run for up to 60 seconds
├── .vercelignore     keeps .env and .venv out of uploads
└── .env.example      settings template
```

## How it works

1. The browser keeps the whole conversation in a list (`history` in `app.js`).
2. Each message sends the **whole history** to `POST /api/chat`.
3. The server adds Nova's system prompt, calls Groq with `stream=True`,
   and forwards each piece to the browser as it arrives.
4. The browser shows the pieces as they come and saves the full reply to `history`.

The server keeps no memory between requests. That is why the browser sends the history every time,
just like `messages` in `chatbot.py`.

## File uploads

Attach files with the paperclip, by dragging them anywhere onto the page, or by pasting.
Up to 5 files per message, 4 MB each.

| File type | Read by | How |
| --- | --- | --- |
| `.txt`, `.md`, `.csv`, `.json`, code files | the browser | `file.text()` |
| `.pdf` | the server (`/api/extract`) | `pypdf`, first 50 pages |
| `.docx` | the server (`/api/extract`) | unzips `word/document.xml` with the standard library |
| images | not yet | needs a vision model |

Each file's text (up to 15,000 characters, about 4,000 tokens) is placed in the message as
`<file name="...">...</file>`, followed by your question. The model reads it like any other text.
Files stay in the conversation history, so follow-up questions can refer to them, but they also
count toward your token usage on every turn.

## Run locally

```powershell
.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

Then open http://localhost:8000. Settings come from `.env` (copy `.env.example`).

## Deploy to Vercel

You need a free Vercel account (vercel.com, sign up with GitHub or Google).
The commands use `npx`, which comes with Node.js, so nothing extra to install.

```powershell
npx vercel@latest login
npx vercel@latest
```

The second command asks a few questions. Accept the defaults, and name the project `nova`.
It uploads the code and gives you a preview URL. The first preview has no API key yet, so add the settings:

```powershell
npx vercel@latest env add GROQ_API_KEY
npx vercel@latest env add GROQ_MODEL
npx vercel@latest env add ACCESS_CODE
```

Each command asks for the value (paste it into the terminal, not into any file) and which
environments to use. Choose Production, Preview and Development. Use `openai/gpt-oss-20b` for
`GROQ_MODEL`, and make up a password for `ACCESS_CODE`.

Then deploy to production:

```powershell
npx vercel@latest --prod
```

Open the URL it prints, type the access code once, and chat.

## Settings

| Variable | Required | What it does |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Your Groq key. Only the server sees it. |
| `GROQ_MODEL` | no | Model name. Default `openai/gpt-oss-20b`. |
| `ACCESS_CODE` | no, but recommended | Password visitors must enter. Without it, anyone with the URL uses your Groq quota. |
| `PLANNER_MODEL`, `ANALYST_MODEL`, `RESEARCHER_MODEL`, `REVIEWER_MODEL` | no | Override the Groq model for an agent. |
| `TAVILY_API_KEY` | no | Turns on the `web_search` tool for the specialists. |
| `NOVA_EMBEDDINGS` | no | Set to `off` to use keyword (BM25) search only. |

## Safety built in

- The API key stays on the server. The browser never sees it.
- Visitors can only send `user` and `assistant` messages, so they cannot replace Nova's system prompt.
- Messages are limited to 50,000 characters (attached file text included), and only the last 20 messages are sent to the model.
- Uploads are capped at 4 MB, PDFs at 50 pages, and Word files are checked against "zip bombs".
- Model output is HTML-escaped before display, so a reply cannot inject code into the page.
