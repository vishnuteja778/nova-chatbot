# Nova (web)

A streaming AI chat app: FastAPI backend + plain HTML/CSS/JS frontend, using Groq.

```
nova-web/
├── app.py            FastAPI backend: /api/chat (streaming) and /api/health
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

## Safety built in

- The API key stays on the server. The browser never sees it.
- Visitors can only send `user` and `assistant` messages, so they cannot replace Nova's system prompt.
- Messages are limited to 50,000 characters (attached file text included), and only the last 20 messages are sent to the model.
- Uploads are capped at 4 MB, PDFs at 50 pages, and Word files are checked against "zip bombs".
- Model output is HTML-escaped before display, so a reply cannot inject code into the page.
