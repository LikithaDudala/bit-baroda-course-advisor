# BIT Baroda Course Advisor — AI Voice Agent

An AI-powered voice agent backend that answers prospective students' questions about course
offerings over live phone calls, runs outbound calling campaigns, and logs every interaction
to a lightweight CRM with an analytics dashboard.

## Problem

Course advisory calls for a training institute are repetitive but high-stakes: prospective
students ask about fees, duration, eligibility, and certification for dozens of courses, and
every missed or poorly-answered call is a potential lost enrollment. Staffing a counsellor for
every inbound/outbound call doesn't scale. This project automates first-line course advisory
over the phone using a retrieval-augmented LLM agent, while still logging every conversation
so human counsellors can follow up where needed.

## Key Features

- **Live voice Q&A** — integrates with [Vapi](https://vapi.ai) as a webhook-driven tool backend
  so a voice assistant can answer caller questions in real time during a phone call.
- **Retrieval-augmented answers (RAG)** — every answer is grounded in a structured course
  catalog via semantic search over Gemini embeddings, rather than the LLM improvising.
- **Outbound calling campaigns** — trigger automated outbound calls to a list of leads via the
  Vapi API.
- **Call logging / lightweight CRM** — every question, matched course, answer, and follow-up
  flag is persisted to SQLite for later review.
- **Analytics dashboard** — a static HTML dashboard (Chart.js) visualizing total calls,
  follow-ups needed, and most-discussed courses.
- **REST API** — FastAPI endpoints for health checks, course listing, ad-hoc search/ask,
  campaign triggering, and CRM/stats retrieval.

## Technical Stack

| Layer            | Technology                                             |
|-------------------|--------------------------------------------------------|
| API framework     | FastAPI + Uvicorn                                       |
| LLM inference     | Groq (Llama 3.3 70B Versatile)                          |
| Embeddings        | Google Gemini (`gemini-embedding-001`)                  |
| Vector search      | NumPy cosine similarity over a local JSON embedding store |
| Voice calling     | Vapi.ai (inbound webhook + outbound call API)            |
| Storage           | SQLite (call logs), JSON (course catalog, leads)        |
| Dashboard         | Static HTML/CSS/JS + Chart.js (served by FastAPI)        |

## System Architecture

```
                    ┌─────────────────┐
                    │  Caller (phone)  │
                    └────────┬─────────┘
                             │ voice call
                    ┌────────▼─────────┐
                    │   Vapi.ai         │  outbound campaigns ──┐
                    │ (voice assistant) │                       │
                    └────────┬─────────┘                        │
                             │ webhook (tool call)                │
                    ┌────────▼─────────────────────┐            │
                    │  FastAPI backend (main.py)     │◄───────────┘
                    │  /vapi-webhook  /start-campaign │
                    └────────┬───────────┬───────────┘
                             │           │
                 ┌───────────▼──┐   ┌────▼─────────┐
                 │ rag_engine.py │   │ database.py   │
                 │ Gemini embed  │   │ SQLite CRM    │
                 │ + Groq LLM    │   │ (calls.db)    │
                 └───────┬───────┘   └────┬──────────┘
                         │                │
              ┌──────────▼─────┐   ┌──────▼─────────────┐
              │knowledge_base   │   │ /dashboard          │
              │  .json (courses)│   │ static/dashboard.html│
              └─────────────────┘   └──────────────────────┘
```

1. A caller's question comes in through a live Vapi voice call as a `tool-calls` webhook event.
2. The backend embeds the question, retrieves the most relevant course/FAQ entries via cosine
   similarity, and asks Groq's Llama 3.3 model to answer using only that retrieved context.
3. The answer is returned to Vapi to be spoken back to the caller, and the full interaction is
   logged to SQLite.
4. The dashboard polls `/stats` and `/calls` to visualize call volume and follow-up needs.

## Dataset / Source Information

The course catalog ([knowledge_base.json](knowledge_base.json)) is a hand-authored dataset
describing 23 courses across 10 categories (Data Science & AI, Web Development, Cybersecurity,
Digital Marketing, Degree & Diploma Programs, etc.) plus 10 general FAQs, modeled on a real
IT training institute's public course offerings. It is included directly in the repository
since it's small, static, and contains no private data — it is the knowledge base the RAG
system retrieves from.

[leads.json](leads.json) is a **sample** outbound-calling lead list with placeholder phone
numbers (`+91XXXXXXXXXX`). Replace it with real lead data locally; it is not meant to carry
real personal data in this repository.

`calls.db` (SQLite) and `embeddings_store.json` (the Gemini embedding index) are **generated
at runtime** and intentionally excluded from version control — see [Setup](#installation--setup).

## Installation & Setup

### Prerequisites

- Python 3.10+
- API keys for [Google Gemini](https://aistudio.google.com/app/apikey) and
  [Groq](https://console.groq.com/keys)
- A [Vapi.ai](https://dashboard.vapi.ai) account (for live/outbound calling features)

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd EduAgent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env with your real API keys

# 4. Build the RAG embedding index (creates embeddings_store.json)
python rag_engine.py

# 5. Run the API server
python -m uvicorn main:app --reload --port 5000
```

The SQLite database (`calls.db`) is created automatically on first run.

### Environment Variables

See [.env.example](.env.example) for the full list:

| Variable                | Description                                              |
|--------------------------|------------------------------------------------------------|
| `GEMINI_API_KEY`         | Google Gemini API key, used to embed the course catalog and search queries |
| `GROQ_API_KEY`           | Groq API key, used for LLM inference (Llama 3.3 70B)      |
| `VAPI_PRIVATE_KEY`       | Vapi private API key, used for outbound call requests     |
| `VAPI_PHONE_NUMBER_ID`   | Vapi phone number ID to place calls from                  |
| `VAPI_ASSISTANT_ID`      | Vapi assistant ID configured for this agent                |

## Running the Project

Once the server is running (`http://localhost:5000`):

| Endpoint            | Method | Description                                      |
|----------------------|--------|----------------------------------------------------|
| `/`                  | GET    | Basic service info                                 |
| `/health`            | GET    | Health check + loaded course/FAQ counts             |
| `/courses`           | GET    | List all courses in the catalog                     |
| `/search?q=...`      | GET    | Raw semantic search over the knowledge base          |
| `/ask-test?q=...`    | GET    | Ask a question in the browser (also logs the call)   |
| `/ask`               | POST   | Ask a question programmatically (`{"question": "..."}`)|
| `/vapi-webhook`      | POST   | Webhook consumed by Vapi during live calls           |
| `/start-campaign`    | POST   | Trigger outbound calls to all leads in `leads.json`   |
| `/call-one`          | POST   | Trigger a single outbound call (`{"number": "+91..."}`)|
| `/calls`             | GET    | Full CRM call log                                    |
| `/stats`             | GET    | Aggregate analytics (totals, follow-ups, top courses) |
| `/dashboard`         | GET    | Analytics dashboard (HTML)                            |

To use live/outbound calling, configure a Vapi assistant with a `search_courses` tool that
points its webhook at `POST /vapi-webhook`.

## Main Implementation Details

- **`main.py`** — FastAPI app: request routing, the RAG-grounded `answer_question()` pipeline,
  the Vapi webhook handler, and outbound campaign logic.
- **`rag_engine.py`** — builds the embedding index from the course catalog (`build_index()`)
  and performs cosine-similarity search (`search()`) against it.
- **`database.py`** — SQLite schema and helper functions for logging and reading back call
  records (`log_call`, `get_all_calls`, `get_stats`).
- **`static/dashboard.html`** — self-contained dashboard that polls `/stats` and `/calls` and
  renders them with Chart.js, auto-refreshing every 10 seconds.

The assistant is instructed (via a system prompt) to answer **only** from retrieved course
context, and to hand off to a human counsellor rather than fabricate details it doesn't have.

## Limitations & Known Constraints

- The embedding index (`embeddings_store.json`) must be rebuilt (`python rag_engine.py`)
  whenever `knowledge_base.json` changes — it is not regenerated automatically.
- Groq's API may require a US-reachable connection depending on your region/network.
- `/start-campaign` and `/call-one` will place real outbound calls through Vapi if valid
  credentials and real phone numbers are configured — use placeholder numbers for testing.
- The CRM is a single local SQLite file with no authentication on its read endpoints
  (`/calls`, `/stats`, `/dashboard`); it is intended for local/internal use, not public
  deployment as-is.
- No automated test suite is currently included.

## Future Improvements

- Add authentication to CRM/dashboard endpoints before any public deployment.
- Persist the embedding index in a proper vector store instead of a flat JSON file.
- Add automated tests for the RAG retrieval and webhook handling logic.
