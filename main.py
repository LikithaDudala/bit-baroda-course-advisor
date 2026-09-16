"""
BIT Baroda Course Advisor Voice Agent - Backend (FastAPI, Groq)
================================================================
Full system with CRM call logging (Phase 2):
- RAG over BIT Baroda course catalog (Gemini embeddings + Groq Llama)
- /vapi-webhook : answers questions during LIVE calls + LOGS each to database
- /start-campaign, /call-one : outbound calling
- /calls : view all logged calls (the CRM)
- /stats : analytics summary

Run with:  python -m uvicorn main:app --reload --port 5000
(Keep VPN connected to US so Groq is reachable.)
"""

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
import requests
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

from rag_engine import search
from database import init_db, log_call, get_all_calls, get_stats

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")

app = FastAPI(title="BIT Baroda Course Advisor")

# Initialize the database when the server starts
init_db()

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
LEADS_PATH = os.path.join(os.path.dirname(__file__), "leads.json")

def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
knowledge_base = load_knowledge_base()


SYSTEM_PROMPT = """You are Riya, a friendly course advisor for BIT Baroda (Baroda Institute of Technology),
an IT training institute in Vadodara. You are speaking on a PHONE call, so keep answers SHORT
and natural - 2 to 3 sentences max.

CRITICAL RULES:
- Answer ONLY using the course information provided in CONTEXT.
- If the specific detail the caller wants is NOT in the context, do NOT make it up.
  Instead say you don't have that exact detail and offer to have a counsellor call them back
  or share the helpline number +91 9328994901.
- If asked about a course we don't offer, say so honestly and suggest the closest course we DO have.
- Be warm, encouraging, and helpful - you are guiding someone about their career and education.
- If they seem interested, gently encourage them to enroll or speak to a counsellor.
- Always be honest about fees being approximate and suggest confirming with a counsellor.
- Clarify you are an AI assistant if asked for personal advice."""


class AskRequest(BaseModel):
    question: str


def build_context(hits: list) -> str:
    return "\n".join(f"- {h['text']}" for h in hits)


def answer_question(question: str) -> dict:
    hits = search(question, top_k=3)
    context = build_context(hits)
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        temperature=0.4,
        max_tokens=150,
    )
    answer = resp.choices[0].message.content.strip()
    # The top-matched course (for logging)
    top_course = hits[0]["name"] if hits else ""
    # Simple heuristic: follow-up needed if we couldn't answer fully
    follow_up = "counsellor" in answer.lower() or "call you back" in answer.lower()
    return {
        "question": question,
        "answer": answer,
        "top_course": top_course,
        "follow_up": follow_up,
        "sources": [{"name": h["name"], "relevance": h["relevance"]} for h in hits],
    }


# ── Basic routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "running",
        "agent": "BIT Baroda Course Advisor",
        "courses_loaded": len(knowledge_base["courses"]),
        "faqs_loaded": len(knowledge_base["general_faqs"]),
        "time": datetime.utcnow().isoformat(),
    }


@app.get("/courses")
def list_courses():
    summary = [
        {"id": c["id"], "name": c["name"], "category": c["category"]}
        for c in knowledge_base["courses"]
    ]
    return {"count": len(summary), "courses": summary}


@app.get("/search")
def search_route(q: str, top_k: int = 3):
    try:
        return {"query": q, "results": search(q, top_k=top_k)}
    except Exception as e:
        return {"error": str(e), "hint": "Run 'python rag_engine.py' first."}


@app.get("/ask-test")
def ask_test(q: str):
    try:
        result = answer_question(q)
        # Log browser tests too (marked as 'test')
        log_call(q, result["top_course"], result["answer"], result["follow_up"], "test")
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer_question(req.question)
    except Exception as e:
        return {"error": str(e)}


# ── Vapi webhook for live voice calls (logs every call) ─────────────────────
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    body = await request.json()
    print("\n" + "=" * 60)
    print(f"[Vapi webhook] {datetime.utcnow().isoformat()}")
    print(json.dumps(body, indent=2)[:1000])
    print("=" * 60)

    msg = body.get("message", {})
    msg_type = msg.get("type", "")

    if msg_type == "tool-calls":
        tool_calls = msg.get("toolCalls", [])
        results = []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_id = tc["id"]
            print(f"[Tool] {fn_name} called with {args}")

            if fn_name == "search_courses":
                question = args.get("question", "")
                result = answer_question(question)
                # LOG THIS CALL to the database (the CRM)
                log_call(
                    caller_question=question,
                    course_discussed=result["top_course"],
                    answer_given=result["answer"],
                    follow_up_needed=result["follow_up"],
                    call_type="phone",
                )
                results.append({"toolCallId": tool_id, "result": result["answer"]})
            else:
                results.append({"toolCallId": tool_id, "result": "I'm not sure how to help with that."})

        return {"results": results}

    return {"status": "ok"}


# ── OUTBOUND calling ────────────────────────────────────────────────────────
@app.post("/start-campaign")
def start_campaign():
    if not all([VAPI_PRIVATE_KEY, VAPI_PHONE_NUMBER_ID, VAPI_ASSISTANT_ID]):
        return {"error": "Missing Vapi credentials in .env"}
    with open(LEADS_PATH, "r", encoding="utf-8") as f:
        leads = json.load(f)["leads"]
    results = []
    for lead in leads:
        number = lead["number"]
        name = lead.get("name", "there")
        if "XXXX" in number:
            results.append({"name": name, "number": number, "status": "skipped (placeholder)"})
            continue
        payload = {
            "assistantId": VAPI_ASSISTANT_ID,
            "phoneNumberId": VAPI_PHONE_NUMBER_ID,
            "customer": {"number": number, "numberE164CheckEnabled": False},
        }
        try:
            r = requests.post("https://api.vapi.ai/call/phone",
                headers={"Authorization": f"Bearer {VAPI_PRIVATE_KEY}", "Content-Type": "application/json"},
                json=payload, timeout=15)
            if r.status_code in (200, 201):
                results.append({"name": name, "number": number, "status": "calling", "call_id": r.json().get("id")})
            else:
                results.append({"name": name, "number": number, "status": f"failed: {r.status_code}", "detail": r.text[:200]})
        except Exception as e:
            results.append({"name": name, "number": number, "status": f"error: {e}"})
    return {"campaign": "BIT Baroda course outreach", "calls": results}


class CallRequest(BaseModel):
    number: str

@app.post("/call-one")
def call_one(req: CallRequest):
    if not all([VAPI_PRIVATE_KEY, VAPI_PHONE_NUMBER_ID, VAPI_ASSISTANT_ID]):
        return {"error": "Missing Vapi credentials in .env"}
    payload = {
        "assistantId": VAPI_ASSISTANT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": req.number, "numberE164CheckEnabled": False},
    }
    try:
        r = requests.post("https://api.vapi.ai/call/phone",
            headers={"Authorization": f"Bearer {VAPI_PRIVATE_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=15)
        return {"status": r.status_code, "response": r.json()}
    except Exception as e:
        return {"error": str(e)}


# ── CRM routes: view logged calls + stats ───────────────────────────────────
@app.get("/calls")
def calls():
    """View all logged calls (the CRM record)."""
    return {"calls": get_all_calls()}


@app.get("/stats")
def stats():
    """Analytics summary."""
    return get_stats()


@app.get("/dashboard")
def dashboard():
    """Serve the analytics dashboard webpage."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))


@app.get("/")
def root():
    return {
        "message": "BIT Baroda Course Advisor backend is running.",
        "try": ["/health", "/courses", "/ask-test?q=...", "/calls", "/stats"],
    }
