"""
RAG Engine (Gemini embeddings, no compiler needed)
===================================================
Smart meaning-based search over BIT Baroda's course catalog.

1. Turn each course into searchable text.
2. Use Gemini to convert text -> embeddings (meaning-numbers).
3. Save those numbers to embeddings_store.json.
4. To search: embed the question, compare with cosine similarity,
   return closest-matching courses.
"""

import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

KB_PATH    = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
STORE_PATH = os.path.join(os.path.dirname(__file__), "embeddings_store.json")

EMBED_MODEL = "models/gemini-embedding-001"


def course_to_text(c: dict) -> str:
    """Combine all fields of a course into one descriptive paragraph."""
    return (
        f"Course Name: {c['name']}. "
        f"Category: {c['category']}. "
        f"Description: {c['description']} "
        f"Duration: {c['duration']}. "
        f"Fee: {c['fee']}. "
        f"Eligibility: {c['eligibility']}. "
        f"Mode: {c['mode']}. "
        f"Certification: {c['certification']}. "
        f"Topics covered: {c['key_topics']}. "
        f"Career outcomes: {c['career_outcomes']}. "
        f"Batch timings: {c['batch_timings']}."
    )


def get_embedding(text: str) -> list:
    result = genai.embed_content(model=EMBED_MODEL, content=text)
    return result["embedding"]


def build_index():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    store = []

    print("Embedding courses...")
    for c in kb["courses"]:
        text = course_to_text(c)
        emb = get_embedding(text)
        store.append({
            "id": c["id"], "name": c["name"], "category": c["category"],
            "type": "course", "text": text, "embedding": emb,
        })
        print(f"  embedded {c['id']} - {c['name']}")

    print("Embedding FAQs...")
    for i, faq in enumerate(kb["general_faqs"]):
        text = f"FAQ: {faq['question']} Answer: {faq['answer']}"
        emb = get_embedding(text)
        store.append({
            "id": f"FAQ{i}", "name": faq["question"], "category": "FAQ",
            "type": "faq", "text": text, "embedding": emb,
        })
        print(f"  embedded FAQ{i}")

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)

    print(f"Index built. {len(store)} items saved to embeddings_store.json")
    return len(store)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, top_k: int = 3) -> list:
    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError("Index not built. Run: python rag_engine.py")

    with open(STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    query_emb = get_embedding(query)
    scored = [(cosine_similarity(query_emb, item["embedding"]), item) for item in store]
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = []
    for score, item in scored[:top_k]:
        hits.append({
            "name": item["name"], "category": item["category"],
            "type": item["type"], "text": item["text"],
            "relevance": round(score, 3),
        })
    return hits


if __name__ == "__main__":
    print("=" * 50)
    print("  Building RAG index for BIT Baroda courses")
    print("=" * 50)
    build_index()
    print("\nTesting a sample search...\n")
    for h in search("I want to learn data science"):
        print(f"  [{h['relevance']}] {h['name']} ({h['category']})")
