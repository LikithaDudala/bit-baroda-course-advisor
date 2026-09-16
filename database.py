"""
Database module - call logging (the CRM)
=========================================
Uses SQLite, which is a real database stored as a single file (calls.db).
It's built into Python - no installation, no server needed.

We log every interaction: what the caller asked, what course was discussed,
the answer given, sentiment, and whether follow-up is needed.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "calls.db")


def init_db():
    """Create the calls table if it doesn't exist. Safe to run every startup."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            caller_question TEXT,
            course_discussed TEXT,
            answer_given TEXT,
            follow_up_needed INTEGER DEFAULT 0,
            call_type TEXT DEFAULT 'inbound'
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database ready (calls.db)")


def log_call(caller_question, course_discussed, answer_given, follow_up_needed=False, call_type="inbound"):
    """Save one interaction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO call_logs (timestamp, caller_question, course_discussed, answer_given, follow_up_needed, call_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        caller_question or "",
        course_discussed or "",
        answer_given or "",
        1 if follow_up_needed else 0,
        call_type,
    ))
    conn.commit()
    conn.close()
    print(f"[DB] Logged call: {caller_question[:50]!r}")


def get_all_calls():
    """Return all logged calls, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM call_logs ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats():
    """Return simple analytics for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM call_logs")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM call_logs WHERE follow_up_needed = 1")
    follow_ups = cur.fetchone()[0]

    # Most-discussed courses
    cur.execute("""
        SELECT course_discussed, COUNT(*) as count
        FROM call_logs
        WHERE course_discussed != ''
        GROUP BY course_discussed
        ORDER BY count DESC
        LIMIT 5
    """)
    top_courses = [{"course": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()
    return {
        "total_calls": total,
        "follow_ups_needed": follow_ups,
        "top_courses": top_courses,
    }


if __name__ == "__main__":
    # Quick self-test
    init_db()
    log_call("How much is the data science course?", "Data Science & AI Certification",
             "It's 8 months, around 75000 rupees.", follow_up_needed=True, call_type="inbound")
    print("Stats:", get_stats())
    print("All calls:", get_all_calls())
