import sqlite3
from datetime import datetime

DB_NAME = "memory.db"
SUMMARY_TRIGGER = 20  # сколько сообщений до сжатия


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            user_id INTEGER PRIMARY KEY,
            content TEXT,
            updated TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_message(user_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def load_recent(user_id: int, limit: int = 6) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))

    rows = c.fetchall()
    conn.close()

    rows.reverse()
    return "\n".join(f"{r}: {c}" for r, c in rows)


def load_summary(user_id: int) -> str:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT content FROM summary WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""


def should_summarize(user_id: int) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
    count = c.fetchone()[0]
    conn.close()
    return count >= SUMMARY_TRIGGER


def replace_with_summary(user_id: int, summary_text: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    c.execute("""
        INSERT OR REPLACE INTO summary (user_id, content, updated)
        VALUES (?, ?, ?)
    """, (user_id, summary_text, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()
