"""
database.py
------------
Lightweight SQLite persistence layer for storing screening runs and
their ranked results, so recruiters can look back at past screening
sessions. Uses only the Python standard library's sqlite3 module.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "resume_screening.db"


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """Create tables if they don't already exist."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS screening_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT,
            job_description TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            resume_file TEXT,
            candidate_name TEXT,
            skills TEXT,
            experience TEXT,
            education TEXT,
            certifications TEXT,
            match_score REAL,
            rank INTEGER,
            FOREIGN KEY (run_id) REFERENCES screening_runs (id)
        )
    """)

    conn.commit()
    conn.close()


def save_run(job_title, job_description, results_df, db_path=DB_PATH):
    """
    Persist a screening run and its ranked candidate results.

    Parameters
    ----------
    job_title : str
    job_description : str
    results_df : pandas.DataFrame
        Output of matching.rank_candidates()

    Returns
    -------
    int : the new run_id
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO screening_runs (job_title, job_description, created_at) VALUES (?, ?, ?)",
        (job_title, job_description, datetime.now().isoformat(timespec="seconds")),
    )
    run_id = cur.lastrowid

    for _, row in results_df.iterrows():
        cur.execute("""
            INSERT INTO candidates
                (run_id, resume_file, candidate_name, skills, experience,
                 education, certifications, match_score, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            row.get("Resume File", ""),
            row.get("Candidate Name", ""),
            row.get("Skills", ""),
            row.get("Experience", ""),
            row.get("Education", ""),
            row.get("Certifications", ""),
            float(row.get("Match Score (%)", 0)),
            int(row.get("Rank", 0)),
        ))

    conn.commit()
    conn.close()
    return run_id


def list_runs(db_path=DB_PATH):
    """Return all past screening runs, most recent first."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM screening_runs ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_run_candidates(run_id, db_path=DB_PATH):
    """Return all candidates for a given screening run, ranked."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM candidates WHERE run_id = ? ORDER BY rank ASC", (run_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
