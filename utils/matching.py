"""
matching.py
------------
Core matching engine: converts the job description and resumes into
TF-IDF vectors and computes cosine similarity to produce a match score
(0-100%) for each resume, then ranks candidates.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import preprocess_pipeline


def compute_match_scores(job_description, resumes):
    """
    Parameters
    ----------
    job_description : str
        Raw job description text.
    resumes : list[dict]
        Each dict must contain at least {"filename": ..., "raw_text": ...}

    Returns
    -------
    list[float] : match score (0-100) per resume, in the same order as `resumes`.
    """
    if not resumes:
        return []

    jd_clean = preprocess_pipeline(job_description)
    resume_clean_texts = [preprocess_pipeline(r["raw_text"]) for r in resumes]

    corpus = [jd_clean] + resume_clean_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
    scores = [round(float(sim) * 100, 2) for sim in similarities]
    return scores


def rank_candidates(job_description, parsed_resumes):
    """
    Parameters
    ----------
    job_description : str
    parsed_resumes : list[dict]
        Each dict should already contain extracted fields:
        filename, raw_text, candidate_name, skills, education,
        experience, certifications.

    Returns
    -------
    pandas.DataFrame sorted by Match Score (descending), with a Rank column.
    """
    scores = compute_match_scores(job_description, parsed_resumes)

    rows = []
    for resume, score in zip(parsed_resumes, scores):
        rows.append({
            "Resume File": resume.get("filename", ""),
            "Candidate Name": resume.get("candidate_name", "Unknown"),
            "Skills": ", ".join(resume.get("skills", [])) or "None detected",
            "Experience": resume.get("experience", "Not specified"),
            "Education": ", ".join(resume.get("education", [])) or "Not specified",
            "Certifications": ", ".join(resume.get("certifications", [])) or "None found",
            "Match Score (%)": score,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
    return df
