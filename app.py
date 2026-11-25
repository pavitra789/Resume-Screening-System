"""
app.py
-------
AI-Based Resume Screening and Candidate Ranking System
Streamlit dashboard entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.text_extraction import extract_text
from utils.skill_extractor import extract_all, DEFAULT_SKILL_BANK
from utils.matching import rank_candidates
import database as db

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)

db.init_db()

st.title("📄 AI-Based Resume Screening and Candidate Ranking System")
st.caption("NLP-powered resume-to-job-description matching using TF-IDF & Cosine Similarity")

# ---------------------------------------------------------------------------
# Sidebar: Job description input
# ---------------------------------------------------------------------------
st.sidebar.header("1️⃣ Job Description")

job_title = st.sidebar.text_input("Job Title", placeholder="e.g. Python Developer")

jd_input_method = st.sidebar.radio("Provide job description via:", ["Paste text", "Upload file"])

job_description = ""
if jd_input_method == "Paste text":
    job_description = st.sidebar.text_area(
        "Paste the Job Description here",
        height=220,
        placeholder="e.g. We are looking for a Python developer with experience in Flask, "
                     "SQL, Pandas, and machine learning...",
    )
else:
    jd_file = st.sidebar.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    if jd_file is not None:
        try:
            job_description = extract_text(jd_file, filename=jd_file.name)
            st.sidebar.success("Job description extracted successfully.")
        except Exception as e:
            st.sidebar.error(f"Error reading job description: {e}")

st.sidebar.header("2️⃣ Upload Resumes")
resume_files = st.sidebar.file_uploader(
    "Upload one or more resumes (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.sidebar.header("3️⃣ Custom Skills (optional)")
custom_skills_raw = st.sidebar.text_area(
    "Add extra skills to detect (comma-separated). These are added to the default skill bank.",
    placeholder="e.g. terraform, snowflake, airflow",
)

run_button = st.sidebar.button("🚀 Run Screening", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_button:
    if not job_description.strip():
        st.error("Please provide a job description before running the screening.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        skill_bank = list(DEFAULT_SKILL_BANK)
        if custom_skills_raw.strip():
            extra = [s.strip().lower() for s in custom_skills_raw.split(",") if s.strip()]
            skill_bank = sorted(set(skill_bank + extra))

        with st.spinner("Extracting text and analyzing resumes..."):
            parsed_resumes = []
            errors = []
            for f in resume_files:
                try:
                    raw_text = extract_text(f, filename=f.name)
                    if not raw_text.strip():
                        errors.append(f"{f.name}: no extractable text found (possibly a scanned/image PDF).")
                        continue
                    extracted = extract_all(raw_text, filename=f.name, skill_bank=skill_bank)
                    extracted["filename"] = f.name
                    extracted["raw_text"] = raw_text
                    parsed_resumes.append(extracted)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

            if errors:
                for err in errors:
                    st.warning(f"⚠️ {err}")

            if parsed_resumes:
                results_df = rank_candidates(job_description, parsed_resumes)
            else:
                results_df = pd.DataFrame()

        if results_df.empty:
            st.error("No resumes could be processed. Please check the uploaded files.")
        else:
            st.success(f"Screening complete — {len(results_df)} resume(s) ranked.")

            # Save to SQLite
            run_id = db.save_run(job_title or "Untitled Role", job_description, results_df)
            st.caption(f"Saved as screening run #{run_id} in the database.")

            # ---------------- Results table ----------------
            st.subheader("🏆 Ranked Candidates")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Match Score (%)": st.column_config.ProgressColumn(
                        "Match Score (%)", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
            )

            # ---------------- Download button ----------------
            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Shortlist as CSV",
                data=csv_bytes,
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )

            # ---------------- Chart ----------------
            st.subheader("📊 Match Score Comparison")
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(results_df))))
            ax.barh(results_df["Candidate Name"], results_df["Match Score (%)"], color="#2563eb")
            ax.set_xlabel("Match Score (%)")
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            for i, v in enumerate(results_df["Match Score (%)"]):
                ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
            st.pyplot(fig)

            # ---------------- Candidate detail expanders ----------------
            st.subheader("🔍 Candidate Details")
            for _, row in results_df.iterrows():
                with st.expander(f"#{row['Rank']} — {row['Candidate Name']} ({row['Match Score (%)']}%)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Resume File:** {row['Resume File']}")
                        st.markdown(f"**Experience:** {row['Experience']}")
                        st.markdown(f"**Education:** {row['Education']}")
                    with col2:
                        st.markdown(f"**Skills:** {row['Skills']}")
                        st.markdown(f"**Certifications:** {row['Certifications']}")

else:
    st.info(
        "👈 Provide a job description and upload resumes in the sidebar, then click "
        "**Run Screening** to see ranked candidates."
    )

    with st.expander("ℹ️ How this works"):
        st.markdown("""
1. **Text Extraction** — Resumes (PDF/DOCX/TXT) are parsed into plain text.
2. **Preprocessing** — Text is cleaned, tokenized, stripped of stop words, and lemmatized.
3. **Feature Extraction** — Skills, education, certifications, and years of experience are
   detected using keyword and pattern matching.
4. **Vectorization** — The job description and every resume are converted into TF-IDF vectors.
5. **Similarity Scoring** — Cosine similarity between each resume vector and the job
   description vector produces a Match Score (0–100%).
6. **Ranking** — Candidates are sorted by Match Score, highest first.
        """)

# ---------------------------------------------------------------------------
# Past screening runs (sidebar navigation)
# ---------------------------------------------------------------------------
st.sidebar.header("📁 Past Screening Runs")
past_runs = db.list_runs()
if past_runs:
    run_labels = [f"#{r['id']} — {r['job_title']} ({r['created_at']})" for r in past_runs]
    selected = st.sidebar.selectbox("View a past run", ["-- Select --"] + run_labels)
    if selected != "-- Select --":
        run_id = int(selected.split("—")[0].strip().lstrip("#"))
        candidates = db.get_run_candidates(run_id)
        if candidates:
            st.subheader(f"📂 Past Run #{run_id} Results")
            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
else:
    st.sidebar.caption("No past runs yet.")
