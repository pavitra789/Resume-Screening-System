# 📄 AI-Based Resume Screening and Candidate Ranking System

An NLP-powered system that automatically analyzes resumes against a job
description, calculates a match score, ranks candidates, and highlights
key skills, education, and experience — helping recruiters shortlist the
most suitable candidates quickly.

---

## ✨ Features

- Extract text from **PDF**, **DOCX**, and **TXT** resumes
- Clean and preprocess text using NLP (tokenization, stop-word removal, lemmatization)
- Extract **skills**, **education**, **experience**, and **certifications**
- Compare resumes against a job description using **TF-IDF + Cosine Similarity**
- Rank candidates by **Match Score (0–100%)**
- Interactive **Streamlit dashboard** with charts, expandable candidate cards, and CSV export
- **SQLite** storage of past screening runs
- Custom skill bank — add extra skills relevant to your role

---

## 🗂️ Project Structure

```
resume_screening_system/
├── app.py                     # Streamlit dashboard (main entry point)
├── database.py                # SQLite persistence layer
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── text_extraction.py     # PDF/DOCX/TXT text extraction
│   ├── preprocessing.py       # NLP cleaning pipeline (NLTK)
│   ├── skill_extractor.py     # Skills / education / experience extraction
│   └── matching.py            # TF-IDF vectorization + cosine similarity + ranking
└── sample_data/
    ├── sample_job_description.txt
    ├── sample_resume_rahul_sharma.txt
    ├── sample_resume_priya_singh.txt
    └── sample_resume_aman_gupta.txt
```

---

## ⚙️ Setup

1. **Clone/extract the project and enter the folder:**
   ```bash
   cd resume_screening_system
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   The dashboard opens automatically at `http://localhost:8501`.

   > NLTK data (stopwords, punkt tokenizer, wordnet) downloads automatically
   > the first time you run the app. If your machine has no internet access,
   > pre-download manually:
   > ```python
   > import nltk
   > nltk.download('punkt')
   > nltk.download('punkt_tab')
   > nltk.download('stopwords')
   > nltk.download('wordnet')
   > nltk.download('omw-1.4')
   > ```

---

## 🚀 How to Use

1. Enter a **Job Title** and paste (or upload) the **Job Description** in the sidebar.
2. Upload one or more **resumes** (PDF, DOCX, or TXT).
3. (Optional) Add any extra skills you want detected, comma-separated.
4. Click **🚀 Run Screening**.
5. View the ranked candidate table, match-score chart, and expandable candidate
   detail cards.
6. Download the shortlist as a CSV, or revisit past runs from the sidebar.

### Quick test with sample data

Use the files in `sample_data/` to try the app immediately:
- Job description: `sample_data/sample_job_description.txt`
- Resumes: the three `sample_resume_*.txt` files

Expected relative ranking with the sample data: Rahul Sharma > Priya Singh > Aman Gupta.

---

## 🧠 How It Works (Pipeline)

1. **Text Extraction** — `utils/text_extraction.py` reads PDF (via `pdfplumber`,
   falling back to `PyPDF2`), DOCX (via `python-docx`), or TXT files.
2. **Preprocessing** — `utils/preprocessing.py` lowercases text, strips
   punctuation/numbers/URLs/emails, tokenizes, removes stop words, and
   lemmatizes using NLTK.
3. **Feature Extraction** — `utils/skill_extractor.py` uses keyword and regex
   matching to pull out skills (from a customizable skill bank), education
   level, years of experience, and certifications, plus a heuristic for the
   candidate's name.
4. **Vectorization & Similarity** — `utils/matching.py` builds a TF-IDF matrix
   from the job description + all resumes using `scikit-learn`, then computes
   cosine similarity between the job description vector and each resume vector.
5. **Ranking** — Candidates are sorted by match score (highest first) and
   displayed with rank numbers.
6. **Persistence** — `database.py` saves each screening run and its ranked
   results to a local SQLite database (`resume_screening.db`) for later review.

---

## 🧩 Technologies Used

| Category              | Tools |
|-----------------------|-------|
| Language              | Python |
| Data handling         | Pandas, NumPy |
| NLP                   | NLTK (tokenization, stop words, lemmatization), Regex |
| ML / Similarity       | Scikit-learn (TF-IDF, cosine similarity) |
| File parsing          | pdfplumber, PyPDF2, python-docx |
| Visualization         | Matplotlib |
| Web interface         | Streamlit |
| Database              | SQLite |

---

## 🔮 Future Enhancements

- **Semantic matching** using transformer models (e.g., BERT / Sentence-Transformers)
  instead of pure TF-IDF, to catch conceptually similar but differently worded skills.
- **AI-powered interview question generation** based on each resume's content.
- **Automatic resume summarization** for quick recruiter skimming.
- **Skill-gap analysis** with recommended online courses for missing skills.
- **Named Entity Recognition (NER)** via spaCy for more accurate name/organization
  extraction, replacing the current heuristic/regex approach.
- **Job portal integration** to pull job descriptions and push shortlists automatically.
- **Email notifications** to shortlisted candidates using SMTP or an email API.

---

## 📌 Notes

- This is a **rule-based + TF-IDF** implementation — no ML model training is
  required to get useful results, as recommended for a first working version.
  The `skill_extractor.py` module is intentionally modular so it can later be
  swapped for a trained NER/ML model without touching `app.py`.
- Scanned/image-only PDFs (no embedded text layer) will not extract text;
  OCR (e.g. `pytesseract`) would need to be added for those cases.
