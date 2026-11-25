"""
skill_extractor.py
--------------------
Extracts structured information from raw resume text:
    - Candidate name (heuristic)
    - Skills (keyword matching against a customizable skill bank)
    - Education (degree detection via regex)
    - Years of experience (regex on "X years" patterns)
    - Certifications (keyword matching)

This is a lightweight, rule-based extractor. It is designed to be swapped
out later for a Named Entity Recognition (NER) model such as spaCy's
`en_core_web_sm`, or a transformer-based extractor, without changing the
calling code in app.py (see "Future Enhancements" in README.md).
"""

import re

# ---------------------------------------------------------------------------
# Skill bank: extend this list to fit your domain / job roles.
# ---------------------------------------------------------------------------
DEFAULT_SKILL_BANK = [
    # Programming languages
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r",
    "go", "golang", "php", "ruby", "kotlin", "swift", "scala", "matlab",
    # Data science / ML
    "machine learning", "deep learning", "data science", "nlp",
    "natural language processing", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy", "opencv",
    "data analysis", "data visualization", "statistics", "matplotlib",
    "seaborn", "power bi", "tableau", "excel",
    # Web / software development
    "html", "css", "react", "react native", "angular", "vue", "node.js",
    "flask", "django", "fastapi", "rest api", "graphql", "spring boot",
    # Databases
    "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    "firebase",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
    "git", "github", "linux", "bash",
    # Soft / general
    "project management", "communication", "leadership", "teamwork",
    "problem solving", "agile", "scrum",
]

EDUCATION_KEYWORDS = [
    r"ph\.?d", r"doctorate", r"m\.?tech", r"b\.?tech", r"m\.?sc",
    r"b\.?sc", r"m\.?e\b", r"b\.?e\b", r"mba", r"bca", r"mca",
    r"bachelor(?:'s)?(?: of)?", r"master(?:'s)?(?: of)?",
    r"diploma", r"high school", r"associate degree",
]

CERTIFICATION_KEYWORDS = [
    "certified", "certification", "certificate", "aws certified",
    "pmp", "scrum master", "google certified", "microsoft certified",
    "coursera", "udemy", "nptel",
]


def extract_name(text, filename=None):
    """
    Heuristic name extraction: assumes the candidate's name appears on
    one of the first few non-empty lines and looks like a proper name
    (2-4 capitalized words, no digits/emails).

    Falls back to a cleaned-up version of the filename if no confident
    match is found.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name_pattern = re.compile(r"^([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3})$")

    for line in lines[:8]:
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        if len(line.split()) > 5:
            continue
        match = name_pattern.match(line.strip())
        if match:
            return match.group(1)

    if filename:
        base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.IGNORECASE)
        base = re.sub(r"[_\-]+", " ", base)
        base = re.sub(r"(resume|cv)", "", base, flags=re.IGNORECASE).strip()
        if base:
            return base.title()

    return "Unknown Candidate"


def extract_skills(text, skill_bank=None):
    """
    Match known skills against resume text (case-insensitive, whole-word
    matching where possible). Returns a sorted list of matched skills.
    """
    if skill_bank is None:
        skill_bank = DEFAULT_SKILL_BANK

    text_lower = text.lower()
    found = []
    for skill in skill_bank:
        skill_lower = skill.lower()
        pattern = re.escape(skill_lower)
        is_alphanumeric = re.match(r"^[a-z0-9\s]+$", skill_lower) is not None

        if is_alphanumeric:
            # Plain word(s): require whole-word match to avoid false positives
            # (e.g. skill "r" or "go" matching inside unrelated words)
            if re.search(r"\b" + pattern + r"\b", text_lower):
                found.append(skill)
        else:
            # Skills with special characters (c++, node.js, c#) - use a
            # boundary-aware substring check instead of strict \b matching,
            # since \b behaves inconsistently around punctuation.
            idx = text_lower.find(skill_lower)
            if idx != -1:
                before = text_lower[idx - 1] if idx > 0 else " "
                after_idx = idx + len(skill_lower)
                after = text_lower[after_idx] if after_idx < len(text_lower) else " "
                if not before.isalnum() and not after.isalnum():
                    found.append(skill)

    return sorted(set(found))


def extract_education(text):
    """Return a list of detected education / degree keywords found in the resume."""
    text_lower = text.lower()
    found = []
    for pattern in EDUCATION_KEYWORDS:
        if re.search(pattern, text_lower):
            # Clean up the pattern for display
            label = pattern.replace(r"\.?", ".").replace(r"\b", "").replace("(?:'s)?", "'s").replace("(?: of)?", "")
            found.append(label.strip())
    return sorted(set(found)) if found else ["Not specified"]


def extract_experience_years(text):
    """
    Extract years of experience using common resume phrasing, e.g.:
    "5 years of experience", "3+ years experience", "over 2 years".
    Returns the maximum number found, or 'Not specified'.
    """
    text_lower = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of)?\s*experience",
        r"experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\+?\s*years?",
        r"(\d+(?:\.\d+)?)\+?\s*yrs?\s*(?:of)?\s*experience",
    ]
    years_found = []
    for pattern in patterns:
        for match in re.findall(pattern, text_lower):
            try:
                years_found.append(float(match))
            except ValueError:
                continue

    if years_found:
        max_years = max(years_found)
        return f"{max_years:g} years"
    return "Not specified"


def extract_certifications(text):
    """Return a list of detected certification-related keywords."""
    text_lower = text.lower()
    found = [kw for kw in CERTIFICATION_KEYWORDS if kw in text_lower]
    return sorted(set(found)) if found else ["None found"]


def extract_all(text, filename=None, skill_bank=None):
    """
    Convenience wrapper: runs all extractors and returns a single dict,
    ready to be used as a row in the results DataFrame.
    """
    return {
        "candidate_name": extract_name(text, filename),
        "skills": extract_skills(text, skill_bank),
        "education": extract_education(text),
        "experience": extract_experience_years(text),
        "certifications": extract_certifications(text),
    }
