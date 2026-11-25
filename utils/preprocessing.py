"""
preprocessing.py
-----------------
NLP text cleaning pipeline used before TF-IDF vectorization:
    1. Lowercase + remove special characters/numbers/punctuation
    2. Tokenization
    3. Stop word removal
    4. Lemmatization

Uses NLTK. Required NLTK data packages are downloaded automatically
on first run (punkt, stopwords, wordnet, omw-1.4).
"""

import re
import nltk

_NLTK_READY = False


def _ensure_nltk_data():
    """Download required NLTK corpora once, silently, if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # Non-fatal: pipeline will fall back to simpler methods
    _NLTK_READY = True


def clean_text(text):
    """Lowercase, remove URLs/emails, strip punctuation & digits, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def tokenize(text):
    """Split cleaned text into tokens."""
    _ensure_nltk_data()
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        # Fallback: simple whitespace split if NLTK tokenizer data unavailable
        return text.split()


def remove_stopwords(tokens):
    """Remove common English stop words and very short tokens."""
    _ensure_nltk_data()
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        # Minimal built-in fallback stop word list
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "of",
            "to", "in", "on", "for", "with", "as", "at", "by", "from", "this",
            "that", "it", "be", "have", "has", "had", "will", "would", "can",
            "could", "i", "you", "he", "she", "we", "they", "them", "his",
            "her", "their", "our", "your",
        }
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def lemmatize(tokens):
    """Reduce tokens to their base/dictionary form."""
    _ensure_nltk_data()
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        return tokens  # Fallback: return tokens unchanged


def preprocess_pipeline(text):
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a single space-joined string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
