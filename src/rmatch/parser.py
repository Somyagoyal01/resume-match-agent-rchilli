"""Plain-text resume parser: contact fields, degree, years of experience, skills.

Deliberately simple and inspectable (regex + a skills taxonomy with aliases), so
every extracted value can be traced back to the text span that produced it.
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel

TODAY = date(2026, 10, 27)

# canonical skill -> aliases seen in resumes (lowercase)
SKILLS = {
    "python": ["python"], "java": ["java"], "sql": ["sql"], "r": [],
    "spring boot": ["spring boot", "springboot"], "microservices": ["microservices", "micro-services"],
    "kafka": ["kafka"], "react": ["react", "react.js", "reactjs"], "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"], "mongodb": ["mongodb", "mongo"],
    "fastapi": ["fastapi"], "flask": ["flask"], "django": ["django"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"],
    "jenkins": ["jenkins"], "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"],
    "grafana": ["grafana"], "prometheus": ["prometheus"], "devops": ["devops"],
    "pandas": ["pandas"], "numpy": ["numpy"], "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "xgboost": ["xgboost"], "pytorch": ["pytorch"], "tensorflow": ["tensorflow"],
    "nlp": ["nlp", "natural language processing"], "ner": ["ner", "named entity recognition"],
    "spacy": ["spacy"], "nltk": ["nltk"], "bert": ["bert"],
    "hugging face transformers": ["hugging face transformers", "huggingface transformers", "transformers"],
    "text classification": ["text classification"], "rag": ["rag", "retrieval-augmented generation"],
    "langchain": ["langchain"], "faiss": ["faiss"], "power bi": ["power bi", "powerbi"], "tableau": ["tableau"],
    "machine learning": ["machine learning"], "computer vision": ["computer vision", "opencv"],
}
DEGREES = ["M.Tech", "B.Tech", "B.E.", "MCA", "BCA", "M.Sc", "B.Sc", "MBA", "PhD", "M.E."]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


class ParsedResume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    degree: str | None = None
    years: float | None = None
    years_source: str | None = None  # "stated" or "date ranges": tells the reviewer how much to trust it
    skills: list[str] = []


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?:\+?91[\s-]?)?([6-9]\d{4}[\s-]?\d{5}|[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4})")
_STATED = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:yrs?|years?)(?:\s*(\d+)\s*months?)?", re.I)
_RANGE = re.compile(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{4})\s*(?:–|-|to)\s*(present|current|([A-Za-z]{3})[a-z]*\.?\s+(\d{4}))", re.I)
_YEAR_RANGE = re.compile(r"\((\d{4})\s*[-–]\s*(\d{4}|present)\)", re.I)
CITIES = ["Bengaluru", "Bangalore", "Hyderabad", "Chennai", "Pune", "Mumbai", "Kochi", "Delhi", "Gurugram", "Noida", "Mohali", "Kolkata"]


def _months(y1, m1, y2, m2):
    return max(0, (y2 - y1) * 12 + (m2 - m1))


def experience_years(text: str) -> tuple[float | None, str | None]:
    """Prefer an explicit statement ('3 yrs 6 months total'); else add up date ranges."""
    for m in _STATED.finditer(text):
        ctx = text[max(0, m.start() - 40): m.end() + 25].lower()
        if any(w in ctx for w in ("total", "experience", "summary", "~", "engineer", "developer")) and "month" not in text[m.start()-3:m.start()]:
            yrs = float(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0)
            return round(yrs, 1), "stated"
    months = 0
    for m in _RANGE.finditer(text):
        y1, m1 = int(m.group(2)), MONTHS.get(m.group(1)[:3].lower(), 1)
        if m.group(3).lower() in ("present", "current"):
            y2, m2 = TODAY.year, TODAY.month
        else:
            y2, m2 = int(m.group(5)), MONTHS.get(m.group(4)[:3].lower(), 1)
        months += _months(y1, m1, y2, m2)
    for m in _YEAR_RANGE.finditer(text):
        y2 = TODAY.year if m.group(2).lower() == "present" else int(m.group(2))
        months += (y2 - int(m.group(1))) * 12
    if months:
        return round(months / 12, 1), "date ranges"
    dur = re.search(r"\((\d+)\s*months?", text, re.I) or re.search(r"(\d+)\s*months?", text, re.I)
    if dur:
        return round(int(dur.group(1)) / 12, 1), "date ranges"
    return None, None


def extract_skills(text: str) -> list[str]:
    low = text.lower()
    found = []
    for canon, aliases in SKILLS.items():
        for a in aliases:
            if re.search(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", low):
                found.append(canon)
                break
    # 'java' inside 'javascript' is excluded by the boundary; 'transformers' alone is ambiguous
    return sorted(set(found))


def _degree_pattern(d: str) -> str:
    """'B.E.' -> matches 'B.E.', 'B.E', 'BE' as a whole token (not the 'BE' in 'BENGALURU')."""
    core = r"\.?".join(re.escape(part) for part in d.strip(".").split("."))
    return r"(?<![\w.])" + core + r"\.?(?![\w])"


def parse(text: str) -> ParsedResume:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    name = lines[0] if lines else None
    if name:
        name = re.sub(r"^name\s*[:\-]\s*", "", name, flags=re.I)
        name = re.split(r"\s[|—–-]\s|\s·\s", name)[0].strip()
    email = (_EMAIL.search(text) or [None])[0]
    pm = _PHONE.search(text)
    phone = re.sub(r"\D", "", pm.group(1)) if pm else None
    degree = next((d for d in DEGREES if re.search(_degree_pattern(d), text)), None)
    loc = next((c for c in CITIES if c.lower() in text.lower()), None)
    years, src = experience_years(text)
    return ParsedResume(name=name, email=email, phone=phone, location=loc, degree=degree,
                        years=years, years_source=src, skills=extract_skills(text))
