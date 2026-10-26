"""Synthetic resumes (three layouts) with ground truth, used as the development set."""
from __future__ import annotations

import random

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Sneha", "Vikram", "Isha", "Arjun", "Nisha"]
LAST = ["Sharma", "Iyer", "Khan", "Patel", "Das", "Reddy", "Singh", "Nair", "Joshi", "Ghosh"]
CITIES = ["Bengaluru", "Hyderabad", "Chennai", "Pune", "Mumbai", "Noida", "Mohali"]
COMPANIES = ["Quantel Labs", "Nexora Tech", "Bluepeak Systems", "Corvid Analytics", "Tessellate AI"]
DEGREES = ["B.Tech", "B.E.", "MCA", "M.Tech", "B.Sc"]
STACKS = {
    "ml": ["python", "scikit-learn", "xgboost", "pandas", "sql", "machine learning"],
    "nlp": ["python", "nlp", "spacy", "bert", "ner", "langchain", "rag", "faiss"],
    "backend": ["java", "spring boot", "microservices", "kafka", "mysql", "docker"],
    "devops": ["kubernetes", "terraform", "jenkins", "aws", "docker", "prometheus"],
    "data": ["sql", "power bi", "tableau", "python", "pandas"],
}
PRETTY = {"scikit-learn": "Scikit-Learn", "spring boot": "Spring Boot", "power bi": "Power BI", "ner": "NER",
          "nlp": "NLP", "sql": "SQL", "aws": "AWS", "rag": "RAG", "bert": "BERT", "faiss": "FAISS"}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _phone(rng):
    return str(rng.choice([6, 7, 8, 9])) + "".join(str(rng.randrange(10)) for _ in range(9))


def make(rng: random.Random, i: int) -> tuple[str, dict]:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    email = name.lower().replace(" ", ".") + f"{i}@example.com"
    phone = _phone(rng)
    city = rng.choice(CITIES)
    deg = rng.choice(DEGREES)
    stack = rng.choice(list(STACKS))
    skills = rng.sample(STACKS[stack], k=rng.randint(3, len(STACKS[stack])))
    months = rng.randint(6, 84)
    end_y, end_m = 2026, 10
    start_total = end_y * 12 + end_m - 1 - months
    sy, sm = divmod(start_total, 12)
    shown = [PRETTY.get(s, s.title() if rng.random() < 0.5 else s) for s in skills]
    layout = i % 3
    if layout == 0:
        txt = (f"{name.upper()}\n{city} | {email} | +91 {phone[:5]} {phone[5:]}\n\nSKILLS\n{', '.join(shown)}\n\n"
               f"EXPERIENCE\nEngineer — {rng.choice(COMPANIES)} — {MON[sm]} {sy} to Present\n\nEDUCATION\n{deg}, State University, {sy}\n")
    elif layout == 1:
        txt = (f"Name: {name}\nEmail - {email}    Phone - {phone}\nLocation - {city}\n\n"
               f"Summary: {round(months / 12, 1)} years of experience in {stack}.\nTools: {' / '.join(shown)}\n"
               f"Education: {deg} ({sy})\n")
    else:
        txt = (f"{name} · {email} · {phone} · {city}\n\nI work at {rng.choice(COMPANIES)} since {MON[sm]} {sy} - present, "
               f"mostly with {', '.join(shown[:-1])} and {shown[-1]}.\n{deg} graduate.\n")
    if layout == 1 and stack in ("nlp", "devops"):
        skills = skills + [stack]  # the summary line names the stack, so it is a stated skill
    truth = {"email": email, "phone": phone, "years": round(months / 12, 1), "degree": deg, "skills": sorted(set(skills)),
             "location": city, "name": name}
    return txt, truth


def corpus(n: int = 300, seed: int = 1):
    rng = random.Random(seed)
    return [make(rng, i) for i in range(n)]
