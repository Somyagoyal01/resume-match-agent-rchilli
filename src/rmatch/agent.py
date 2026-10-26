"""A small recruiter agent: guardrail -> plan a tool call (validated JSON) -> run the tool.

The planner is offline by default (a scripted parser of the request), so the repo
runs without API keys. `plan_with_llm` shows where a real model plugs in with
tool/function calling; its output goes through the SAME Pydantic validation and
the SAME guardrail, so swapping the planner does not change the safety contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .parser import SKILLS, ParsedResume, parse

# Requests to filter or rank on these are refused (Indian and common HR fairness norms).
PROTECTED = {
    "age": r"\b(age|aged|under \d\d|over \d\d|younger|older|young|old)\b",
    "gender": r"\b(male|female|men|women|gender|girls?|boys?)\b",
    "marital status": r"\b(married|unmarried|single|marital|pregnan\w*)\b",
    "religion": r"\b(religion|religious|hindu|muslim|christian|sikh|jain)\b",
    "caste": r"\b(caste|reservation category|sc/st|obc)\b",
    "disability": r"\b(disabilit\w*|disabled|handicap\w*)\b",
}


class SearchArgs(BaseModel):
    skills: list[str] = Field(min_length=1)
    min_years: float = Field(default=0, ge=0, le=40)
    location: str | None = None


class ParseArgs(BaseModel):
    file: str = Field(pattern=r"^[\w.-]+\.txt$")


class ScoreArgs(BaseModel):
    file: str = Field(pattern=r"^[\w.-]+\.txt$")
    jd_skills: list[str] = Field(min_length=1)


class ToolCall(BaseModel):
    tool: Literal["search", "parse", "score"]
    args: dict


class AgentReply(BaseModel):
    action: Literal["refuse", "search", "parse", "score", "clarify"]
    message: str
    result: object | None = None
    tool_call: dict | None = None


def guardrail(request: str) -> str | None:
    low = request.lower()
    for attr, pat in PROTECTED.items():
        if re.search(pat, low):
            return attr
    return None


def _skills_in(text: str) -> list[str]:
    low = text.lower()
    out = []
    for canon, aliases in SKILLS.items():
        if any(re.search(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", low) for a in aliases):
            out.append(canon)
    if "backend" in low and "python" in out:
        out += ["fastapi", "postgresql"]  # what 'backend python' usually means in a JD
    return sorted(set(out))


def plan_offline(request: str) -> ToolCall | None:
    low = request.lower()
    file = re.search(r"([\w-]+\.txt)", request)
    if file and ("score" in low or "against" in low or "match" in low):
        return ToolCall(tool="score", args={"file": file.group(1), "jd_skills": _skills_in(request.split("against")[-1])})
    if file:
        return ToolCall(tool="parse", args={"file": file.group(1)})
    yrs = re.search(r"(\d+)\s*\+?\s*(?:years|yrs)", low)
    loc = next((c for c in ["Bengaluru", "Hyderabad", "Chennai", "Pune", "Mumbai", "Kochi", "Mohali"] if c.lower() in low), None)
    skills = _skills_in(request)
    if not skills:
        return None
    return ToolCall(tool="search", args={"skills": skills, "min_years": float(yrs.group(1)) if yrs else 0, "location": loc})


def plan_with_llm(request: str, client=None, model: str = "gpt-4o-mini") -> ToolCall | None:  # pragma: no cover
    """Real-LLM path (needs OPENAI_API_KEY). Not used in tests or results."""
    tools = [{"type": "function", "function": {"name": n, "parameters": m.model_json_schema()}}
             for n, m in (("search", SearchArgs), ("parse", ParseArgs), ("score", ScoreArgs))]
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": request}], tools=tools)
    calls = resp.choices[0].message.tool_calls
    if not calls:
        return None
    return ToolCall(tool=calls[0].function.name, args=json.loads(calls[0].function.arguments))


class Agent:
    def __init__(self, *resume_dirs: Path, planner=plan_offline):
        self.planner = planner
        self.db: dict[str, ParsedResume] = {p.name: parse(p.read_text())
                                            for d in resume_dirs for p in sorted(Path(d).glob("*.txt"))}

    def search(self, a: SearchArgs):
        rows = []
        for f, r in self.db.items():
            hit = [s for s in a.skills if s in r.skills]
            if not hit or (r.years or 0) < a.min_years:
                continue
            rows.append({"file": f, "name": r.name, "matched": hit, "coverage": round(len(hit) / len(a.skills), 2),
                         "years": r.years, "location": r.location, "location_match": a.location == r.location})
        return sorted(rows, key=lambda x: (-x["coverage"], -x["location_match"], -(x["years"] or 0)))

    def score(self, a: ScoreArgs):
        r = self.db[a.file]
        hit = [s for s in a.jd_skills if s in r.skills]
        return {"file": a.file, "coverage": round(len(hit) / len(a.jd_skills), 2), "matched": hit,
                "missing": [s for s in a.jd_skills if s not in r.skills]}

    def run(self, request: str) -> AgentReply:
        attr = guardrail(request)
        if attr:
            return AgentReply(action="refuse", message=f"I can't filter or rank candidates on {attr}. "
                              "I can search on skills, experience and location instead.")
        call = self.planner(request)
        if call is None:
            return AgentReply(action="clarify", message="Which skills or which resume file should I use?")
        try:
            model = {"search": SearchArgs, "parse": ParseArgs, "score": ScoreArgs}[call.tool]
            args = model(**call.args)
        except ValidationError as e:
            return AgentReply(action="clarify", message=f"I couldn't build a valid {call.tool} call: {e.errors()[0]['msg']}",
                              tool_call=call.model_dump())
        if call.tool in ("parse", "score") and args.file not in self.db:
            return AgentReply(action="clarify", message=f"No resume named {args.file}.", tool_call=call.model_dump())
        result = {"search": self.search, "score": self.score,
                  "parse": lambda a: self.db[a.file].model_dump()}[call.tool](args)
        return AgentReply(action=call.tool, message="ok", result=result, tool_call=call.model_dump())
