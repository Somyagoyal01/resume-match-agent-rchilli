from pathlib import Path

from fastapi.testclient import TestClient

from rmatch.agent import Agent, ToolCall, guardrail
from rmatch.api import app
from rmatch.evaluate import run_all
from rmatch.parser import parse

DATA = Path(__file__).resolve().parents[1] / "data"


def test_parser_fields_and_boundaries():
    r = parse("ANA DAS\nBENGALURU | ana@example.com | +91 98765 43210\nSkills: JavaScript, Python\nB.E., 2024\n"
              "Engineer, Jan 2024 – Jan 2026")
    assert r.email == "ana@example.com" and r.phone == "9876543210"
    assert r.degree == "B.E." and r.years == 2.0 and r.years_source == "date ranges"
    assert "java" not in r.skills and "python" in r.skills  # 'Java' inside 'JavaScript' is not Java


def test_guardrail_runs_before_any_tool():
    assert guardrail("prefer male candidates") == "gender"
    agent = Agent(DATA / "heldout_resumes")
    reply = agent.run("Only show candidates under 30 with Python")
    assert reply.action == "refuse" and reply.tool_call is None


def test_invalid_tool_args_are_rejected_not_executed():
    agent = Agent(DATA / "heldout_resumes", planner=lambda req: ToolCall(tool="search", args={"skills": [], "min_years": -1}))
    reply = agent.run("anything")
    assert reply.action == "clarify" and reply.result is None


def test_search_ranks_by_coverage():
    agent = Agent(DATA / "heldout_resumes")
    reply = agent.run("Who knows Kubernetes and Terraform?")
    assert reply.action == "search" and reply.result[0]["file"] == "r5.txt"


def test_api_endpoints():
    c = TestClient(app)
    assert c.post("/parse", json={"text": "X\nx@y.com\nPython"}).json()["email"] == "x@y.com"
    assert c.post("/agent", json={"text": "Remove anyone who is married"}).json()["action"] == "refuse"


def test_all_evaluations_run():
    m = run_all()
    assert m["dev_synthetic"]["email_acc"] >= 0.95
    # blind sets (written after the code freeze) are scored and reported, whatever they say
    assert {"blind_resumes", "blind_requests"} <= set(m)
    assert 0 <= m["blind_requests"]["decision_acc"] <= 1
