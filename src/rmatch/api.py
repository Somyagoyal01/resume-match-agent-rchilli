"""FastAPI service: POST /parse (resume text -> fields) and POST /agent (recruiter request)."""
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import Agent, AgentReply
from .parser import ParsedResume, parse

RESUMES = Path(__file__).resolve().parents[2] / "data" / "heldout_resumes"
app = FastAPI(title="resume-match-agent")
agent = Agent(RESUMES)


class Text(BaseModel):
    text: str


@app.post("/parse", response_model=ParsedResume)
def parse_endpoint(body: Text):
    return parse(body.text)


@app.post("/agent", response_model=AgentReply)
def agent_endpoint(body: Text):
    return agent.run(body.text)
