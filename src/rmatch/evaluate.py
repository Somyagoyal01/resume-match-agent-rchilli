"""Field-level accuracy on synthetic (dev) and hand-written resumes; agent decisions on hand-written requests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .agent import Agent
from .generate import corpus
from .parser import parse

DATA = Path(__file__).resolve().parents[2] / "data"
FIELDS = ["email", "phone", "degree", "years"]


def _field_ok(field, pred, true):
    if field == "years":
        return true is None and pred is None or (pred is not None and true is not None and abs(pred - true) <= 0.5)
    return pred == true


def score_resumes(pairs):
    rows, tp, fp, fn = [], 0, 0, 0
    for text, truth in pairs:
        p = parse(text).model_dump()
        rows.append({f: _field_ok(f, p[f], truth[f]) for f in FIELDS})
        ps, ts = set(p["skills"]), set(truth["skills"])
        tp, fp, fn = tp + len(ps & ts), fp + len(ps - ts), fn + len(ts - ps)
    df = pd.DataFrame(rows)
    out = {f"{f}_acc": round(float(df[f].mean()), 3) for f in FIELDS}
    out["skill_precision"] = round(tp / max(tp + fp, 1), 3)
    out["skill_recall"] = round(tp / max(tp + fn, 1), 3)
    out["n"] = len(pairs)
    return out


def score_requests(csv_path: Path, agent: Agent):
    req = pd.read_csv(csv_path)
    req["got"] = [agent.run(r).action for r in req.request]
    unsafe = req.expected == "refuse"
    return {
        "n": len(req),
        "decision_acc": round(float((req.got == req.expected).mean()), 3),
        "unsafe_refused": f"{int((req.got[unsafe] == 'refuse').sum())}/{int(unsafe.sum())}",
        "safe_wrongly_refused": f"{int((req.got[~unsafe] == 'refuse').sum())}/{int((~unsafe).sum())}",
        "errors": req[req.got != req.expected][["request", "expected", "got"]].to_dict("records"),
    }


def run_all() -> dict:
    res = {"dev_synthetic": score_resumes(corpus())}
    truth = json.loads((DATA / "heldout_resumes_truth.json").read_text())
    pairs = [((DATA / "heldout_resumes" / f).read_text(), t) for f, t in truth.items()]
    res["heldout_resumes_seen"] = score_resumes(pairs)
    agent = Agent(DATA / "heldout_resumes")
    res["requests_seen"] = score_requests(DATA / "heldout_requests_handwritten.csv", agent)
    blind_r = DATA / "blind_resumes_truth.json"
    if blind_r.exists():
        bt = json.loads(blind_r.read_text())
        res["blind_resumes"] = score_resumes([((DATA / "blind_resumes" / f).read_text(), t) for f, t in bt.items()])
    blind_q = DATA / "blind_requests.csv"
    if blind_q.exists():
        # evaluation wiring only: the blind requests refer to the blind resumes, so load both folders
        res["blind_requests"] = score_requests(blind_q, Agent(DATA / "heldout_resumes", DATA / "blind_resumes"))
    return res
