# resume-match-agent-rchilli

*Personal project by Somya Goyal, built while applying to RChilli. Not affiliated with or endorsed by RChilli; uses synthetic and hand-written sample resumes only (no real people).*

A small recruiter agent on top of a resume parser. It turns a recruiter's request into a **validated tool call** (search / parse / score), and **refuses requests to filter on protected attributes before any tool runs**. The project is evaluated on synthetic resumes, on hand-written ones, and on blind sets written after the code was frozen.

## Why this matters for RChilli

- **The problem.** Once a parser sits behind an agent, a recruiter can type anything, including "only show me candidates under 30" or "remove anyone who is married". An HR-tech agent has to (a) turn requests into correct, typed API calls and (b) refuse discriminatory filters reliably.
- **Who feels it.** RChilli's customers (ATS/HR platforms) and the AI team building agentic features. The AI Developer posting (RC504) asks for "tool calling", "function calling", agent frameworks, FastAPI, NER, and background in "resume parsing, or job matching".
- **What this shows.** A guardrail-first agent loop (guardrail → plan → Pydantic validation → tool) with a FastAPI service, an offline planner that needs no API key, and a documented real-LLM tool-calling path that goes through the same validation. It also shows an honest measurement of where a keyword guardrail breaks.
- **Expected impact (estimate).** Rules on hiring AI are tightening: the EU AI Act lists employment uses as high-risk, and New York City requires bias audits for automated hiring tools. A refusal layer with a measured miss rate gives a product team something concrete to show customers. That assumption is not tested here.

Source: RChilli *AI Developer (RC504)* posting on careerspage.io/rchilli (Oct 2026).

## Demo

```bash
pip install -r requirements.txt
python run.py                       # writes results/metrics.json
pytest -q                           # 6 tests
uvicorn rmatch.api:app --app-dir src   # POST /parse {"text": ...}, POST /agent {"text": ...}
```

Example: `POST /agent {"text": "Who knows Kubernetes and Terraform?"}` returns a `search` tool call with `{"skills": ["kubernetes", "terraform"], "min_years": 0}` and ranked matches. `"Prefer male candidates..."` is refused, and no tool is called.

## How it works

- `parser.py`: regex + a skills taxonomy with aliases (word-boundary matching, so "JavaScript" is not "Java"). It extracts email, Indian phone numbers, degree, and years of experience. Years come from a stated line ("3 yrs 6 months total") or from adding up date ranges, and `years_source` records which, so a reviewer knows how far to trust the number.
- `agent.py`:
  1. `guardrail()` checks protected attributes (age, gender, marital status, religion, caste, disability) **before** planning.
  2. The planner returns a `ToolCall`. `plan_offline` is scripted; `plan_with_llm` uses OpenAI tool calling and is not run here.
  3. Arguments are validated with Pydantic (`SearchArgs`, `ParseArgs`, `ScoreArgs`). Invalid calls become a clarifying question and are never executed.
- `api.py`: FastAPI with typed request and response models.
- `evaluate.py`: field accuracy, skill precision/recall, and agent decision accuracy.

## Results (from `python run.py`)

| set | written | email | phone | degree | years (±0.5) | skill P / R |
|---|---|---|---|---|---|---|
| synthetic dev (300, 3 layouts) | generator | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| hand-written (5) | before code, but **seen while writing the parser** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| **blind (3)** | **after code freeze** | 1.00 | 1.00 | **0.67** | **0.67** | 1.00 / **0.92** |

| recruiter requests | decision accuracy | unsafe refused | safe wrongly refused |
|---|---|---|---|
| hand-written (10), seen while writing the guardrail | 1.00 | 5/5 | 0/5 |
| **blind (10), written after freeze** | **0.70** | **1/4** | 0/6 |

### What the numbers say

- **The perfect scores are not the result.** The first hand-written sets were in front of me while I wrote the patterns, so they are self-confirming. The blind sets are the honest numbers.
- **The keyword guardrail caught only 1 of 4 blind unsafe requests.** It missed "candidates who have **kids**", "anyone **above 35**" (age written without the word "age"), and "**North India** / Hindi speakers" (region and language were not in the list). It did catch "ideally women". There were no false refusals. This is the main finding: a word list is not a guardrail. The next step is a small classifier or LLM-as-judge on the request, calibrated on a labelled set, with the keyword layer kept as a fast first pass.
- **Parser misses on blind resumes:** "Bachelor of Technology" written out in full (the degree list only has abbreviations); "four years" spelled as a word; a résumé that starts with "Curriculum Vitae", so the name is wrong (the name isn't scored, but it shows in the output); and "Hugging Face" without "transformers".
- One harness bug was found and fixed after the freeze: the blind requests point to the blind resumes, so the evaluation agent now loads both folders. No parsing or guardrail logic changed.

## Limitations / next steps

- Plain-text resumes only. PDF/DOCX, layout-aware extraction and NER models (spaCy or fine-tuned BERT) would replace the regex layer.
- The guardrail should become a request classifier with the measured refusal recall reported, including region, language, family status and age written as numbers.
- No MCP server yet. The three tools map directly onto MCP tools, which would let other agent platforms call them.
- Matching is skill coverage only. Embedding-based skill similarity ("PySpark" ≈ "Spark") would raise recall on resumes like b1.

## Files

- `src/rmatch/parser.py`, `agent.py`, `api.py`, `generate.py`, `evaluate.py`; `run.py`
- `data/heldout_*`: first hand-written sets; `data/blind_*`: written after the code freeze
- `tests/test_agent.py`: 6 tests (parser boundaries, guardrail-before-tools, invalid args not executed, ranking, API, evaluation runs)
