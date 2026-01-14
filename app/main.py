# app/api/main.py
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.coach import router as coach_router
from app.core.engine import run_engine, score_answer
from app.llm.ollama_client import OllamaClient

app = FastAPI(
    title="AI Interview Coach API",
    description="Local interview coach API (rule-based scoring + optional Ollama coaching).",
    version="0.4.1",
)

# CORS (ok for local dev; tighten for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount routes from app/api/coach.py
# Provides:
#   POST /questions
#   POST /coach-structured
app.include_router(coach_router)


# ----------------------------
# Request Models (scoring)
# ----------------------------
class ScoreRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    use_llm: bool = Field(False, description="If true, uses hybrid engine (optional).")
    model: str | None = Field(None, description="Optional Ollama model override.")


class SubmissionScoreRequest(BaseModel):
    """
    ✅ Supports BOTH formats:

    A) Legacy format used by older UI:
       { "answers": { "question": "answer", ... }, "use_llm": false, "model": null }

    B) New format used by updated UI:
       { "items": [ { "question": "...", "answer": "...", "keywords": [...], "tech_terms": [...] }, ... ] }
       (keywords/tech_terms are optional and are ignored by this API layer unless you wire them into engine)
    """

    answers: dict[str, str] | None = Field(
        default=None, description="Mapping of question -> answer"
    )
    items: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of items with question/answer fields (new UI payload).",
    )
    use_llm: bool = Field(False, description="Enable hybrid engine for each answer.")
    model: str | None = Field(None, description="Optional Ollama model override.")


# ----------------------------
# Response Models (scoring)
# ----------------------------
class ScoreBreakdownResponse(BaseModel):
    score: int
    keyword_coverage: float
    depth_score: float
    clarity_score: float
    seniority_alignment: str
    feedback: str
    tips: list[str] = Field(default_factory=list)

    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)


class SubmissionScoreResponse(BaseModel):
    results: list[ScoreBreakdownResponse]
    overall_average: float
    overall_level: str


# ----------------------------
# Helpers
# ----------------------------
def _level_from_avg(avg: float) -> str:
    if avg >= 80:
        return "Senior"
    if avg >= 55:
        return "Mid"
    return "Junior"


def _normalize_single_result(engine_result: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize whatever app.core.engine returns into a stable API response.
    Defensive so the UI never crashes.
    """
    score = int(engine_result.get("score", 0))

    keyword_coverage = float(
        engine_result.get("keyword_coverage", engine_result.get("keyword_score", 0))
    )
    depth_score = float(engine_result.get("depth_score", 0))
    clarity_score = float(engine_result.get("clarity_score", 0))

    seniority_alignment = str(
        engine_result.get("seniority", engine_result.get("level", "Junior"))
    )
    feedback = str(engine_result.get("feedback", engine_result.get("summary", "")))

    tips = engine_result.get("tips") or []
    if not isinstance(tips, list):
        tips = [str(tips)]
    tips = [str(t).strip() for t in tips if str(t).strip()]

    matched = (
        engine_result.get("matched_keywords") or engine_result.get("matched") or []
    )
    if not isinstance(matched, list):
        matched = []
    matched = [str(x).strip() for x in matched if str(x).strip()]

    missing = (
        engine_result.get("missing_keywords") or engine_result.get("missing") or []
    )
    if not isinstance(missing, list):
        missing = []
    missing = [str(x).strip() for x in missing if str(x).strip()]

    return {
        "score": score,
        "keyword_coverage": keyword_coverage,
        "depth_score": depth_score,
        "clarity_score": clarity_score,
        "seniority_alignment": seniority_alignment,
        "feedback": feedback,
        "tips": tips,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


def _answers_from_submission(req: SubmissionScoreRequest) -> dict[str, str]:
    """
    ✅ Unifies payload shapes into a single dict[str, str].
    """
    # A) direct answers dict
    if isinstance(req.answers, dict) and req.answers:
        return {
            str(k).strip(): (v or "").strip()
            for k, v in req.answers.items()
            if str(k).strip()
        }

    # B) items list -> answers dict
    out: dict[str, str] = {}
    if isinstance(req.items, list):
        for it in req.items:
            if not isinstance(it, dict):
                continue
            q = (it.get("question") or "").strip()
            a = (it.get("answer") or "").strip()
            if q:
                out[q] = a
    return out


def score_submission(
    answers: dict[str, str],
    use_llm: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for q, ans in (answers or {}).items():
        q = (q or "").strip()
        ans = (ans or "").strip()
        if not q:
            continue

        if use_llm:
            engine_result: dict[str, Any] = run_engine(
                {"question": q, "answer": ans},
                use_llm=True,
                model=model,
            )
        else:
            # rule-based scoring
            engine_result = score_answer(q, ans)

        results.append(_normalize_single_result(engine_result))

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "results": results,
        "overall_average": float(avg),
        "overall_level": _level_from_avg(float(avg)),
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/llm/health")
def llm_health():
    c = OllamaClient()
    return {
        "ollama_ok": c.health_check(),
        "base_url": c.base_url,
        "model": c.model,
    }


@app.post("/score", response_model=ScoreBreakdownResponse)
def score(req: ScoreRequest):
    engine_result: dict[str, Any] = run_engine(
        {"question": req.question, "answer": req.answer},
        use_llm=req.use_llm,
        model=req.model,
    )
    return ScoreBreakdownResponse(**_normalize_single_result(engine_result))


@app.post("/score-submission", response_model=SubmissionScoreResponse)
def score_all(req: SubmissionScoreRequest):
    answers_dict = _answers_from_submission(req)

    submission_result: dict[str, Any] = score_submission(
        answers=answers_dict,
        use_llm=req.use_llm,
        model=req.model,
    )

    normalized_results = [
        ScoreBreakdownResponse(**r) for r in submission_result["results"]
    ]
    return SubmissionScoreResponse(
        results=normalized_results,
        overall_average=float(submission_result.get("overall_average", 0.0)),
        overall_level=str(submission_result.get("overall_level", "Junior")),
    )
