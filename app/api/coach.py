# app/api/coach.py
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.llm.ollama_client import OllamaClient

router = APIRouter()

# ============================================================
# Question Bank
# ============================================================
QuestionLevel = Literal["junior", "mid", "senior", "mixed"]

QUESTION_BANK: list[dict[str, Any]] = [
    {
        "level": "junior",
        "question": "Explain what an embedding is and why embeddings are useful in search or RAG.",
        "keywords": ["embedding", "vector", "similarity", "semantic"],
        "tech_terms": ["cosine similarity", "vector space"],
    },
    {
        "level": "junior",
        "question": "What is prompt engineering? Give two examples of techniques that improve LLM answers.",
        "keywords": ["prompt", "constraints", "examples", "format"],
        "tech_terms": ["few-shot", "system prompt"],
    },
    {
        "level": "junior",
        "question": "What is Retrieval-Augmented Generation (RAG) in simple terms?",
        "keywords": ["retrieve", "context", "grounding", "documents"],
        "tech_terms": ["vector database", "retriever"],
    },
    {
        "level": "mid",
        "question": "Compare RAG vs fine-tuning for a production LLM app. When would you choose each?",
        "keywords": ["latency", "cost", "freshness", "control", "data"],
        "tech_terms": ["LoRA", "context window"],
    },
    {
        "level": "senior",
        "question": "Explain failure modes in RAG and how to mitigate them.",
        "keywords": ["chunks", "drift", "stale", "monitoring"],
        "tech_terms": ["reranker", "telemetry"],
    },
]


# ============================================================
# /questions
# ============================================================
class QuestionsRequest(BaseModel):
    num: int = Field(3, ge=1, le=10)
    level: QuestionLevel = "mixed"


class QuestionItem(BaseModel):
    question: str
    level: str
    keywords: list[str]
    tech_terms: list[str]


class QuestionsResponse(BaseModel):
    questions: list[QuestionItem]
    meta: dict[str, Any] = Field(default_factory=dict)


def _pool_for_level(level: str) -> list[dict[str, Any]]:
    lvl = (level or "mixed").lower().strip()
    if lvl == "mixed":
        return list(QUESTION_BANK)
    filtered = [q for q in QUESTION_BANK if (q.get("level") or "").lower() == lvl]
    return filtered if filtered else list(QUESTION_BANK)


@router.post("/questions", response_model=QuestionsResponse)
def questions(req: QuestionsRequest):
    """
    ✅ Always returns exactly req.num questions whenever possible.
    If the requested difficulty pool has fewer than req.num items,
    the endpoint will include repeats so the UI gets the count requested.
    """
    import random

    pool = _pool_for_level(req.level)
    if not pool:
        return QuestionsResponse(questions=[], meta={"error": "Question bank empty"})

    want = max(1, int(req.num))

    # Case A: enough unique questions
    if want <= len(pool):
        selected = random.sample(pool, k=want)

    # Case B: not enough questions in this pool -> allow repeats
    else:
        shuffled = list(pool)
        random.shuffle(shuffled)
        selected = list(shuffled)

        while len(selected) < want:
            candidate = random.choice(pool)
            # Avoid immediate duplicate if possible
            if len(pool) > 1:
                while candidate == selected[-1]:
                    candidate = random.choice(pool)
            selected.append(candidate)

    out_items: list[QuestionItem] = []
    for q in selected:
        out_items.append(
            QuestionItem(
                question=str(q.get("question", "")).strip(),
                level=str(q.get("level", "mixed")).strip(),
                keywords=list(q.get("keywords") or []),
                tech_terms=list(q.get("tech_terms") or []),
            )
        )

    return QuestionsResponse(
        questions=out_items,
        meta={
            "requested_num": want,
            "returned_num": len(out_items),
            "level": req.level,
            "unique_in_bank_for_level": len(pool),
            "repeats_used": max(0, want - len(pool)),
        },
    )


# ============================================================
# ✅ SCORING
# ============================================================
class ScoreItem(BaseModel):
    question: str
    answer: str
    keywords: list[str] = Field(default_factory=list)
    tech_terms: list[str] = Field(default_factory=list)


class ScoreSubmission(BaseModel):
    items: list[ScoreItem] = Field(default_factory=list)


class PerQuestionScore(BaseModel):
    question: str
    score: float
    level: str


class ScoreResponse(BaseModel):
    overall_average: float
    overall_level: str
    per_question: list[PerQuestionScore]


def score_answer(answer: str) -> float:
    a = (answer or "").strip()
    if not a:
        return 0.0
    length = min(len(a), 600)
    return round(min(100.0, (length / 600) * 100), 1)


def level_from_score(score: float) -> str:
    if score >= 85:
        return "Senior"
    if score >= 65:
        return "Mid"
    if score >= 40:
        return "Junior"
    return "Needs Practice"


@router.post("/score-submission", response_model=ScoreResponse)
def score_submission(req: ScoreSubmission):
    per_q: list[PerQuestionScore] = []
    scores: list[float] = []

    for it in req.items:
        s = score_answer(it.answer)
        scores.append(s)
        per_q.append(
            PerQuestionScore(
                question=it.question,
                score=s,
                level=level_from_score(s),
            )
        )

    overall = round(sum(scores) / len(scores), 1) if scores else 0.0

    return ScoreResponse(
        overall_average=overall,
        overall_level=level_from_score(overall),
        per_question=per_q,
    )


# ============================================================
# Structured Coaching (unchanged, but safer)
# ============================================================
class CoachStructuredRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    model: str | None = None


class CoachStructuredResponse(BaseModel):
    ollama_ok: bool
    model: str
    why_weak: str
    how_to_improve: list[str]
    senior_version: str
    llm_error: str | None = None


def _normalize_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


@router.post("/coach-structured", response_model=CoachStructuredResponse)
def coach_structured(req: CoachStructuredRequest):
    c = OllamaClient()

    if not c.health_check():
        return CoachStructuredResponse(
            ollama_ok=False,
            model=req.model or c.model,
            why_weak="",
            how_to_improve=[],
            senior_version="",
            llm_error="Ollama not reachable. Start Ollama and try again.",
        )

    system = (
        "You are an interview coach for AI Developer / LLM roles.\n"
        "Return STRICT JSON ONLY. No markdown. No extra keys.\n"
        "Schema:\n"
        '{ "why_weak": string, "how_to_improve": [string,...], "senior_version": string }\n'
    )

    prompt = f"""
Question:
{req.question}

Candidate answer:
{req.answer}

Return ONLY valid JSON matching the schema exactly.
- why_weak: 1-3 sentences
- how_to_improve: 3-6 bullet strings
- senior_version: a strong, concise answer (4-8 sentences) using correct terminology
""".strip()

    try:
        data: dict[str, Any] = c.generate_json(
            prompt=prompt,
            system=system,
            temperature=0.2,
            max_tokens=800,
            model=req.model,
        )

        why_weak = str(data.get("why_weak") or "").strip()
        how_to_improve = _normalize_list(data.get("how_to_improve"))
        senior_version = str(data.get("senior_version") or "").strip()

        if not why_weak and not how_to_improve and not senior_version:
            return CoachStructuredResponse(
                ollama_ok=False,
                model=req.model or c.model,
                why_weak="",
                how_to_improve=[],
                senior_version="",
                llm_error="LLM returned empty/invalid JSON. Try again or switch model.",
            )

        return CoachStructuredResponse(
            ollama_ok=True,
            model=req.model or c.model,
            why_weak=why_weak,
            how_to_improve=how_to_improve,
            senior_version=senior_version,
            llm_error=None,
        )

    except Exception as e:
        return CoachStructuredResponse(
            ollama_ok=False,
            model=req.model or c.model,
            why_weak="",
            how_to_improve=[],
            senior_version="",
            llm_error=str(e),
        )
