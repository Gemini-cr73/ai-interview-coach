"""
AI Developer / LLM Interview Engine
Rule-based scoring + optional LLM coaching layer handled by API.

This engine supports:
- Static built-in question bank
- Dynamic question specs passed from the API (for Ollama-generated questions)

Scoring:
- Keyword coverage
- Depth
- Clarity
- Seniority alignment
- Tips
"""

from __future__ import annotations

import re
from typing import Any

# -----------------------------
# Helpers
# -----------------------------


def _normalize(text: str) -> str:
    """Lowercase + normalize whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize_words(text: str) -> list[str]:
    """Basic word tokenizer (letters/numbers + apostrophes)."""
    return re.findall(r"[a-zA-Z0-9']+", _normalize(text))


def _count_sentences(text: str) -> int:
    """Heuristic sentence count."""
    t = (text or "").strip()
    if not t:
        return 0
    parts = re.split(r"[.!?]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))


def _has_structure(text: str) -> bool:
    """Detect bullet/numbered structure or multi-paragraph."""
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return True
    return bool(re.search(r"(^|\n)\s*([-*]|\d+\.)\s+", text))


def _keyword_hits(answer: str, keywords: list[str]) -> dict[str, Any]:
    """
    Count keyword hits:
    - supports phrase keywords (contains)
    - supports simple stems (e.g., "train" matches "training")
    """
    a_norm = _normalize(answer)
    a_words = set(_tokenize_words(answer))

    matched: list[str] = []
    for kw in keywords:
        kw_norm = _normalize(kw)
        if not kw_norm:
            continue

        # phrase match
        if " " in kw_norm and kw_norm in a_norm:
            matched.append(kw)
            continue

        # word/stem match
        if kw_norm in a_words:
            matched.append(kw)
            continue

        if len(kw_norm) >= 4 and any(w.startswith(kw_norm) for w in a_words):
            matched.append(kw)
            continue

    return {
        "matched": matched,
        "hit_count": len(matched),
        "required": len(keywords),
    }


# -----------------------------
# Interview Question Bank
# -----------------------------
QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "Explain the difference between prompt engineering and fine-tuning for LLMs.",
        "keywords": ["prompt", "fine-tuning", "training", "context", "parameters"],
        "tech_terms": [
            "inference",
            "weights",
            "gradient",
            "dataset",
            "alignment",
            "rlhf",
            "lora",
            "adapter",
        ],
        "topic": "LLM Foundations",
        "level": "Mid",
    },
    {
        "question": "What is a vector database and why is it useful in RAG applications?",
        "keywords": ["embedding", "similarity search", "faiss", "chromadb", "semantic"],
        "tech_terms": [
            "cosine",
            "nearest neighbors",
            "index",
            "chunk",
            "retriever",
            "metadata",
            "hybrid",
        ],
        "topic": "RAG",
        "level": "Mid",
    },
    {
        "question": "How does Retrieval-Augmented Generation improve factual accuracy?",
        "keywords": ["retrieval", "documents", "hallucination", "knowledge base"],
        "tech_terms": [
            "citations",
            "grounding",
            "context window",
            "reranker",
            "source",
            "prompt injection",
        ],
        "topic": "RAG",
        "level": "Mid",
    },
    {
        "question": "Explain what tokenization is and how it affects transformer model performance.",
        "keywords": ["tokens", "bpe", "vocabulary", "context window"],
        "tech_terms": [
            "subword",
            "byte-pair",
            "compression",
            "sequence length",
            "latency",
            "throughput",
        ],
        "topic": "LLM Foundations",
        "level": "Mid",
    },
    {
        "question": "What are some security risks specific to LLM applications?",
        "keywords": ["prompt injection", "data leakage", "security", "guardrails"],
        "tech_terms": [
            "jailbreak",
            "exfiltration",
            "policy",
            "red teaming",
            "pii",
            "sandbox",
            "rate limiting",
        ],
        "topic": "Security",
        "level": "Mid",
    },
]


def get_interview_questions(num: int = 3) -> list[str]:
    """
    Backward compatible: returns question strings only (built-in bank).
    """
    num = max(1, min(int(num), len(QUESTIONS)))
    return [q["question"] for q in QUESTIONS[:num]]


def _get_bank_match(question: str) -> dict[str, Any] | None:
    return next((q for q in QUESTIONS if q["question"] == question), None)


# -----------------------------
# Scoring Engine
# -----------------------------
def score_answer(
    question: str,
    answer: str,
    question_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Multi-factor scoring:
    - keyword_score: coverage of required keywords
    - depth_score: length + technical density + explanation signals
    - clarity_score: structure + sentence length heuristics
    - seniority: Junior / Mid / Senior (based on overall)
    - overall score (0-100) returned as `score` for backward compatibility

    question_spec: Optional rubric dict that may include:
      - keywords: list[str]
      - tech_terms: list[str]
    Used for Ollama-generated questions not in the bank.
    """
    match = _get_bank_match(question)
    if not match and question_spec:
        # Use dynamic spec for scoring (hybrid / Ollama-generated questions)
        match = {
            "question": question,
            "keywords": list(question_spec.get("keywords") or []),
            "tech_terms": list(question_spec.get("tech_terms") or []),
        }

    if not match:
        return {
            "score": 0,
            "feedback": "Unknown question (no rubric provided).",
            "keywords_matched": 0,
            "keywords_required": 0,
            "keyword_score": 0,
            "depth_score": 0,
            "clarity_score": 0,
            "seniority": "Junior",
            "tips": [
                "Generate questions through the app so a scoring rubric is attached."
            ],
            "matched_keywords": [],
            "missing_keywords": [],
        }

    answer = (answer or "").strip()
    if not answer:
        return {
            "score": 0,
            "feedback": "No answer provided.",
            "keywords_matched": 0,
            "keywords_required": len(match["keywords"]),
            "keyword_score": 0,
            "depth_score": 0,
            "clarity_score": 0,
            "seniority": "Junior",
            "tips": ["Write a short answer first, then score again."],
            "matched_keywords": [],
            "missing_keywords": list(match["keywords"]),
        }

    # ---- Keyword coverage ----
    kh = _keyword_hits(answer, match["keywords"])
    hit_count = kh["hit_count"]
    required = kh["required"]
    keyword_score = int(round((hit_count / max(1, required)) * 100))

    missing_keywords = [k for k in match["keywords"] if k not in kh["matched"]]

    # ---- Depth score ----
    words = _tokenize_words(answer)
    wc = len(words)

    length_score = min(100, int(round((min(wc, 140) / 120) * 100)))

    explain_markers = [
        "because",
        "therefore",
        "so that",
        "for example",
        "e.g",
        "in practice",
        "tradeoff",
        "however",
    ]
    a_norm = _normalize(answer)
    explain_hits = sum(1 for m in explain_markers if m in a_norm)
    explain_score = min(100, explain_hits * 20)

    tech_terms = [t.lower() for t in match.get("tech_terms", [])]
    tech_hit = sum(1 for t in tech_terms if t in a_norm)
    tech_score = min(100, tech_hit * 15)

    depth_score = int(
        round(0.55 * length_score + 0.25 * tech_score + 0.20 * explain_score)
    )

    # ---- Clarity score ----
    sentences = _count_sentences(answer)
    structured = _has_structure(answer)
    avg_sent_len = wc / max(1, sentences)

    if avg_sent_len <= 18:
        runon_penalty = 0
    elif avg_sent_len <= 28:
        runon_penalty = 10
    else:
        runon_penalty = 25

    structure_bonus = 10 if structured else 0
    sentence_ok_bonus = 10 if 2 <= sentences <= 6 else 0

    clarity_raw = 70 + structure_bonus + sentence_ok_bonus - runon_penalty
    clarity_score = max(0, min(100, int(round(clarity_raw))))

    # ---- Overall score ----
    overall = int(
        round(0.45 * keyword_score + 0.35 * depth_score + 0.20 * clarity_score)
    )

    # ---- Seniority ----
    if overall >= 80 and depth_score >= 70 and keyword_score >= 75:
        seniority = "Senior"
    elif overall >= 55:
        seniority = "Mid"
    else:
        seniority = "Junior"

    # ---- Tips ----
    tips: list[str] = []

    if keyword_score < 60:
        if missing_keywords:
            tips.append(
                f"Include more key terms: {', '.join(missing_keywords[:5])}{'...' if len(missing_keywords) > 5 else ''}"
            )
        tips.append("Use the question’s core vocabulary (definitions + contrasts).")

    if depth_score < 60:
        tips.append(
            "Add 1–2 concrete details (how it works, tradeoffs, example, or workflow)."
        )
        tips.append("Explain the ‘why’ and ‘when’ — not only the ‘what’.")

    if clarity_score < 70:
        tips.append(
            "Use 2–4 short sentences or bullets (definition → mechanism → example → tradeoff)."
        )
        tips.append("Avoid run-on sentences; keep each sentence focused on one idea.")

    if not tips:
        tips.append(
            "Strong answer — add a brief example or production consideration to be extra impressive."
        )

    if overall >= 85:
        feedback = "Excellent — strong coverage, depth, and clarity."
    elif overall >= 70:
        feedback = "Very good — solid answer. A bit more depth or precision would push it higher."
    elif overall >= 50:
        feedback = "Decent start — you hit some key ideas, but expand with more technical specifics."
    else:
        feedback = "Needs improvement — focus on core concepts, add key terms, and structure your explanation."

    return {
        "score": overall,
        "feedback": feedback,
        "keywords_matched": hit_count,
        "keywords_required": required,
        "keyword_score": keyword_score,
        "depth_score": depth_score,
        "clarity_score": clarity_score,
        "seniority": seniority,
        "tips": tips,
        "matched_keywords": kh["matched"],
        "missing_keywords": missing_keywords,
    }


def run_engine(
    payload: dict[str, Any],
    use_llm: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Keeps API compatibility:
    - Always returns the rule-based scoring result.
    - LLM layer is handled by separate /coach endpoint.
    """
    question = str(payload.get("question", ""))
    answer = str(payload.get("answer", ""))

    question_spec = payload.get("question_spec")
    if question_spec and not isinstance(question_spec, dict):
        question_spec = None

    return score_answer(question, answer, question_spec=question_spec)
