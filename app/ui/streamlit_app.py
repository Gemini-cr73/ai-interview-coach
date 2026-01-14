# app/ui/streamlit_app.py
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DEFAULT_API_URL = os.getenv("AI_INTERVIEW_COACH_API_URL", "http://127.0.0.1:8000")

# Dev tools are hidden unless you explicitly enable them:
# PowerShell (current session):  $env:AI_INTERVIEW_COACH_DEV="1"
# Persistent (Windows):          setx AI_INTERVIEW_COACH_DEV 1
DEV_MODE = os.getenv("AI_INTERVIEW_COACH_DEV", "0").strip() == "1"

st.set_page_config(page_title="AI Interview Coach", layout="wide")
st.title("AI Interview Coach")
st.caption(
    "Local interview practice for AI Developer / LLM Applications (no paid API)."
)


# -----------------------------
# HTTP Helpers
# -----------------------------
def api_get(base_url: str, path: str, timeout: int = 15) -> tuple[bool, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


def api_post(
    base_url: str, path: str, payload: Any, timeout: int = 60
) -> tuple[bool, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    r = None
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        if not r.text.strip():
            return True, {}
        return True, r.json()
    except Exception as e:
        if r is not None:
            return False, f"{e}\n\nResponse:\n{r.text}"
        return False, str(e)


# -----------------------------
# Response Normalizers
# -----------------------------
def normalize_questions_response(
    resp: Any,
) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """
    Accept response shapes:
      A) {"questions": [{"question": "...", "keywords": [...], "tech_terms":[...]} ...]}
      B) [{"question": "...", "keywords": [...], "tech_terms":[...]} ...]
      C) {"questions": ["...", "..."]}
      D) ["...", "..."]

    Returns:
      questions: [str]
      specs: {question_text: {"keywords":[...], "tech_terms":[...]}}
    """
    items: list[Any] = []
    if isinstance(resp, list):
        items = resp
    elif isinstance(resp, dict):
        val = resp.get("questions") or resp.get("items") or resp.get("data") or []
        items = val if isinstance(val, list) else []
    else:
        items = []

    questions: list[str] = []
    specs: dict[str, dict[str, list[str]]] = {}

    for it in items:
        if isinstance(it, str):
            q = it.strip()
            if q:
                questions.append(q)
                specs[q] = {"keywords": [], "tech_terms": []}
            continue

        if isinstance(it, dict):
            q = (it.get("question") or it.get("text") or "").strip()
            if not q:
                continue

            keywords = (
                it.get("keywords")
                or it.get("expected_keywords")
                or it.get("key_terms")
                or []
            )
            tech_terms = (
                it.get("tech_terms")
                or it.get("techTerms")
                or it.get("expected_tech_terms")
                or []
            )

            questions.append(q)
            specs[q] = {
                "keywords": list(keywords) if isinstance(keywords, list) else [],
                "tech_terms": list(tech_terms) if isinstance(tech_terms, list) else [],
            }

    return questions, specs


# -----------------------------
# Copy Button
# -----------------------------
def copy_button_html(
    text_to_copy: str, button_label: str = "Copy improved answer"
) -> None:
    safe = json.dumps(text_to_copy)
    html = f"""
    <div style="margin: 0.25rem 0 0.75rem 0;">
      <button
        style="
          background: #0e1117;
          color: white;
          border: 1px solid #2a2f3a;
          padding: 8px 12px;
          border-radius: 8px;
          cursor: pointer;
        "
        onclick="navigator.clipboard.writeText({safe});"
      >
        ✅ {button_label}
      </button>
    </div>
    """
    st.components.v1.html(html, height=60)


# -----------------------------
# Session State
# -----------------------------
if "questions" not in st.session_state:
    st.session_state.questions: list[str] = []

if "question_instances" not in st.session_state:
    # Each entry: {"id": "md5_1", "text": "question text"}
    st.session_state.question_instances: list[dict[str, str]] = []

if "question_specs" not in st.session_state:
    st.session_state.question_specs: dict[str, dict[str, list[str]]] = {}

if "answers" not in st.session_state:
    # IMPORTANT: answers now keyed by instance_id (not question text)
    st.session_state.answers: dict[str, str] = {}

if "score_results" not in st.session_state:
    st.session_state.score_results: dict[str, Any] = {}

if "coaching_cache" not in st.session_state:
    # cache keyed by question text (fine)
    st.session_state.coaching_cache: dict[str, Any] = {}

if "last_error" not in st.session_state:
    st.session_state.last_error = ""

if "target_difficulty" not in st.session_state:
    st.session_state.target_difficulty = "Mixed"

if "backend_accepted_level" not in st.session_state:
    st.session_state.backend_accepted_level = True

# Dev-only state
if "debug_last_payload" not in st.session_state:
    st.session_state.debug_last_payload = None

if "debug_last_response" not in st.session_state:
    st.session_state.debug_last_response = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    num_q = st.selectbox("Number of questions", [3, 5], index=0)

    difficulty = st.selectbox(
        "Question difficulty", ["Junior", "Mid", "Senior", "Mixed"], index=3
    )
    st.session_state.target_difficulty = difficulty

    use_llm = st.toggle("LLM-Enhanced Coaching (Ollama)", value=True)
    ollama_model = st.selectbox("Ollama model", ["llama3.1:8b", "llama3.2:3b"], index=0)

    st.caption(f"API: {api_url}")

    if DEV_MODE:
        st.divider()
        st.subheader("Developer Tools")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("API Health"):
                ok, data = api_get(api_url, "/health")
                st.success(data if ok else data)

        with c2:
            if st.button("Ollama Health"):
                ok, data = api_get(api_url, "/llm/health")
                st.success(data if ok else data)

        show_debug = st.checkbox("Show debug panel", value=False)
    else:
        show_debug = False


# -----------------------------
# Layout
# -----------------------------
colA, colB = st.columns([1.2, 1.0], gap="large")


# -----------------------------
# Scoring helpers
# -----------------------------
def build_items_list() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for inst in st.session_state.question_instances:
        q_text = inst["text"]
        q_id = inst["id"]
        specs = st.session_state.question_specs.get(q_text, {}) or {}
        items.append(
            {
                "question": q_text,
                "answer": (st.session_state.answers.get(q_id, "") or "").strip(),
                "keywords": specs.get("keywords", []) or [],
                "tech_terms": specs.get("tech_terms", []) or [],
            }
        )
    return items


def try_score_submission(base_url: str) -> tuple[bool, Any, Any]:
    payload = {"items": build_items_list()}
    ok, resp = api_post(base_url, "/score-submission", payload, timeout=60)
    return ok, resp, payload


# -----------------------------
# Left: Questions + Answers
# -----------------------------
with colA:
    st.subheader("Interview Questions")
    st.caption(f"Target difficulty: **{st.session_state.target_difficulty}**")

    if st.button("Generate Questions"):
        st.session_state.last_error = ""
        st.session_state.score_results = {}
        st.session_state.coaching_cache = {}
        st.session_state.backend_accepted_level = True

        payload_with_level = {
            "num": int(num_q),
            "level": st.session_state.target_difficulty.lower(),
        }
        ok, resp = api_post(api_url, "/questions", payload_with_level, timeout=60)

        if not ok and (
            "level" in str(resp).lower() or "extra fields" in str(resp).lower()
        ):
            st.session_state.backend_accepted_level = False
            ok, resp = api_post(api_url, "/questions", {"num": int(num_q)}, timeout=60)

        if not ok:
            st.session_state.last_error = str(resp)
        else:
            questions, specs = normalize_questions_response(resp)
            if not questions:
                st.session_state.last_error = (
                    "Questions endpoint returned no questions. "
                    "Open /docs and confirm POST /questions response format."
                )
            else:
                st.session_state.questions = questions
                st.session_state.question_specs = specs

                # ✅ Build unique instance IDs to prevent DuplicateWidgetID
                counts: dict[str, int] = {}
                instances: list[dict[str, str]] = []
                for q in questions:
                    counts[q] = counts.get(q, 0) + 1
                    q_md5 = hashlib.md5(q.encode("utf-8")).hexdigest()[:12]
                    inst_id = (
                        f"{q_md5}_{counts[q]}"  # e.g. abc123def456_1, abc123def456_2
                    )
                    instances.append({"id": inst_id, "text": q})

                st.session_state.question_instances = instances

                # ✅ Keep only answers that still exist (by instance id)
                existing_ids = {x["id"] for x in instances}
                st.session_state.answers = {
                    k: v
                    for k, v in st.session_state.answers.items()
                    if k in existing_ids
                }

                # Ensure keys exist
                for inst in instances:
                    st.session_state.answers.setdefault(inst["id"], "")

    if not st.session_state.backend_accepted_level:
        st.warning(
            "Your backend does not appear to accept a 'level' parameter for /questions yet, "
            "so the UI cannot force Junior/Mid/Senior question generation."
        )

    if st.session_state.last_error:
        st.error(f"Failed to generate questions: {st.session_state.last_error}")
    elif not st.session_state.question_instances:
        st.info("Click Generate Questions to begin.")

    for idx, inst in enumerate(st.session_state.question_instances, start=1):
        q_text = inst["text"]
        q_id = inst["id"]

        label = st.session_state.target_difficulty.upper()
        st.markdown(f"**Q{idx} ({label}). {q_text}**")

        # ✅ Unique, stable widget key (no more DuplicateWidgetID)
        st.session_state.answers[q_id] = st.text_area(
            f"Your answer for Q{idx}",
            value=st.session_state.answers.get(q_id, ""),
            key=f"answer_{q_id}",
            height=110,
            placeholder="Type your answer here...",
        )


# -----------------------------
# Right: Feedback & Scoring
# -----------------------------
with colB:
    st.subheader("Feedback & Scoring")
    st.caption(
        "Note: **Overall Level** is your *performance grade*, not the question difficulty."
    )

    if st.button("Score My Answers"):
        st.session_state.last_error = ""

        ok, resp, payload_used = try_score_submission(api_url)
        st.session_state.debug_last_payload = payload_used
        st.session_state.debug_last_response = resp

        if ok:
            st.session_state.score_results = resp or {}
        else:
            st.session_state.last_error = str(resp)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    results = st.session_state.score_results or {}

    overall_avg = results.get("overall_average")
    overall_level = results.get("overall_level")

    if overall_avg is not None:
        st.markdown(f"### Overall Average: {float(overall_avg):.1f}%")
    else:
        st.markdown("### Overall Average: 0.0%")

    if overall_level:
        st.markdown(f"**Overall Level: {overall_level}**")

    st.divider()

    per_q = results.get("per_question") or []
    if isinstance(per_q, dict):
        per_q = [{"question": k, **v} for k, v in per_q.items()]

    for i, row in enumerate(per_q, start=1):
        q_text = (row.get("question") or "").strip()
        if not q_text:
            continue

        score = row.get("score")
        level = row.get("level")

        st.markdown(f"### Q{i} Score: {score}% · Level: {level}")

        with st.expander("LLM-Enhanced Coaching (Structured JSON)", expanded=False):
            if not use_llm:
                st.info(
                    "Enable LLM-Enhanced Coaching (Ollama) in the sidebar to use this."
                )
            else:
                # Find the FIRST instance matching this question (good enough for now)
                inst_id = None
                for inst in st.session_state.question_instances:
                    if inst["text"] == q_text:
                        inst_id = inst["id"]
                        break

                answer_text = (
                    st.session_state.answers.get(inst_id, "") if inst_id else ""
                ).strip()

                if not answer_text:
                    st.warning("Add an answer first, then generate coaching.")
                else:
                    if st.button(
                        f"Generate structured coaching for Q{i}", key=f"coach_btn_{i}"
                    ):
                        payload = {
                            "question": q_text,
                            "answer": answer_text,
                            "model": ollama_model,
                            "target_level": st.session_state.target_difficulty.lower(),
                        }
                        okc, coach_resp = api_post(
                            api_url, "/coach-structured", payload, timeout=120
                        )
                        if okc:
                            st.session_state.coaching_cache[q_text] = coach_resp
                        else:
                            st.error(str(coach_resp))

                    coach = st.session_state.coaching_cache.get(q_text)
                    if coach:
                        why_weak = coach.get("why_weak") or ""
                        how_to_improve = coach.get("how_to_improve") or []
                        senior_version = coach.get("senior_version") or ""

                        st.subheader("Why this answer is weak")
                        st.write(why_weak if why_weak else "—")

                        st.subheader("How to improve")
                        if isinstance(how_to_improve, list) and how_to_improve:
                            for b in how_to_improve:
                                st.write(f"- {b}")
                        else:
                            st.write("—")

                        st.subheader("Senior-level version")
                        if senior_version:
                            st.code(senior_version, language="text")
                            copy_button_html(senior_version, "Copy improved answer")
                        else:
                            st.write("—")

        st.divider()

    if DEV_MODE and show_debug:
        st.subheader("Debug")
        st.write("Last payload used:")
        st.code(
            json.dumps(st.session_state.debug_last_payload, indent=2), language="json"
        )
        st.write("Last response:")
        try:
            st.code(
                json.dumps(st.session_state.debug_last_response, indent=2),
                language="json",
            )
        except Exception:
            st.write(st.session_state.debug_last_response)
