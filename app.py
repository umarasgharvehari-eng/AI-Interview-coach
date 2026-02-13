import os
import re
from typing import List, Dict, Tuple

import streamlit as st
from groq import Groq

# -------------------------
# Config
# -------------------------
APP_TITLE = "AI Interview Coach (Groq + Llama)"
DEFAULT_MODEL = "llama-3.1-8b-instant"  # supported + fast
MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

SYSTEM_INTERVIEWER = """You are a professional interviewer and career coach.
You conduct structured, realistic mock interviews based on the candidate's job role.

Rules:
- Ask ONE question at a time.
- Wait for the candidate's answer before continuing.
- Ask smart follow-ups when relevant (based on the last answer).
- Keep it professional and friendly.
- Keep questions concise, but insightful.
- Do NOT provide feedback until the interview ends.
- Target 5–7 questions total unless user ends early.
"""

SYSTEM_FEEDBACK = """You are an expert interview evaluator.
Given the full interview transcript, produce:
1) Overall rating (0-10)
2) Confidence score (0-100%)
3) Strengths (bullets)
4) Areas for improvement (bullets)
5) Communication (clarity, structure, concision)
6) Technical depth (role-specific)
7) Suggested next steps (bullets)

Be constructive, specific, and actionable.
Output in clear Markdown.

IMPORTANT:
- Output ONLY the report (no preface).
- Include the numeric rating and confidence score clearly.
"""

# -------------------------
# Backend helpers
# -------------------------
def get_client() -> Groq:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Set it in Streamlit Secrets or environment variables."
        )
    return Groq(api_key=api_key)

def groq_chat(
    client: Groq,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 600,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def clean_question(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(question\s*[:\-]\s*)", "", text, flags=re.I).strip()
    return text

def transcript_from_turns(turns: List[Dict[str, str]]) -> str:
    """
    turns list example:
      {"role":"assistant","content":"Q1"}
      {"role":"user","content":"A1"}
      ...
    """
    lines = []
    for t in turns:
        if t["role"] == "assistant":
            lines.append(f"Interviewer: {t['content']}")
        else:
            lines.append(f"Candidate: {t['content']}")
    return "\n".join(lines).strip()

def build_first_question(role: str, difficulty: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INTERVIEWER},
        {"role": "user", "content": f"""
Start a mock interview for the role: {role}
Difficulty level: {difficulty}

Ask the FIRST interview question now.
Ask only the question (no feedback, no multi-question lists).
""".strip()},
    ]

def build_next_question(role: str, difficulty: str, transcript: str, asked: int, max_q: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INTERVIEWER},
        {"role": "user", "content": f"""
Continue this mock interview.

Role: {role}
Difficulty: {difficulty}
Questions asked so far: {asked}/{max_q}

Transcript so far:
{transcript}

Ask the NEXT single interview question.
Prefer a follow-up if the last answer suggests it; otherwise move forward.
Ask only ONE question.
""".strip()},
    ]

def build_feedback(role: str, difficulty: str, transcript: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_FEEDBACK},
        {"role": "user", "content": f"""
Evaluate this mock interview.

Role: {role}
Difficulty: {difficulty}

Full transcript:
{transcript}
""".strip()},
    ]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎤", layout="wide")
st.title("🎤 AI Interview Coach")
st.caption("Mock interview powered by Groq (API key hidden in backend).")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    max_questions = st.slider("Max Questions", 5, 10, 7, 1)
    st.divider()
    st.subheader("Deployment note")
    st.write("Set `GROQ_API_KEY` in Streamlit Secrets (recommended).")

# Initialize session state
if "role" not in st.session_state:
    st.session_state.role = ""
if "turns" not in st.session_state:
    st.session_state.turns = []  # list of {"role":..., "content":...}
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Interview")
    role = st.text_input("Job Role", value=st.session_state.role, placeholder='e.g., "Software Developer"')

    btn_row = st.columns(3)
    start_clicked = btn_row[0].button("▶️ Start Interview", use_container_width=True)
    end_clicked = btn_row[1].button("⏹️ End & Get Feedback", use_container_width=True)
    reset_clicked = btn_row[2].button("🔄 Reset", use_container_width=True)

    # Actions
    if reset_clicked:
        st.session_state.role = ""
        st.session_state.turns = []
        st.session_state.interview_active = False
        st.session_state.questions_asked = 0
        st.session_state.feedback = ""
        st.rerun()

    if start_clicked:
        if not role.strip():
            st.error("Please enter a job role.")
        else:
            st.session_state.role = role.strip()
            st.session_state.turns = []
            st.session_state.feedback = ""
            st.session_state.questions_asked = 0
            st.session_state.interview_active = True

            try:
                client = get_client()
                q1 = groq_chat(client, build_first_question(st.session_state.role, difficulty), model=model, temperature=0.6, max_tokens=250)
                q1 = clean_question(q1)
                st.session_state.turns.append({"role": "assistant", "content": q1})
                st.session_state.questions_asked = 1
                st.success("Interview started.")
                st.rerun()
            except Exception as e:
                st.session_state.interview_active = False
                st.error(f"Backend error: {e}")

    # Render chat history
    for t in st.session_state.turns:
        with st.chat_message("assistant" if t["role"] == "assistant" else "user"):
            st.markdown(t["content"])

    # User answer input
    if st.session_state.interview_active:
        user_answer = st.chat_input("Type your answer and press Enter...")
        if user_answer:
            st.session_state.turns.append({"role": "user", "content": user_answer})

            # auto-end if reached max
            if st.session_state.questions_asked >= max_questions:
                st.session_state.interview_active = False
                st.info("Max questions reached. Generating feedback...")
            else:
                # ask next question
                try:
                    client = get_client()
                    transcript = transcript_from_turns(st.session_state.turns)
                    next_q = groq_chat(
                        client,
                        build_next_question(st.session_state.role, difficulty, transcript, st.session_state.questions_asked, max_questions),
                        model=model,
                        temperature=0.6,
                        max_tokens=250,
                    )
                    next_q = clean_question(next_q)
                    st.session_state.turns.append({"role": "assistant", "content": next_q})
                    st.session_state.questions_asked += 1
                except Exception as e:
                    st.session_state.interview_active = False
                    st.error(f"Backend error: {e}")

            st.rerun()

    if end_clicked and (st.session_state.turns or st.session_state.interview_active):
        st.session_state.interview_active = False
        st.info("Generating feedback...")
        try:
            client = get_client()
            transcript = transcript_from_turns(st.session_state.turns)
            fb = groq_chat(
                client,
                build_feedback(st.session_state.role or role or "Unknown Role", difficulty, transcript),
                model=model,
                temperature=0.4,
                max_tokens=900,
            )
            st.session_state.feedback = (fb or "").strip()
            st.rerun()
        except Exception as e:
            st.error(f"Backend error: {e}")

with col2:
    st.subheader("Summary & Feedback")
    st.write(f"**Role:** {st.session_state.role or '—'}")
    st.write(f"**Difficulty:** {difficulty}")
    st.write(f"**Questions Asked:** {st.session_state.questions_asked}/{max_questions}")
    st.divider()

    if st.session_state.feedback:
        st.markdown(st.session_state.feedback)
    else:
        st.info("Feedback will appear here after ending the interview.")

# Footer
st.caption("Tip: On Streamlit Cloud, set GROQ_API_KEY in Secrets to keep it hidden.")
