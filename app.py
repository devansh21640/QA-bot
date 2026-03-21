from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
POLICY_FILES = {
    "leave_policy.txt": "Leave Policy",
    "it_policy.txt": "IT Policy",
    "travel_policy.txt": "Travel Policy",
}
FALLBACK_MESSAGE = "Information not available in policy documents."
SEMANTIC_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
BASE_SCORE_THRESHOLD = 0.25
PROCESS_QUERY_PATTERN = re.compile(r"\b(how|step[-\s]?by[-\s]?step|process|procedure|kaise)\b", re.IGNORECASE)
PROCESS_DETAIL_PATTERN = re.compile(
    r"\b(step|submit|apply|form|portal|workflow|email|request|approval chain|approver)\b",
    re.IGNORECASE,
)

QUERY_EXPANSIONS = {
    "manager": ["managerial", "approval authority"],
    "approval": ["approve", "authorization", "sign-off"],
    "remote": ["work from home", "wfh"],
    "reimbursement": ["claim", "refund"],
    "international": ["abroad", "overseas"],
    "leave": ["time off", "vacation"],
}


@dataclass
class PolicyChunk:
    document_file: str
    document_name: str
    text: str


@st.cache_data
def load_policy_chunks() -> List[PolicyChunk]:
    chunks: List[PolicyChunk] = []

    for file_name, doc_name in POLICY_FILES.items():
        file_path = BASE_DIR / file_name
        if not file_path.exists():
            continue

        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        for line in lines:
            if line.lower().endswith("policy"):
                continue
            cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
            if cleaned:
                chunks.append(
                    PolicyChunk(
                        document_file=file_name,
                        document_name=doc_name,
                        text=cleaned,
                    )
                )

    return chunks


@st.cache_resource
def build_retriever(chunks: List[PolicyChunk]):
    corpus = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    model = None
    semantic_matrix = None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        semantic_matrix = model.encode(corpus, normalize_embeddings=True)
    except Exception:
        model = None
        semantic_matrix = None

    return vectorizer, tfidf_matrix, model, semantic_matrix


def build_query_variants(question: str) -> List[str]:
    lowered = question.lower().strip()
    variants = {lowered}

    for term, replacements in QUERY_EXPANSIONS.items():
        if term in lowered:
            for replacement in replacements:
                variants.add(lowered.replace(term, replacement))

    return list(variants)


def is_answer_adequate(question: str, answer_text: str, score: float) -> bool:
    if score < BASE_SCORE_THRESHOLD:
        return False

    if PROCESS_QUERY_PATTERN.search(question):
        return PROCESS_DETAIL_PATTERN.search(answer_text) is not None

    return True


def find_best_match(question: str):
    chunks = load_policy_chunks()
    if not chunks:
        return None

    vectorizer, tfidf_matrix, model, semantic_matrix = build_retriever(chunks)
    query_variants = build_query_variants(question)

    tfidf_scores = None
    semantic_scores = None
    semantic_available = model is not None and semantic_matrix is not None

    for variant in query_variants:
        q_tfidf_vec = vectorizer.transform([variant])
        current_tfidf = cosine_similarity(q_tfidf_vec, tfidf_matrix).flatten()

        if semantic_available:
            q_semantic_vec = model.encode([variant], normalize_embeddings=True)
            current_semantic = cosine_similarity(q_semantic_vec, semantic_matrix).flatten()
        else:
            current_semantic = None

        if tfidf_scores is None:
            tfidf_scores = current_tfidf
            semantic_scores = current_semantic if semantic_available else np.zeros_like(current_tfidf)
        else:
            tfidf_scores = np.maximum(tfidf_scores, current_tfidf)
            if semantic_available:
                semantic_scores = np.maximum(semantic_scores, current_semantic)

    if semantic_available:
        blended_scores = (SEMANTIC_WEIGHT * semantic_scores) + (TFIDF_WEIGHT * tfidf_scores)
    else:
        blended_scores = tfidf_scores

    best_idx = int(blended_scores.argmax())
    best_score = float(blended_scores[best_idx])

    best_chunk = chunks[best_idx]
    if not is_answer_adequate(question, best_chunk.text, best_score):
        return None

    return {
        "answer": best_chunk.text,
        "document": best_chunk.document_name,
        "document_file": best_chunk.document_file,
        "score": best_score,
        "semantic_score": float(semantic_scores[best_idx]) if semantic_available else 0.0,
        "tfidf_score": float(tfidf_scores[best_idx]),
        "retrieval_mode": "hybrid" if semantic_available else "tfidf_only",
    }


def render_ui() -> None:
    st.set_page_config(
        page_title="Policy Q&A Bot",
        page_icon="📄",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .hero {
                border-radius: 14px;
                padding: 1.2rem 1.4rem;
                background: linear-gradient(120deg, #f8fafc, #eef2ff);
                border: 1px solid #e2e8f0;
                margin-bottom: 1rem;
            }
            .badge {
                display: inline-block;
                background: #e0e7ff;
                color: #3730a3;
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                font-size: 0.85rem;
                margin-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Simple Policy Q&A Bot</h2>
            <p style="margin:0.4rem 0 0 0;">
                Ask questions from Leave, IT, and Travel policies. Answers are retrieved from policy documents only.
            </p>
            <span class="badge">Hybrid: Semantic Embeddings + TF-IDF</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2.2, 1])

    with col1:
        question = st.text_input(
            "Enter your question",
            placeholder="Example: Is VPN mandatory for remote work?",
        )

        ask_clicked = st.button("Get Answer", type="primary", use_container_width=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                match = find_best_match(question)
                if not match:
                    st.error(FALLBACK_MESSAGE)
                else:
                    st.success(match["answer"])
                    st.info(f"Source document: {match['document']} ({match['document_file']})")
                    if match["retrieval_mode"] == "hybrid":
                        st.caption(
                            f"Hybrid score: {match['score']:.3f} | "
                            f"Semantic: {match['semantic_score']:.3f} | "
                            f"TF-IDF: {match['tfidf_score']:.3f}"
                        )
                    else:
                        st.caption(f"Fallback mode: TF-IDF only | Score: {match['tfidf_score']:.3f}")

    with col2:
        st.subheader("Available Policies")
        st.markdown("- Leave Policy")
        st.markdown("- IT Policy")
        st.markdown("- Travel Policy")
        st.divider()
        st.caption("If an answer is not present in the documents, the app returns the required fallback message.")


if __name__ == "__main__":
    render_ui()
