from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
POLICY_FILES = {
    "leave_policy.txt": "Leave Policy",
    "it_policy.txt": "IT Policy",
    "travel_policy.txt": "Travel Policy",
}
FALLBACK_MESSAGE = "Information not available in policy documents."
SEMANTIC_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
BASE_SCORE_THRESHOLD = 0.12
PROCESS_QUERY_PATTERN = re.compile(r"\b(how\s+(to|can|do)|step[-\s]?by[-\s]?step|process|procedure|kaise)\b", re.IGNORECASE)
QUANT_QUERY_PATTERN = re.compile(r"\bhow\s+many\b", re.IGNORECASE)
PROCESS_DETAIL_PATTERN = re.compile(
    r"\b(step|submit|apply|form|portal|workflow|email|request|approval chain|approver)\b",
    re.IGNORECASE,
)

QUERY_EXPANSIONS = {
    "manager": ["managerial", "approval authority"],
    "managerial": ["manager", "approval authority"],
    "approval": ["approve", "authorization", "sign-off"],
    "approve": ["approval", "authorization", "sign-off"],
    "remote": ["work from home", "wfh"],
    "reimbursement": ["claim", "refund"],
    "international": ["abroad", "overseas"],
    "abroad": ["international", "overseas"],
    "overseas": ["international", "abroad"],
    "travel": ["trip", "journey"],
    "leave": ["time off", "vacation"],
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "that", "the", "to", "was", "what", "when", "where",
    "who", "why", "with", "do", "does", "can", "should", "necessary", "need", "required", "going",
}


@dataclass
class PolicyChunk:
    document_file: str
    document_name: str
    text: str


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_tfidf_vector(tokens: List[str], idf_map: Dict[str, float]) -> Dict[str, float]:
    if not tokens:
        return {}

    term_counts = Counter(tokens)
    total_terms = len(tokens)
    vector: Dict[str, float] = {}
    for token, count in term_counts.items():
        if token not in idf_map:
            continue
        tf = count / total_terms
        vector[token] = tf * idf_map[token]
    return vector


def cosine_sim_sparse(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot = 0.0
    for key, value in vec_a.items():
        dot += value * vec_b.get(key, 0.0)

    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_token_set(question: str) -> Set[str]:
    lowered = question.lower()
    tokens = set(tokenize(question))
    expanded = {t for t in tokens if t not in STOPWORDS and len(t) > 2}

    for term, replacements in QUERY_EXPANSIONS.items():
        replacement_tokens = set()
        for replacement in replacements:
            replacement_tokens.update(tokenize(replacement))

        if term in lowered or term in tokens or any(rep in lowered for rep in replacements) or any(rt in tokens for rt in replacement_tokens):
            expanded.add(term)

    return expanded


def semantic_overlap_score(query_tokens: Set[str], chunk_tokens: Set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0

    chunk_filtered = {token for token in chunk_tokens if token not in STOPWORDS and len(token) > 2}
    if not chunk_filtered:
        return 0.0

    intersection = len(query_tokens.intersection(chunk_filtered))
    coverage = intersection / max(1, len(query_tokens))
    precision = intersection / max(1, len(chunk_filtered))
    return min(1.0, (0.75 * coverage) + (0.25 * precision))


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
    tokenized_chunks: List[List[str]] = [tokenize(chunk.text) for chunk in chunks]

    doc_freq: Counter = Counter()
    for tokens in tokenized_chunks:
        doc_freq.update(set(tokens))

    total_docs = len(tokenized_chunks)
    idf_map: Dict[str, float] = {}
    for token, frequency in doc_freq.items():
        idf_map[token] = math.log((1 + total_docs) / (1 + frequency)) + 1.0

    chunk_tfidf_vectors = [build_tfidf_vector(tokens, idf_map) for tokens in tokenized_chunks]
    chunk_token_sets = [set(tokens) for tokens in tokenized_chunks]

    return idf_map, chunk_tfidf_vectors, chunk_token_sets


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

    is_quant_query = QUANT_QUERY_PATTERN.search(question) is not None
    is_process_query = PROCESS_QUERY_PATTERN.search(question) is not None

    if is_process_query and not is_quant_query:
        return PROCESS_DETAIL_PATTERN.search(answer_text) is not None

    return True


def find_best_match(question: str):
    chunks = load_policy_chunks()
    if not chunks:
        return None

    idf_map, chunk_tfidf_vectors, chunk_token_sets = build_retriever(chunks)
    query_variants = build_query_variants(question)

    tfidf_scores = [0.0] * len(chunks)
    semantic_scores = [0.0] * len(chunks)

    for variant in query_variants:
        query_tokens = tokenize(variant)
        query_tfidf_vector = build_tfidf_vector(query_tokens, idf_map)
        expanded_tokens = semantic_token_set(variant)

        for idx in range(len(chunks)):
            tfidf_score = cosine_sim_sparse(query_tfidf_vector, chunk_tfidf_vectors[idx])
            semantic_score = semantic_overlap_score(expanded_tokens, chunk_token_sets[idx])
            tfidf_scores[idx] = max(tfidf_scores[idx], tfidf_score)
            semantic_scores[idx] = max(semantic_scores[idx], semantic_score)

    blended_scores = [
        (SEMANTIC_WEIGHT * semantic_scores[idx]) + (TFIDF_WEIGHT * tfidf_scores[idx])
        for idx in range(len(chunks))
    ]

    best_idx = max(range(len(chunks)), key=lambda i: blended_scores[i])
    best_score = float(blended_scores[best_idx])

    best_chunk = chunks[best_idx]
    if not is_answer_adequate(question, best_chunk.text, best_score):
        return None

    return {
        "answer": best_chunk.text,
        "document": best_chunk.document_name,
        "document_file": best_chunk.document_file,
        "score": best_score,
        "semantic_score": float(semantic_scores[best_idx]),
        "tfidf_score": float(tfidf_scores[best_idx]),
        "retrieval_mode": "hybrid",
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
            <span class="badge">Hybrid retrieval: semantic expansion + TF-IDF</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2.2, 1])

    with col1:
        with st.form("policy_qa_form", clear_on_submit=False):
            question = st.text_input(
                "Enter your question",
                placeholder="Example: Is VPN mandatory for remote work?",
            )
            ask_clicked = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

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
                    st.caption(
                        f"Hybrid score: {match['score']:.3f} | "
                        f"Semantic: {match['semantic_score']:.3f} | "
                        f"TF-IDF: {match['tfidf_score']:.3f}"
                    )

    with col2:
        st.subheader("Available Policies")
        st.markdown("- Leave Policy")
        st.markdown("- IT Policy")
        st.markdown("- Travel Policy")
        st.divider()
        st.caption("If an answer is not present in the documents, the app returns the required fallback message.")


if __name__ == "__main__":
    render_ui()
