from __future__ import annotations

import re
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
USE_DRIVE = True
DRIVE_FILES = {
    "leave_policy.txt": "https://drive.google.com/file/d/17V-2ryfPQQ-KKhlLF_SCPMKVU2v8Kqyg/view?usp=sharing",
    "it_policy.txt": "https://drive.google.com/file/d/1FBWWZgAfYkBo5Z_I4RHSRraVnzaNh7XT/view?usp=sharing",
    "travel_policy.txt": "https://drive.google.com/file/d/1FqUPC2ExAw9dUOYTWzPRtP3SFL51y953/view?usp=sharing",
}

POLICY_FILES = {
    "leave_policy.txt": "Leave Policy",
    "it_policy.txt": "IT Policy",
    "travel_policy.txt": "Travel Policy",
}
FALLBACK_MESSAGE = "Information not available in policy documents."
SEMANTIC_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
RELEVANCE_THRESHOLD = 0.25

PROCESS_QUERY_PATTERN = re.compile(
    r"\b(how\s+do|how\s+can|how\s+to|what\s+is\s+the\s+process|"
    r"steps?\s+to|procedure\s+for|process\s+of|apply\s+for)\b",
    re.IGNORECASE,
)
HOW_MANY_PATTERN = re.compile(
    r"\b(how\s+many|how\s+much|what\s+is\s+the\s+(number|count|limit|maximum|total)|"
    r"how\s+long|number\s+of|total\s+number)\b",
    re.IGNORECASE,
)
PROCESS_DETAIL_PATTERN = re.compile(
    r"\b(explain|describe|tell\s+me\s+about|what\s+are\s+the\s+rules|"
    r"guidelines\s+for|rules\s+for|requirements\s+for)\b",
    re.IGNORECASE,
)

QUERY_EXPANSION = {
    "leave": ["leave", "entitlement", "days", "paid", "annual"],
    "leaves": ["leave", "entitlement", "days", "paid", "annual"],
    "vacation": ["leave", "paid", "annual"],
    "holiday": ["leave", "paid", "casual"],
    "free": ["paid", "leave", "entitlement", "days"],
    "sick": ["sick", "leave", "medical", "days"],
    "maternity": ["maternity", "leave", "weeks"],
    "casual": ["casual", "leave", "consecutive", "days"],
    "carry": ["carry", "forward", "unused", "leave"],
    "unused": ["unused", "carry", "forward", "leave"],
    "days": ["days", "leave", "entitlement", "per", "year"],
    "annual": ["yearly", "per", "year", "entitlement"],
    "travel": ["travel", "flight", "hotel", "reimbursement", "booking", "expense"],
    "flight": ["flight", "booking", "economy", "class"],
    "hotel": ["hotel", "reimbursement", "night", "limit"],
    "reimbursement": ["reimbursement", "claim", "expense", "reimburse"],
    "reimburse": ["reimbursement", "claim", "expense"],
    "expense": ["expense", "reimbursement", "personal", "claim"],
    "international": ["international", "travel", "approval", "managerial", "abroad", "overseas"],
    "abroad": ["international", "travel", "overseas", "managerial", "approval"],
    "overseas": ["international", "travel", "abroad", "managerial", "approval"],
    "local": ["local", "travel", "bills", "claim"],
    "bill": ["bills", "claim", "local", "travel"],
    "bills": ["bills", "claim", "local", "travel"],
    "vpn": ["vpn", "remote", "access", "mandatory"],
    "remote": ["remote", "vpn", "access", "work"],
    "password": ["password", "share", "employees"],
    "laptop": ["laptop", "company", "issued", "official"],
    "antivirus": ["antivirus", "software", "devices", "updated"],
    "usb": ["usb", "external", "devices", "approval"],
    "device": ["devices", "antivirus", "usb", "external"],
    "devices": ["devices", "antivirus", "usb", "external"],
    "approval": ["approval", "managerial", "international", "usb", "it", "permission"],
    "permission": ["approval", "managerial", "prior", "authorized", "authorisation"],
    "manager": ["managerial", "approval", "manager", "prior"],
    "managerial": ["managerial", "approval", "international"],
    "required": ["requires", "mandatory", "must", "approval", "prior"],
    "requires": ["required", "mandatory", "approval", "prior"],
    "process": ["process", "procedure", "how", "apply", "get"],
    "apply": ["apply", "process", "how", "take"],
    "policy": ["policy", "rule", "guideline"],
    "allowed": ["allowed", "mandatory", "must", "not"],
    "how": ["how", "process", "procedure", "step"],
    "many": ["many", "number", "entitlement", "days", "limit"],
    "much": ["many", "number", "entitlement", "days", "limit"],
    "limit": ["limit", "maximum", "entitlement", "cap"],
    "maximum": ["maximum", "limit", "consecutive", "cap"],
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "no",
    "so", "if", "as", "that", "this", "it", "its", "my", "your", "our",
    "their", "i", "we", "you", "he", "she", "they", "what", "which",
    "who", "how", "when", "where", "why", "me", "us", "him", "her", "them",
    "about", "than", "up", "out", "per", "must", "all", "any", "more",
    "also", "into", "through", "during", "before", "after", "there",
    "then", "just", "because", "while", "although",
}


@dataclass
class PolicyChunk:
    document_file: str
    document_name: str
    text: str


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    normalized: List[str] = []
    for token in tokens:
        if token == "leaves":
            token = "leave"
        if token not in STOPWORDS and len(token) >= 2:
            normalized.append(token)
    return normalized


def split_into_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        cleaned = re.sub(r"^\s*(?:\d+[\.)]\s*|[-•]\s*)", "", line).strip()
        if len(cleaned) <= 5:
            continue

        if cleaned.lower().endswith("policy") and len(cleaned.split()) <= 3:
            continue

        sentences.append(cleaned)
    return sentences


def parse_drive_url(share_url: str) -> str:
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", share_url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", share_url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"

    raise ValueError(f"Cannot parse Google Drive URL: {share_url}")


def load_policy_file(filename: str) -> str | None:
    if USE_DRIVE and filename in DRIVE_FILES:
        share_url = DRIVE_FILES[filename]
        try:
            direct_url = parse_drive_url(share_url)
            request = urllib.request.Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                content = response.read().decode("utf-8", errors="ignore")
            if content.strip():
                return content
        except Exception:
            pass

    local_path = BASE_DIR / filename
    if local_path.exists():
        content = local_path.read_text(encoding="utf-8", errors="ignore")
        if content.strip():
            return content

    return None


def build_tfidf_vector(tokens: List[str], idf_map: Dict[str, float], default_idf: float = 1.0) -> Dict[str, float]:
    if not tokens:
        return {}

    term_counts = Counter(tokens)
    total_terms = len(tokens)
    vector: Dict[str, float] = {}
    for token, count in term_counts.items():
        tf = count / total_terms
        vector[token] = tf * idf_map.get(token, default_idf)
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


def expand_query(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        for extra in QUERY_EXPANSION.get(token, []):
            if extra not in seen:
                expanded.append(extra)
                seen.add(extra)
    return expanded


def semantic_overlap_score(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0

    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    if not query_set or not chunk_set:
        return 0.0

    return (2 * len(query_set & chunk_set)) / (len(query_set) + len(chunk_set))


def check_process_query(question: str) -> str:
    if PROCESS_QUERY_PATTERN.search(question):
        return "process"
    if HOW_MANY_PATTERN.search(question):
        return "quantity"
    if PROCESS_DETAIL_PATTERN.search(question):
        return "detail"
    return "general"


@st.cache_data
def load_policy_chunks() -> List[PolicyChunk]:
    chunks: List[PolicyChunk] = []

    required_files = ["leave_policy.txt", "it_policy.txt", "travel_policy.txt"]
    for file_name in required_files:
        doc_name = POLICY_FILES[file_name]
        raw_text = load_policy_file(file_name)
        if raw_text is None:
            continue

        for sentence in split_into_sentences(raw_text):
            if sentence:
                chunks.append(
                    PolicyChunk(
                        document_file=file_name,
                        document_name=doc_name,
                        text=sentence,
                    )
                )

    return chunks


class TFIDFEngine:
    def __init__(self) -> None:
        self.corpus_texts: List[str] = []
        self.doc_labels: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []

    def fit(self, sentences: List[str], labels: List[str]) -> None:
        self.corpus_texts = sentences
        self.doc_labels = labels
        self.corpus_tokens = [tokenize(sentence) for sentence in sentences]
        total_docs = len(self.corpus_tokens)

        doc_freq: Counter = Counter()
        for tokens in self.corpus_tokens:
            doc_freq.update(set(tokens))

        self.idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
            for token, frequency in doc_freq.items()
        }

        self.tfidf_vectors = [build_tfidf_vector(tokens, self.idf, default_idf=1.0) for tokens in self.corpus_tokens]

    def _sparse_cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        return cosine_sim_sparse(vec_a, vec_b)

    def query_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        return build_tfidf_vector(query_tokens, self.idf, default_idf=0.5)

    def tfidf_scores(self, query_tokens: List[str]) -> List[float]:
        query_vec = self.query_vector(query_tokens)
        return [self._sparse_cosine(query_vec, vector) for vector in self.tfidf_vectors]


@st.cache_resource
def build_retriever():
    chunks = load_policy_chunks()
    engine = TFIDFEngine()
    engine.fit([chunk.text for chunk in chunks], [chunk.document_file for chunk in chunks])
    return chunks, engine

def pick_best_result(question: str, query_type: str, results: List[dict]) -> dict | None:
    if not results:
        return None

    query_tokens = set(tokenize(question))
    if query_type == "quantity" and ("leave" in query_tokens or "leaves" in query_tokens):
        specific_leave_types = {"sick", "maternity", "casual"}
        is_generic_leave_query = len(query_tokens.intersection(specific_leave_types)) == 0

        if is_generic_leave_query:
            disallowed_terms = {"sick", "maternity", "casual", "unused", "carry", "forward", "consecutive", "weeks"}

            for item in results:
                sentence_lower = item["sentence"].lower()
                if (
                    ("paid leaves" in sentence_lower or "paid leave" in sentence_lower)
                    and re.search(r"\b\d+\b", sentence_lower)
                    and "per year" in sentence_lower
                    and not any(term in sentence_lower for term in disallowed_terms)
                ):
                    return item

            for item in results:
                sentence_lower = item["sentence"].lower()
                if (
                    ("paid leaves" in sentence_lower or "paid leave" in sentence_lower)
                    and not any(term in sentence_lower for term in disallowed_terms)
                ):
                    return item

            for item in results:
                sentence_lower = item["sentence"].lower()
                if (
                    "entitled" in sentence_lower
                    and "leave" in sentence_lower
                    and "per year" in sentence_lower
                    and not any(term in sentence_lower for term in disallowed_terms)
                ):
                    return item

        for item in results:
            sentence_lower = item["sentence"].lower()
            if any(keyword in sentence_lower for keyword in ["entitled", "entitlement", "per year", "days per year", "paid leaves"]):
                return item

    return results[0]


def find_best_match(question: str):
    chunks, engine = build_retriever()
    if not chunks:
        return None

    query_type = check_process_query(question)
    expanded_tokens = expand_query(tokenize(question))
    tfidf_scores = engine.tfidf_scores(expanded_tokens)

    results: List[dict] = []
    for idx, chunk in enumerate(chunks):
        semantic_score = semantic_overlap_score(expanded_tokens, engine.corpus_tokens[idx])
        tfidf_score = tfidf_scores[idx]
        hybrid_score = (SEMANTIC_WEIGHT * semantic_score) + (TFIDF_WEIGHT * tfidf_score)

        if hybrid_score >= RELEVANCE_THRESHOLD:
            results.append(
                {
                    "sentence": chunk.text,
                    "doc_label": chunk.document_file,
                    "document": chunk.document_name,
                    "document_file": chunk.document_file,
                    "hybrid_score": round(hybrid_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "tfidf_score": round(tfidf_score, 4),
                }
            )

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    best = pick_best_result(question, query_type, results[:5])
    if not best:
        return None

    return {
        "answer": best["sentence"],
        "document": best["document"],
        "document_file": best["document_file"],
        "score": float(best["hybrid_score"]),
        "semantic_score": float(best["semantic_score"]),
        "tfidf_score": float(best["tfidf_score"]),
        "retrieval_mode": "hybrid",
        "query_type": query_type,
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
