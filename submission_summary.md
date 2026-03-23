# Assignment 2 – Policy Q&A Bot (Final Summary)

## Live Deployment (Top Priority)
**Live demo:** https://policy-bot-devansh.streamlit.app/

## Executive Snapshot
This implementation is a **retrieval-first, policy-grounded Q&A bot** that answers strictly from provided policy documents and blocks unsupported responses with an exact fallback.

**Exact fallback (as required):**
`Information not available in policy documents.`

## Objective
Build a simple and auditable policy assistant that:
- always retrieves evidence before answering,
- shows policy source in every valid response,
- rejects weak/irrelevant matches using threshold + intent guardrails,
- remains lightweight (CPU-only, no paid APIs).

## Policy Corpus
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`

## Ingestion & Reliability Design
To ensure robust execution across notebook/local/cloud runs:
1. Load policy text from Google Drive share URLs (primary path).
2. Fall back to local files if Drive access fails.
3. Raise clear runtime error if no policy data is available.

This dual-ingestion setup improves portability and minimizes demo failures.

## End-to-End Retrieval Architecture
1. **Text Preprocessing**
  - regex tokenization,
  - lowercasing and normalization,
  - stopword filtering,
  - policy-line cleanup (e.g., removes list prefixes like `1.` / `4)`).

2. **Chunking**
  - each non-empty policy line is treated as a searchable chunk.

3. **Scoring Layer**
  - TF-IDF vectorization with sparse cosine similarity,
  - semantic overlap score (Dice-style token overlap).

4. **Hybrid Ranking**

\[
	ext{Hybrid Score} = 0.7 \times \text{Semantic} + 0.3 \times \text{TF-IDF}
\]

5. **Selection & Filtering**
  - relevance threshold: `RELEVANCE_THRESHOLD = 0.25`,
  - intent-aware expansion for synonym/wording variation,
  - ambiguity controls for quantity-focused leave queries,
  - reject low-confidence candidate and return exact fallback.

## Intent Handling & Safety Guardrails
The query classifier routes prompts into: `process`, `quantity`, `detail`, `general`.

Guardrails applied:
- procedural question without procedural evidence → fallback,
- low match confidence → fallback,
- responses are kept concise and limited to policy-grounded text.

These constraints reduce hallucination risk and keep the bot policy-compliant.

## Output Contract
For each user query, the system returns:
- best answer line,
- source policy/document,
- Hybrid, Semantic, and TF-IDF scores.

Generated files:
- `assignment2_results.csv`
- `assignment2_results.json`

## Compliance Checklist
- ✅ Retrieval-first response generation
- ✅ Exact fallback string when evidence is unavailable
- ✅ Source attribution with each answer
- ✅ CPU-only, no paid API dependency
- ✅ Reproducible outputs and downloadable result artifacts

## Key Insights from Evaluation
1. Hybrid scoring performs better than TF-IDF-only for synonym-rich, short policy queries.
2. Intent expansion significantly improves practical recall for real-world question phrasing.
3. Threshold + adequacy checks are critical for trustworthy fallback behavior.
4. Dual ingestion improves reliability in notebook demos and deployment environments.

## Limitations and Improvement Roadmap
Current limitations:
- line-level chunks may lose multi-line policy context,
- typo handling is limited,
- no explicit policy section hierarchy.

Suggested upgrades:
- sentence-window and section-aware chunking,
- typo-tolerant token matching,
- metadata-aware retrieval by policy title/section,
- threshold calibration on a validation set for better precision-recall balance.

## Conclusion
The final solution satisfies assignment expectations with a clear retrieval-first pipeline, explicit source grounding, strict fallback compliance, and deployment-ready simplicity.

## Deployment Details
- Platform: Streamlit Community Cloud
- Entrypoint: `app.py`
- Live application: https://policy-bot-devansh.streamlit.app/