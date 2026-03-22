# Assignment 2: Policy Q&A Bot

## Overview
This project implements a policy question-answering system using the provided documents:
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`

The app follows a **retrieval-first** approach and answers only from policy content.

## Key Features
- Hybrid retrieval: **semantic intent expansion + TF-IDF similarity**
- Displays the **source document** used for answer generation
- Returns exact fallback when information is not found:
  - `Information not available in policy documents.`
- Clean Streamlit UI for evaluation/demo
- CPU-friendly implementation (no paid APIs, no GPU models)

## Retrieval Design
1. Parse policy rules into searchable chunks.
2. Build query variants using intent/synonym expansion.
3. Compute:
   - semantic overlap score
   - TF-IDF score
4. Combine scores using weighted hybrid scoring:
   - `Hybrid = 0.7 * Semantic + 0.3 * TF-IDF`
5. Apply adequacy checks:
   - confident match required
   - procedural queries without process details return fallback

## Assignment Compliance Checklist
- ✅ Retrieval from policy documents before answering
- ✅ Exact fallback response for missing information
- ✅ Source document shown with each answer
- ✅ No paid API usage
- ✅ Runs on CPU

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Use:
   - Main file path: `app.py`
   - Branch: `main`
4. Deploy.

## Live Demo
- https://policy-bot-devansh.streamlit.app/

## Sample Validation Queries
- `Is VPN mandatory for remote work?`
- `How many sick leaves can employee take?`
- `What is hotel reimbursement cap?`
- `How to apply for approval step-by-step?` → should return fallback

## Project Files
- `app.py` — Streamlit app and retrieval logic
- `requirements.txt` — dependencies
- `runtime.txt` / `.python-version` — runtime hints
- `leave_policy.txt`, `it_policy.txt`, `travel_policy.txt` — source policies
