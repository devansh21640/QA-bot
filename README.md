# Assignment 2 - Simple Policy Q&A Bot

This project implements Assignment 2 using the provided policy documents:
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`

## Retrieval Approach
- Hybrid retrieval: semantic embeddings + TF-IDF cosine similarity
- Deploy-safe fallback: if semantic model is unavailable, app auto-switches to TF-IDF mode
- Retrieval-first answering from policy text only

## Required Assignment Compliance
- Displays source document used to answer
- Returns exact fallback when information is missing:
  - `Information not available in policy documents.`
- No paid APIs
- CPU-only runtime (no GPU large model required)

## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start app:
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Community Cloud)
1. Push this folder to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Set:
   - Main file path: `app.py`
   - Python version: `3.12`
4. Click Deploy.

## Deployment-Ready Files in Repo
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`
