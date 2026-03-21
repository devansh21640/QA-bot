# Assignment 2 - Simple Policy Q&A Bot

This project implements Assignment 2 using the provided policy documents:
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`

## Retrieval Approach
- Hybrid retrieval: semantic embeddings + TF-IDF cosine similarity
- Hybrid mode is mandatory in this app (semantic + TF-IDF always enabled)
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
   - Python version: `3.12` (pinned via `runtime.txt`)
4. Click Deploy.

### Note for Cloud Deploy
- This app installs `sentence-transformers` to run hybrid retrieval in deployment.
- First cloud build can take longer (typically 8-20 minutes) due dependency/model setup.

## Deployment-Ready Files in Repo
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `runtime.txt`
- `leave_policy.txt`
- `it_policy.txt`
- `travel_policy.txt`
