# Sakhi Clinical Companion — Python FastAPI Backend (Google Cloud Run)

Standalone Python FastAPI backend for **Sakhi** (Multilingual Reproductive Healthcare AI Companion), prepared for deployment as **`backend-service`** on **Google Cloud Run** (`asia-south1`).

## Key Capabilities Preserved
- Pre-Care curated medical knowledge RAG + Verified Doctor escalation when data is absent
- Post-Care zero-trust symptom triage + Emergency caretaker staff escalation
- Multilingual support (English, Telugu, Hindi, Tinglish)
- Google Gemini integration (`GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` via `GEMINI_API_KEY`)
- Firebase Admin SDK (Application Default Credentials on Cloud Run — no JSON key file in production), Firebase ID token verification, cross-user ownership checks, and Cloud Firestore write-through persistence
- Cloud Run health endpoint (`GET /health`) with zero live LLM dependencies
- Structured non-PHI JSON logging for Google Cloud Logging

## Local Development
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Run Automated Tests
```bash
python3 -m unittest discover -s tests -v
```

## Docker Build & Run (Cloud Run Emulation on Port 8080)
```bash
docker build -t sakhi-backend:local .
docker run --rm -p 8080:8080 \
  -e PORT=8080 \
  -e APP_ENV=production \
  -e CORS_ORIGINS=http://localhost:5173 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  sakhi-backend:local
```

## Google Cloud Run Deployment (`asia-south1` / `backend-service`)
```bash
gcloud run deploy backend-service \
  --source . \
  --region asia-south1 \
  --platform managed \
  --service-account sakhi-backend-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --set-env-vars "APP_ENV=production,GOOGLE_CLOUD_PROJECT=<PROJECT_ID>,FIREBASE_PROJECT_ID=<PROJECT_ID>,GEMINI_MODEL=gemini-4-hydrogen,GEMINI_FALLBACK_MODEL=gemini-3.1-flash-lite,CORS_ORIGINS=https://<FRONTEND_CLOUD_RUN_URL>,REQUIRE_FIREBASE_AUTH=true" \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,WEBHOOK_SECRET=WEBHOOK_SECRET:latest"
```
