# ResuMatch — AI Resume-JD Matcher Agent

An AI-powered career advisor built with Google ADK and Gemini that instantly analyzes how well your resume matches a job description. Get a match score, gap analysis, and a personalized elevator pitch — all in one chat.

## Features

- **Smart Match Scoring** — 0-100% compatibility score
- **Strength Identification** — Skills and experience that align with the JD
- **Gap Analysis** — Missing skills with actionable suggestions
- **Personalized Pitch** — Ready-to-use elevator pitch tailored to the JD
- **ATS Keywords** — Keyword suggestions from the JD for your profile

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Google ADK |
| LLM | Gemini 2.5 Flash (Vertex AI) |
| Language | Python 3.12 |
| Deployment | Google Cloud Run |

## Project Structure

```
resumatch/
├── resumatch_agent/
│   ├── __init__.py       # Package init
│   ├── agent.py          # ADK Agent definition
│   └── .env              # Environment config
└── requirements.txt      # Dependencies
```

## Setup & Run Locally

### Prerequisites
- Python 3.12+
- Google Cloud project with billing enabled
- `gcloud` CLI authenticated

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd resumatch
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure Environment

Edit `resumatch_agent/.env`:
```env
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 3. Enable APIs

```bash
gcloud services enable aiplatform.googleapis.com
```

### 4. Run Locally

```bash
# Terminal mode
adk run resumatch_agent

# Web UI mode (recommended)
adk web
# Open http://localhost:8000
```

## Deploy to Cloud Run

```bash
# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  compute.googleapis.com

# Create service account
export PROJECT_ID=$(gcloud config get-value project)
export SA_NAME=resumatch-sa
export SERVICE_ACCOUNT=${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

gcloud iam service-accounts create ${SA_NAME} \
    --display-name="ResuMatch Service Account"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user"

# Deploy
adk deploy cloud_run \
  --project=$PROJECT_ID \
  --region=us-central1 \
  --service_name=resumatch \
  --service-account=$SERVICE_ACCOUNT
```

## Usage

1. Open the Cloud Run URL or `http://localhost:8000`
2. Select `resumatch_agent` from the agent dropdown
3. Paste your resume/LinkedIn profile text
4. Paste the job description
5. Get your match analysis with score, gaps, and personalized pitch!

## Cleanup

```bash
gcloud run services delete resumatch --region=us-central1 --quiet
gcloud artifacts repositories delete cloud-run-source-deploy --location=us-central1 --quiet
```
