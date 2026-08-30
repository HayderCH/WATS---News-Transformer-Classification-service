# Deployment Guide

This document captures the reference deployment flow for the News Topic Intelligence service and dashboard. It covers local development, staging, and production roll-outs along with the supporting observability and operational playbooks.

## 1. System Overview

The platform is composed of three logical components that deploy together:

1. **FastAPI inference API (`app/main.py`)** – serves classification, summarization, metrics, trends, and review workflow endpoints.
2. **Background & Typer tooling (`scripts/manage.py`)** – database migrations, seeding, training, evaluation, and artifact bundling CLI.
3. **Streamlit operator console (`dashboard/streamlit_app.py`)** – human-in-the-loop review experience and health dashboards.

All components share the same database (`DB_URL`) and model artifact directory (`MODEL_DIR`). The API exposes `/metrics` for lightweight observability and accepts API-key headers for write-sensitive operations.

## 2. Environment Matrix

| Environment    | Purpose                                        | Differences                                                                             | Secrets                                                                                       | Observability                                             |
| -------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Local**      | Developer iteration, demos, manual QA          | SQLite `data/feedback.db`, models stored on host filesystem, MLflow logging to `mlruns` | `.env` sourced from `.env.example`                                                            | Console logs, `/metrics` endpoint                         |
| **Staging**    | Pre-production validation, integration testing | Containerized stack, remote Postgres (RDS/Azure SQL), artifact store on S3/Blob         | Secrets injected via orchestrator (GitHub Environments, Azure Key Vault, AWS Secrets Manager) | Centralized logs + traces via OpenTelemetry collector     |
| **Production** | External access, live human-in-loop            | Auto-scaling API replicas, managed database, managed object storage, CDN on Streamlit   | Secrets rotated automatically; read-only service principals for models                        | Metrics scraped by Prometheus, alerts via PagerDuty/Teams |

## 3. Container Images

- **Base image:** `python:3.11-slim` (multi-stage build recommended for production).
- **Build stage:** install poetry/pip, compile wheels, download transformer artifacts if baking into the image.
- **Runtime stage:** non-root user, copy application code + wheels, install only runtime dependencies, expose port `8001`.
- **Entrypoint:** `uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers ${UVICORN_WORKERS:-4}`.
- **Healthcheck:** `CMD curl --fail http://localhost:8001/health || exit 1` (add to the Dockerfile when targeting ECS/AKS).

## 4. Configuration Management

1. Copy `.env.example` to `.env` locally; for remote environments inject individual environment variables instead of uploading the file.
2. Required secrets:
   - `API_KEY` (write operations) or `API_KEYS` (semicolon list for multiple keys).
   - Database credentials (`DB_URL`) – use managed Postgres in non-local environments.
   - Optional S3-style credentials for artifact pushes.
   - MLflow tracking URI + token if using a remote tracking server.
   - Hugging Face revision pins (`SUMMARIZER_MODEL_REVISION`, `HF_MODEL_REVISION`, `HF_DATASET_AG_NEWS_REVISION`, `HF_DATASET_JSON_REVISION`) to guarantee reproducible model/dataset downloads.
3. Store secrets in the orchestrator: GitHub Environments, AWS Secrets Manager, Azure Key Vault, or HashiCorp Vault.
4. Never commit populated `.env` files; rely on `.gitignore` safeguards and secret scanning in CI.

## 5. Deployment Recipes

### 5.1 Local Docker Compose

```powershell
# Build and run API + Streamlit (defined in docker-compose.yml)
docker compose up --build
```

- Uses the same `.env` values; mount `data/` and `models/` volumes for persistence.
- Alembic migrations run automatically on start via the `docker-entrypoint.sh` script.
- Streamlit defaults to port `8501` and expects the API on `http://api:8001` internally.

### 5.2 GitHub Actions → Container Registry → ECS/AKS

1. CI (`.github/workflows/ci.yml`) runs lint + pytest for every push/pr.
2. Add a `deploy.yml` workflow that:
   - Builds the Docker image (tagged with git SHA).
   - Pushes to ECR (AWS) or ACR (Azure).
   - Triggers the infrastructure deploy step (`aws ecs update-service` or `az containerapp update`).
3. Use environment protection rules to require approvals before production deployments.

### 5.3 Manual Staging Smoke Test

```powershell
# Port-forward staging API service
azure containerapp ingress show ...
# or
aws ecs execute-command ...

# Verify health and metrics
Invoke-RestMethod -Headers @{ 'x-api-key' = '<staging-key>' } -Uri https://staging.yourdomain.tld/health
Invoke-RestMethod -Headers @{ 'x-api-key' = '<staging-key>' } -Uri https://staging.yourdomain.tld/metrics
```

Ensure a human reviewer can label items through the Streamlit dashboard using staging credentials prior to promoting artifacts to production.

## 6. Observability & Alerting

- **Logging:** `LOG_JSON=1` produces structured logs; ship to CloudWatch, Log Analytics, or ELK. Capture `x-request-id` for correlation.
- **Metrics:** Expose `/metrics` JSON; pair with a sidecar (Prometheus exporter or OpenTelemetry collector) to convert to Prometheus format. Alert on request error rates and latency percentiles.
- **Tracing (optional):** Integrate `opentelemetry-instrumentation-fastapi` and export spans to OTLP collector.
- **Runbooks:** Document restart procedures, environment variable defaults, and common errors in `RUNBOOK.md` (recommended future work).

## 7. Database & Migrations

- Apply migrations with `alembic upgrade head` during deployment before starting API workers.
- For blue/green or rolling deploys, ensure database migrations are backward compatible.
- Schedule nightly backups (Postgres pg_dump or managed snapshots) and test restore procedures regularly.

## 8. Security Checklist

- Rotate API keys quarterly; prefer `API_KEYS` multi-key format to enable transition periods.
- Enforce HTTPS via load balancer / Application Gateway; redirect HTTP to HTTPS.
- Restrict outbound egress for API containers to trusted domains (Hugging Face, MLflow, object storage).
- Enable vulnerability scans on container images (e.g., Trivy, Amazon Inspector, Azure Defender).
- Run `bandit` and dependency scans (e.g., `pip-audit`, `safety`) in CI for early warning.

## 9. Next Steps

- Automate load tests with k6/Locust and capture baselines.
- Add infrastructure-as-code (Terraform/Bicep/CloudFormation) to manage networking, databases, and scaling rules.
- Implement canary or shadow deploys for new model artifacts before flipping traffic.

---

For questions or environment-specific runbooks, open an issue or extend this guide with team-specific details.
