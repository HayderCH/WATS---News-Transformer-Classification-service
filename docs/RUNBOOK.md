# Operations Runbook

This runbook covers the day-to-day operational procedures for the News Topic Intelligence platform. It is designed for on-call engineers responding to incidents or performing routine maintenance.

## 1. Service Map

| Component                                | Description                                                                    | Key Ports                   | Deployment Notes                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------- |
| FastAPI API (`uvicorn app.main:app`)     | Serves classification, summarization, review, metrics, and feedback endpoints. | 8001 (HTTP)                 | Horizontal scale via multiple Uvicorn workers or container replicas behind a load balancer. |
| Streamlit Dashboard (`streamlit_app.py`) | Operator console for manual review, trends, and metrics.                       | 8501 (HTTP)                 | Typically internal-only; authenticate via VPN or SSO proxy.                                 |
| Database (`DB_URL`)                      | Stores feedback, review queue, and labeling state.                             | Postgres 5432 / SQLite file | Production uses managed Postgres with automated backups.                                    |
| MLflow Tracking (`MLFLOW_TRACKING_URI`)  | Optional experiment tracking backend.                                          | 5000 (default)              | Disable with `MLFLOW_ENABLED=0` if not required.                                            |

## 2. Health Checks

1. **Liveness:** `GET /health`
   - Returns `{"status": "ok"}` on success.
   - If failing, verify process status and recent deployment history.
2. **Metrics:** `GET /metrics`
   - JSON payload with route counters and latency histograms.
   - Ensure metrics are scraping successfully; absence indicates telemetry issue.
3. **Streamlit:** Navigate to `/` on the dashboard host; check sidebar shows API connectivity indicator.

## 3. Common Alerts & Playbooks

| Alert                | Symptom                               | Diagnostic Steps                                                                                          | Mitigation                                                                     |
| -------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| High 5xx rate        | Error spikes on API                   | Check recent deployments, review `/metrics` for affected routes, inspect logs filtered by `x-request-id`. | Roll back latest deployment or restart pods; confirm database connectivity.    |
| Review queue backlog | Pending review items exceed threshold | Call `/review/active-learning` to inspect counts and labels; check Streamlit reviewers availability.      | Trigger active fine-tune via CLI or allocate additional reviewers.             |
| Slow responses (>2s) | Latency alarm                         | Inspect `/metrics` latency histograms, review CPU/memory dashboards, verify model artifact sizes.         | Scale out workers, warm up model weights on start, consider enabling batching. |
| API key failures     | 401 responses on protected routes     | Validate `API_KEY`/`API_KEYS`, inspect auth logs for stale clients.                                       | Rotate key, update clients, ensure load balancers pass headers.                |

## 4. Incident Response Timeline

1. **Acknowledge alert** within 5 minutes.
2. **Triage**
   - Identify affected routes and scope (single tenant vs all).
   - Capture relevant `x-request-id` values for correlation.
3. **Mitigate**
   - Roll back or redeploy healthy version.
   - Scale resources vertically/horizontally if demand surge.
4. **Communicate**
   - Update status channel with ETA and actions.
   - File incident ticket (include timelines, impact, root cause).
5. **Post-incident**
   - Backfill data if gaps created (e.g., missed feedback submissions).
   - Schedule postmortem within 48 hours.

## 5. Operational Tasks

### 5.1 Rotate API Keys

1. Generate new token and append to `API_KEYS` env var separated by semicolon.
2. Redeploy API with updated env var.
3. Notify clients to switch to new key.
4. Remove old key after confirmation (max 24 hours).

### 5.2 Database Maintenance (Postgres)

- Backups: Verify automated snapshot schedule weekly.
- Vacuum/Analyze: Run `VACUUM ANALYZE` monthly if using self-hosted Postgres.
- Migrations: Apply via `alembic upgrade head` prior to deploying code expecting new schema.

### 5.3 Model Artifact Promotion

1. Use `python scripts/manage.py bundle-artifacts --label <tag>` to produce archive.
2. Upload to artifact store (`--push`) if the deployment uses remote storage.
3. Update environment variable `CLASSIFIER_VERSION` if versioning is encoded in settings.
4. Restart API pods to load new model weights.

## 6. Troubleshooting Guide

- **API returns 422 on `/classify_news`:** Ensure request body matches schema; verify clients updated after schema changes.
- **Streamlit dashboard can't reach API:** Confirm `API_BASE_URL` sidebar setting, check network ACLs, and ensure API CORS configuration allows the dashboard host.
- **MLflow client failures:** Inspect `MLFLOW_TRACKING_URI` for network/firewall issues; disable MLflow if non-critical using `MLFLOW_ENABLED=0` until resolved.

## 7. Escalation Matrix

| Severity | Criteria                                            | Primary           | Backup        |
| -------- | --------------------------------------------------- | ----------------- | ------------- |
| SEV-1    | Complete outage, data loss risk                     | On-call engineer  | Tech lead     |
| SEV-2    | Degraded experience (latency, partial feature loss) | On-call engineer  | ML engineer   |
| SEV-3    | Non-urgent bugs, tooling failures                   | File GitHub issue | Product owner |

## 8. Change Management

- All production changes require:
  1. Passing CI (lint + pytest).
  2. Staging verification with `/health` and `/metrics` checks.
  3. Approval from product/tech lead for production deploys.
- Maintain changelog entries for user-visible modifications.

## 9. Future Enhancements

- Add synthetic monitoring (k6 ping or cron job calling `/health`).
- Integrate PagerDuty/Teams webhook for automated alert routing.
- Extend runbook with database restore drill outcomes and Streamlit SSO procedures.

---

Keep this runbook updated as the platform evolves. Submit pull requests for any improvements discovered during real incidents.
