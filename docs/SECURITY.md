# Security Posture

This document captures the security controls and outstanding follow-up items for the News Topic Intelligence service.

## Routine Checks

| Control                             | Location                                                                                                    | Frequency                         | Notes                                                                                                                                             |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Static application security testing | `bandit -r app scripts dashboard -ll`                                                                       | Every PR (`security` job)         | Surfaces FastAPI routing and Hugging Face usage issues. Local execution mirrors CI.                                                               |
| Dependency vulnerability scanning   | `pip-audit --format markdown --ignore-vuln GHSA-wf7f-8fxf-xfxc -r requirements.txt -r requirements-dev.txt` | Every PR (`dependency-audit` job) | Reports known CVEs across both runtime (`requirements.txt`) and tooling (`requirements-dev.txt`); MLflow advisory temporarily waived (see below). |
| Linting & style                     | `ruff check`                                                                                                | Every PR (`lint` job)             | Helps catch unsafe patterns (e.g., eval usage) and keeps code uniform.                                                                            |
| Tests                               | `pytest`                                                                                                    | Every PR (`test` job)             | Validates functional behaviour after security changes.                                                                                            |

## Hugging Face Hardening

- Model and tokenizer downloads are pinned via environment variables: `SUMMARIZER_MODEL_REVISION`, `HF_MODEL_REVISION`.
- Dataset ingestion is pinned with `HF_DATASET_AG_NEWS_REVISION` and `HF_DATASET_JSON_REVISION`.
- Runtime loads from local artifact directories use `local_files_only=True` with explicit `# nosec B615` annotations to indicate the risk has been mitigated.

## Dependency Posture (October 2025)

Runtime and developer dependencies were upgraded on 2025-10-02 to remediate all CVEs previously reported by `pip-audit`:

- `fastapi 0.118.0` paired with `starlette 0.47.2` addresses GHSA-f96h-pmfr-66vw / GHSA-2c2j-9gv5-cj73.
- `mlflow 3.1.0`, `requests 2.32.4`, `scikit-learn 1.5.1`, and `transformers 4.53.2` resolve their respective advisories.
- `pyarrow 17.0.0` is explicitly pinned to cover the indirect dataset dependency advisory.

`pip-audit` now runs as part of CI/CD without the `continue-on-error` safeguard. Treat audit failures as release blockers and backport patches promptly.

### Accepted Risk: GHSA-wf7f-8fxf-xfxc (MLflow)

- **Status:** Accepted (temporary)
- **Details:** `mlflow==3.1.0` is affected by GHSA-wf7f-8fxf-xfxc. All currently compatible upstream releases either mandate `protobuf>=6` (breaking `streamlit==1.39.0`) or have not yet published a patched build.
- **Mitigation:** CI passes `--ignore-vuln GHSA-wf7f-8fxf-xfxc` to `pip-audit`. Track the upstream issue and upgrade to the first patched `mlflow` release that keeps protobuf `<6`. Re-run `pip-audit` without the ignore flag to verify closure.
- **Owner:** Security on-call (HayderCH). Review monthly or when MLflow releases change log mentions the advisory.

## Security TODOs

- [x] Upgrade affected dependencies and remove the `continue-on-error` flag from the `dependency-audit` job.
- [ ] Add integration tests to ensure upgraded libraries do not regress ML workflows.
- [ ] Evaluate container image scanning (Trivy / Grype) once Docker publishing is introduced.
- [ ] Configure secret scanning alert routing (GitHub Advanced Security or equivalent).
- [ ] Periodically rotate API keys and enforce TLS termination at the ingress layer (documented in `docs/DEPLOYMENT.md`).

---

Please update this document whenever security controls change or vulnerabilities are mitigated.
