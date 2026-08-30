# News Topic Classification + Abstractive Summarization API

## 1. One‑Sentence Pitch

Build a FastAPI microservice that classifies news articles into predefined topics and generates concise abstractive summaries, while tracking topic trends over time and enabling future extensions like bias detection and article clustering.

---

## 2. Why This Project (Relevance & Story)

Content flows (media portals, blog networks, newsletters, dashboards) need:

- Automated topic tagging for categorization, routing, or personalization.
- Summaries for faster editorial review / user previews.
- Trend aggregation (which topics spike when).
  This is a clean, production-style applied NLP system using public datasets—zero scraping hurdles, fast to prototype, and extensible to recommendation, clustering, or moderation.

You position it as: “I deliver end-to-end intelligent content services—classification + summarization + analytics, not just a notebook model.”

---

## 3. Datasets (Where & How to Get Them)

### 3.1 Primary Options

| Dataset                            | Purpose                                                                | Size                              | Source                                                       | Notes                           |
| ---------------------------------- | ---------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------- |
| News Category Dataset (HuffPost)   | Multi-topic classification (business, entertainment, world news, etc.) | ~200k rows                        | Kaggle: ‘News Category Dataset’ by Rishabh Misra             | Rich categories, some imbalance |
| AG News                            | Simpler 4‑class baseline (World, Sports, Business, Sci/Tech)           | 120k train, 7.6k test             | Kaggle: ‘AG News Classification’ OR Hugging Face: `ag_news`  | Good for quick baselines        |
| (Optional) CNN/DailyMail Summaries | Pretrained summarization alignment                                     | Hugging Face: `cnn_dailymail`     | Use only for zero-shot; main dataset articles lack summaries |
| (Optional) BBC News                | Extra domain generalization                                            | Kaggle: ‘BBC News Classification’ | Smaller, for transfer tests                                  |

### 3.2 Acquisition Steps

#### A. Kaggle (CLI)

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (chmod 600).
3. Download:
   ```bash
   kaggle datasets download -d rmisra/news-category-dataset
   unzip news-category-dataset.zip -d data/raw/huffpost
   kaggle datasets download -d aghosh001/ag-news-classification-dataset
   unzip ag-news-classification-dataset.zip -d data/raw/ag_news
   ```

#### B. Hugging Face Datasets

```python
from datasets import load_dataset
ds = load_dataset("ag_news")
```

### 3.3 Dataset Selection Strategy

- Start with AG News for a 1‑day baseline (simple 4 classes).
- Move to HuffPost for richer taxonomy (≈40 categories).
- Optionally collapse infrequent HuffPost categories into “other” if long-tail noise hurts macro F1.

### 3.4 Creating Summaries

HuffPost dataset does NOT include human summaries. Strategies:

1. Use article `short_description` as a pseudo-reference summary (store as `ref_summary`).
2. Evaluate model summary vs `short_description` with ROUGE/L scores (approximate, not perfect).
3. For long body text (some entries short), optionally concatenate title + short_description as input to summarizer for demonstration of compression.
4. (Optional) Curate 50 manual “clean” summaries for a tiny human evaluation set (improves credibility).

---

## 4. Core Tasks

| Task                                  | Description                         | Models                                                                                                                        |
| ------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Topic Classification                  | Predict category (multi-class)      | Baseline: TF-IDF + Logistic Regression; Advanced: Fine-tune `distilbert-base-uncased` or `xlm-roberta-base` (if multilingual) |
| Abstractive Summarization             | Generate 1–3 sentence summary       | Pretrained: `sshleifer/distilbart-cnn-12-6` or `facebook/bart-large-cnn`                                                      |
| Trend Aggregation                     | Track frequency of topics over time | Daily/weekly grouping of classified inputs                                                                                    |
| (Optional) Similarity / Clustering    | Group related articles              | Sentence embeddings (`sentence-transformers`) + HDBSCAN or k-means                                                            |
| (Optional) Bias / Sentiment Indicator | Category + sentiment blend          | Add lightweight sentiment classifier                                                                                          |

---

## 5. High-Level Architecture

```
          +-------------------+
Request → |  FastAPI Service  | → Classification Model (loaded)
          |   /classify_news  | → Summarizer (BART / DistilBART)
          |   /ab_test        | → A/B Testing Service
          |   /summarize      | → Embeddings (optional)
          |   /trends         |
          +---------+---------+
                    |
                    v
             PostgreSQL (articles, predictions, logs, trends, experiments)
                    |
                    v
                MLflow (runs: clf_v1, clf_finetune_v2, summarizer baseline)
```

### 5.1 Pre-MLflow Production Hardening Roadmap

Before layering in MLflow experiment tracking, round out the service so it reads like a production resume piece:

- **Secure & Instrument the API**
  - ✅ Enforce an API key middleware on write-sensitive routes (`/classify_batch`, review queue mutations, dataset export, feedback submissions).
  - ✅ Emit structured logs with request IDs, expose per-route latency counters, and add a `/metrics/reset` maintenance hook.
- **Harden Data + Testing Workflows**
  - Introduce Alembic migrations plus seed scripts so review/feedback tables can be recreated deterministically.
  - Expand pytest coverage to include auth failures, pagination edge cases, and migration smoke tests; surface coverage % in CI.
- **Enhance Consumer Experience**
  - Deliver a Streamlit or lightweight React dashboard that hits the API (classify, summarize, review queue, dataset export charts).
  - Refresh documentation with architecture diagrams, API reference tables, auth setup, and Windows PowerShell examples.
- **Package & Deployment Upgrades**
  - Add Dockerfile + docker-compose (API + DB) and a Makefile/Invoke-Build script for common workflows.
  - Wire up GitHub Actions (or Azure Pipelines) to lint, test, build the image, and push to a registry on tagged releases.
- **Operational Readiness Extras**
  - Schedule background jobs for review queue triage (expire stale items, recompute stats) and datastore hygiene.
  - Provide webhook/Slack alert hooks when error rates spike or review backlog exceeds thresholds.

---

## 6. Data Model (Initial DB Tables)

| Table                     | Fields                                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| articles                  | id, title, raw_text, source_dataset, created_at, ingested_at                                                                       |
| predictions               | id, article_id (FK), model_version, top_category, all_probs JSON, summary_generated TEXT, summary_ref TEXT, latency_ms, created_at |
| trend_cache               | id, bucket_date, category, count                                                                                                   |
| model_runs (optional)     | id, run_id, model_version, params JSON, metrics JSON                                                                               |
| logs_inference (optional) | id, route, latency_ms, status_code, created_at                                                                                     |

---

## 7. API Endpoints

| Method | Endpoint                     | Input                  | Output                                                 |
| ------ | ---------------------------- | ---------------------- | ------------------------------------------------------ |
| GET    | /health                      | -                      | status                                                 |
| POST   | /classify_news               | { title?, text }       | categories+probs, top_category, model_version, latency |
| POST   | /summarize                   | { text, max_len? }     | summary, model_version, latency                        |
| POST   | /ingest_bulk                 | list[articles]         | stored ids                                             |
| GET    | /trends?window=7d            | window param           | category counts over window                            |
| POST   | /similar_articles (optional) | { article_id or text } | similar list (ids, similarity)                         |
| GET    | /metrics                     | -                      | backend metadata, request counters, latency stats      |
| POST   | /metrics/reset               | - (API key required)   | reset in-memory request/latency counters               |

### 7.1 Example Classification Request

```json
{
  "title": "New Satellite Launch Boosts Scientific Ambitions",
  "text": "A private aerospace firm successfully launched..."
}
```

### 7.2 Example Classification Response

```json
{
  "top_category": "SCIENCE",
  "categories": [
    { "name": "SCIENCE", "prob": 0.91 },
    { "name": "WORLD NEWS", "prob": 0.04 },
    { "name": "BUSINESS", "prob": 0.03 }
  ],
  "model_version": "clf_v1",
  "latency_ms": 42
}
```

### 7.3 Example Summarization Response

```json
{
  "summary": "A private aerospace company launched a new satellite supporting future scientific missions.",
  "model_version": "sum_v1",
  "latency_ms": 310
}
```

### 7.4 Trends Response

```json
{
  "window": "7d",
  "updated_at": "2025-01-12T10:20:00Z",
  "counts": [
    { "category": "POLITICS", "count": 58 },
    { "category": "BUSINESS", "count": 41 },
    { "category": "TECH", "count": 33 }
  ]
}
```

---

## 8. Modeling Plan

### 8.1 Classification Phases

1. Baseline (Day 1–2):
   - Preprocess text (lowercase, strip HTML).
   - TF-IDF (title + text) → Logistic Regression.
   - Evaluate macro F1 (AG News expected >0.90; HuffPost >0.70 baseline due to imbalance).
2. Upgrade (Day 3–4):
   - Fine-tune DistilBERT (sequence classification) or `roberta-base`.
   - Early stopping on validation macro F1.
3. Optimization (Day 5+):
   - Class weight tuning / focal loss variant if needed.
   - Category collapse for extremely rare labels (<200 rows).
4. Logging:
   - Store confusion matrix snapshot in MLflow artifact.

### 8.2 Summarization

- Use off-the-shelf `distilbart-cnn-12-6` initially (no fine-tune).
- Input truncation: limit raw text to ~512–768 tokens.
- Generation params: `max_new_tokens=120`, `min_new_tokens=25`, `num_beams=4`.
- Evaluate (proxy) using ROUGE-L vs `short_description` (HuffPost) — document limitation.

### 8.3 Trend Aggregation

- After predictions, append to `predictions` table.
- Daily job (cron or background task) rebuilds trend counts:
  ```sql
  INSERT INTO trend_cache (bucket_date, category, count)
  SELECT date(created_at), top_category, COUNT(*)
  FROM predictions
  WHERE created_at >= NOW() - interval '30 days'
  GROUP BY 1,2
  ON CONFLICT DO UPDATE ...
  ```

### 8.4 Optional Similarity / Clustering

- Use sentence-transformers embedding (e.g., `all-MiniLM-L6-v2`).
- Maintain vector index in memory; refresh on ingestion.
- Endpoint finds top K similar by cosine.

---

## 9. Evaluation Metrics

| Component             | Metric                           | Target / Notes                            |
| --------------------- | -------------------------------- | ----------------------------------------- |
| Classification        | Macro F1                         | >0.70 HuffPost v1; >0.80 after tuning     |
| Classification        | Weighted F1                      | For imbalance reference                   |
| Classification        | Per-Class F1                     | Identify weak classes                     |
| Summarization         | ROUGE-1/2/L                      | Informal (since ref summary is heuristic) |
| API                   | Latency p95 classify             | <120 ms (baseline logistic)               |
| API                   | Latency p95 summarization        | <1.5 s (DistilBART)                       |
| Trends                | Update freshness                 | <5 min delay or on-demand                 |
| Similarity (optional) | Average intra-cluster similarity | If clustering added                       |

---

## 10. Tooling / Stack

| Layer                 | Tool                                               |
| --------------------- | -------------------------------------------------- |
| API                   | FastAPI + Uvicorn                                  |
| Storage               | PostgreSQL                                         |
| ORM (optional)        | SQLAlchemy                                         |
| ML Experiments        | MLflow                                             |
| Models                | scikit-learn (baseline), Hugging Face Transformers |
| Embeddings (optional) | sentence-transformers                              |
| Monitoring            | Simple metrics endpoint + logs                     |
| Deployment            | Docker + docker-compose                            |
| Testing               | pytest + HTTPX                                     |

---

## 11. Directory Structure (Proposed)

```
news-intelligence/
  app/
    main.py
    api/
      routes/
        classify.py
        summarize.py
        trends.py
        metrics.py
        ingest.py
        similar.py (optional)
    core/
      config.py
      logging.py
    models/
      db_models.py
      schemas.py
    services/
      classifier.py
      summarizer.py
      trends.py
      embeddings.py
      similarity.py
      preprocessing.py
      metrics_tracker.py
    db/
      database.py
      migrations/
  data/
    raw/
      huffpost/
      ag_news/
    processed/
      train.csv
      val.csv
      test.csv
  scripts/
    prepare_huffpost.py
    train_baseline.py
    train_transformer.py
    evaluate.py
    export_models.py
    build_trends.py
  models/
    classifier/
      tfidf_vectorizer.pkl
      logreg.pkl
      label_encoder.pkl
    transformer/
      best/
    summarizer/
      distilbart/
  experiments/
    notebooks/
      01_eda.ipynb
      02_baseline_clf.ipynb
      03_transformer_finetune.ipynb
  mlflow/
  tests/
    test_classify.py
    test_summarize.py
    test_trends.py
  docker-compose.yml
  Dockerfile
  requirements.txt
  README.md
  .env.example
```

---

## 12. 7‑Day MVP Timeline (Aggressive)

| Day | Deliverable                                                                                    |
| --- | ---------------------------------------------------------------------------------------------- |
| 1   | Download AG News + HuffPost; EDA; baseline TF-IDF + logistic; macro F1 reported                |
| 2   | Switch to HuffPost full; handle rare categories; implement FastAPI /classify_news (baseline)   |
| 3   | Integrate DistilBART summarizer; /summarize endpoint; latency logging                          |
| 4   | Fine-tune transformer classifier (DistilBERT); compare vs baseline; MLflow tracking            |
| 5   | Add /trends aggregation + caching; /metrics endpoint; Dockerize                                |
| 6   | Add similarity optional OR what-if threshold sweeps; implement tests                           |
| 7   | README polish: architecture diagram, metrics snapshot, example requests; record short demo GIF |

If time slips: push fine-tune to Day 5–6 and move trends earlier.

---

## 13. Extended Features (Week 2+)

- Bias Detection: Aggregate topic distribution; highlight over/under representation vs baseline.
- Article Clustering: Group daily articles → cluster label naming via top TF-IDF terms.
- Active Learning: Flag low-confidence predictions (entropy > threshold) for manual review.
- Multi-Label (if dataset adapted): Allow multiple topics (some HuffPost items might logically map to 2 categories—requires relabel or heuristic expansion).
- Summarization Compression Ratio Metric: length(summary)/length(input).
- Rate Limiting & Auth for API hardening.

---

## 14. Example FastAPI Classification Service (Pseudo-Flow)

```python
text_clean = preprocess(text)
vector = tfidf.transform([text_clean])
probs = clf.predict_proba(vector)[0]
top_idx = probs.argmax()
response = {
  "top_category": label_encoder.inverse_transform([top_idx])[0],
  "categories": sorted(
      [{"name": c, "prob": float(p)} for c,p in zip(label_encoder.classes_, probs)],
      key=lambda x: x["prob"],
      reverse=True
  )[:5]
}
```

Later swapped with transformer pipeline:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Tokenize, infer logits → softmax → probabilities
```

---

## 15. Summarization Strategy

Input assembly:

- If article text length < threshold → prepend title for richer context.
- Truncate tokens > 512 (DistilBART).
- Use deterministic decoding for reproducibility (beam search).
  Config example:

```python
summary_ids = model.generate(
   input_ids,
   num_beams=4,
   max_new_tokens=120,
   min_new_tokens=30,
   length_penalty=2.0,
   no_repeat_ngram_size=3
)
```

---

## 16. MLflow Experiment Plan

| Run Name                     | Variant                  | Purpose                          |
| ---------------------------- | ------------------------ | -------------------------------- |
| baseline_tfidf_logreg        | TF-IDF + Logistic        | Establish macro F1 baseline      |
| tfidf_class_weights          | Weighted logistic        | Handle class imbalance           |
| distilbert_finetune_v1       | Transformer fine-tune    | + expected boost ~5–10 F1 points |
| distilbert_finetune_lr_tuned | Adjust LR                | Compare learning rates           |
| summarizer_baseline          | DistilBART zero-shot     | Capture average ROUGE            |
| summarizer_len_tuned         | Adjust generation length | Summarization quality vs brevity |

Track params: `model_type`, `lr`, `epochs`, `max_len`, `beam_width`.  
Track metrics: `macro_f1`, `weighted_f1`, `per_class_f1(json)`, `rougeL`, `latency_classify_ms`, `latency_summarize_ms`.

---

## 17. Metrics Endpoint (JSON Spec)

```json
{
  "model_version_classify": "clf_v2",
  "model_version_summarize": "sum_v1",
  "macro_f1": 0.78,
  "weighted_f1": 0.86,
  "rougeL": 0.31,
  "requests_count": 1542,
  "avg_latency_classify_ms": 38.4,
  "avg_latency_summarize_ms": 912.7,
  "topic_distribution_last_7d": {
    "BUSINESS": 120,
    "POLITICS": 98,
    "TECH": 77
  },
  "updated_at": "2025-01-13T12:04:22Z"
}
```

---

## 18. Risks & Mitigations

| Risk                                  | Impact                | Mitigation                                                |
| ------------------------------------- | --------------------- | --------------------------------------------------------- |
| Class imbalance (long-tail HuffPost)  | Low macro F1          | Collapse rare labels or apply class weights               |
| Slow summarization                    | High latency          | Use DistilBART (fast) first; async batch; cache summaries |
| Memory usage (transformer)            | Container bloat       | Use `--device cpu` + quantization (opt)                   |
| No real summary ground-truth          | Limited ROUGE meaning | Document limitation; manual sample evaluation             |
| Overfitting on small fine-tune sample | Poor generalization   | Early stopping + validation monitoring                    |
| Trend misinterpretation               | Misleading dashboards | Smooth counts (moving average), note sampling size        |

---

## 19. Example README Headings (Later)

1. Overview
2. Dataset Sources & Licensing
3. Architecture Diagram
4. Model Lifecycle (Train → Serve → Monitor)
5. Endpoints & Schemas
6. Experiments (MLflow Table)
7. Metrics Snapshot
8. Performance & Scaling Notes
9. Limitations (Topic granularity, summarization noise)
10. Roadmap

---

## 20. Immediate “Day 1” Action Plan

1. Create repo: `news-topic-intelligence`.
2. Download AG News via Hugging Face; run baseline TF-IDF logistic classification (store macro F1).
3. Scaffold FastAPI project & /classify_news using baseline model.
4. Commit: “Baseline classification working (AG News)”.
5. Begin loading HuffPost dataset; map categories; evaluate baseline cross-domain.

Momentum beats perfection—ship baseline early.

---

## 21. Upgrade Path (After MVP)

| Goal                | Description                                       |
| ------------------- | ------------------------------------------------- |
| Add Caching         | Cache summaries by content hash                   |
| Embedding Index     | For similar article retrieval                     |
| Multi-Label Support | Allow dual-topic output for ambiguous articles    |
| SHAP Explanations   | Add model interpretability with SHAP values       |
| Semantic Trends     | Track emerging topics via clustering new articles |
| Bias Audit          | Category frequency vs expected distribution       |

---

## 22. Licensing & Attribution

- HuffPost News Category Dataset: cite original source (Rishabh Misra).
- AG News: cite Kaggle or original academic reference.
- Pretrained Models: follow Hugging Face model card licenses (e.g., MIT/Apache).
  Include a LICENSE section in README documenting dataset usage for educational purposes.

---

## 23. Final Narrative (Use on CV / LinkedIn)

“Implemented a News Intelligence microservice: transformer-based topic classification, abstractive summarization (DistilBART), and real-time topic trend aggregation. Exposed production-style FastAPI endpoints with latency monitoring, experiment tracking (MLflow), and extensible architecture for future clustering and bias analysis.”

---

## 24. Optional Extensions (If You Have Extra Week)

- Add Q&A endpoint (/answer_question) using a retrieval + generative model for article queries.
- Build a minimal React dashboard: topic distribution chart, latency stats, sample classification results.
- Introduce streaming ingestion simulation (Kafka placeholder replaced by simple Python producer script).

---

## 25. Quick Tech Requirement Summary

| Area      | Choice                                                       |
| --------- | ------------------------------------------------------------ |
| Python    | 3.11                                                         |
| Core Libs | scikit-learn, transformers, sentence-transformers (optional) |
| Serving   | FastAPI, Uvicorn                                             |
| Tracking  | MLflow                                                       |
| Storage   | PostgreSQL                                                   |
| Packaging | Docker                                                       |
| Testing   | pytest, HTTPX                                                |

---

## 26. Minimal Requirements File (Draft)

```
fastapi
uvicorn[standard]
pydantic
scikit-learn
transformers
sentence-transformers
pandas
numpy
mlflow
SQLAlchemy
psycopg[binary]
python-dotenv
orjson
```

(Adjust versions + pin later.)

---

## 27. Success Criteria (MVP Definition)

- /classify_news returns valid JSON with top_category + probability distribution.
- Macro F1 (HuffPost) ≥ 0.70 (baseline or transformer).
- /summarize returns < 150-token summary under 1.5s p95 latency (CPU).
- /trends returns aggregated counts for at least last 7 days of ingested items.
- README documents dataset construction + limitations.

Hit these → project is “portfolio-grade”.

---

You’re set. Start with AG News baseline today, then scale to HuffPost for depth. Ask if you want the actual FastAPI scaffold next.

---

## 28. Completion Snapshot — October 2025

- ✅ **v1.0-v5.0 Roadmap delivered.** Complete ML Ops pipeline from baseline models to A/B testing infrastructure.
- ✅ **v4.0 ML Ops completed.** BentoML serving, Evidently drift detection, GitHub Actions CI/CD, and automated retraining pipeline fully implemented.
- ✅ **v5.0 A/B Testing completed.** Traffic splitting infrastructure, hash-based user assignment, real-time metrics tracking, and automated winner determination fully implemented.
- ✅ **Production-ready deployment.** Kubernetes manifests, health checks, and containerized deployment with drift-triggered model updates.
- ✅ **CLI regression fixed.** Typer 0.12.3 is paired with Click 8.1.7 (pinned in `requirements.txt` and installed in the project venv) so `seed-db`, `bundle-artifacts`, and transformer training commands succeed under pytest.
- ✅ **Ops artifacts archived.** Latest transformer training run (3 epochs, batch sizes 16/32) is logged to MLflow; artifacts are ready for bundling and deployment.
- ✅ **Active learning uplift.** Review queue labels surface in the dashboard and power the new `active-finetune` CLI pathway to refresh the transformer with human feedback.
- 🔄 **Optional next moves.** Explore advanced monitoring, SHAP explanations, or multi-model serving architectures.

**Project status: Full ML Ops pipeline complete with A/B testing, production-ready with automated monitoring, retraining, and safe model rollouts.**
