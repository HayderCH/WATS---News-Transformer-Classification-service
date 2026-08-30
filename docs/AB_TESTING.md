# A/B Testing Implementation Guide

## Overview

This document describes the A/B testing infrastructure implemented for safe model rollouts and performance comparison between different classification models (ensemble vs transformer).

## Architecture

### Core Components

1. **A/B Testing Service** (`app/services/ab_testing.py`)

   - Manages experiments and traffic splitting
   - Tracks real-time metrics per variant
   - Handles winner determination logic

2. **FastAPI Routes** (`app/api/routes/ab_test.py`)

   - `/ab_test` - POST endpoint for A/B classification
   - `/ab_test/results/{experiment}` - GET experiment metrics
   - `/ab_test/complete/{experiment}` - POST to complete experiment

3. **Backend Override** (`app/services/classifier.py`)
   - Modified `classify_text()` to accept `backend` parameter
   - Enables per-request model selection

## Key Design Decisions

### Traffic Splitting

**Hash-based User Assignment**

```python
# Ensures users always get the same variant
hash_value = hash(user_id + experiment_name) % 100
variant = control_model if hash_value < threshold else treatment_model
```

**Why**: Prevents result contamination and ensures statistical validity. Random assignment would cause users to see different models on different requests, making it impossible to measure true performance differences.

### Metrics Tracking

**Real-time Performance Monitoring**

- Request counts per variant
- Average latency per variant
- Accuracy tracking (when ground truth available)
- Rolling window of recent requests (max 1000)

**Why**: Enables data-driven decisions about model performance in production.

### Winner Determination

**Automated Decision Logic**

```python
if control_accuracy > treatment_accuracy:
    winner = control_model
elif treatment_accuracy > control_accuracy:
    winner = treatment_model
else:
    # Tie-breaker: lower latency wins
    winner = control_model if control_latency <= treatment_latency else treatment_model
```

**Why**: Removes human bias from model selection and ensures consistent, metric-driven decisions.

## API Usage

### Basic A/B Classification

```bash
curl -X POST "http://localhost:8001/ab_test" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple announces new iPhone with advanced AI features",
    "user_id": "user123",
    "experiment_name": "ensemble_vs_transformer"
  }'
```

**Response**:

```json
{
  "experiment_name": "ensemble_vs_transformer",
  "assigned_variant": "ensemble",
  "prediction": {
    "top_category": "TECH",
    "categories": [...],
    "confidence_score": 0.89,
    "latency_ms": 245.0
  },
  "latency_ms": 250.0,
  "experiment_results": {
    "control_requests": 45,
    "treatment_requests": 42,
    "control_accuracy": 0.0,
    "treatment_accuracy": 0.0,
    "control_latency": 180.5,
    "treatment_latency": 220.3
  }
}
```

### View Experiment Results

```bash
curl "http://localhost:8001/ab_test/results/ensemble_vs_transformer" \
  -H "X-API-Key: your-api-key"
```

### Complete Experiment

```bash
curl -X POST "http://localhost:8001/ab_test/complete/ensemble_vs_transformer" \
  -H "X-API-Key: your-api-key"
```

## Experiment Configuration

### Default Experiment

The service automatically creates a default experiment comparing ensemble vs transformer models:

```python
experiment = ExperimentConfig(
    name="ensemble_vs_transformer",
    control_model="ensemble",      # Current production model
    treatment_model="transformer", # New model to test
    traffic_split=0.5,            # 50/50 split
    status=ExperimentStatus.ACTIVE
)
```

### Custom Experiments

You can create custom experiments by modifying the service initialization in `app/services/ab_testing.py`.

## Testing

### Unit Tests

Run the A/B testing tests:

```bash
python test_ab_testing.py
```

### Integration Testing

The test script demonstrates:

- Variant assignment consistency
- Backend override functionality
- API endpoint integration
- Metrics tracking

## Production Considerations

### Safety Features

1. **Fallback Behavior**: If experiment doesn't exist, defaults to ensemble model
2. **Error Handling**: Failed requests are still tracked for error rate monitoring
3. **Resource Limits**: Request history capped at 1000 entries to prevent memory issues

### Monitoring

Track these metrics in production:

- Traffic distribution per variant
- Latency differences between models
- Error rates per variant
- Experiment completion status

### Scaling

The current implementation is suitable for moderate traffic. For high-traffic scenarios, consider:

- External metrics storage (Redis/InfluxDB)
- Distributed experiment coordination
- Statistical significance testing

## Troubleshooting

### Common Issues

1. **Inconsistent Variant Assignment**

   - Check that `user_id` is consistent across requests
   - Verify experiment configuration

2. **Backend Override Not Working**

   - Ensure `classify_text()` is called with `backend` parameter
   - Check that requested backend model is available

3. **Metrics Not Updating**
   - Verify API key authentication
   - Check service initialization

### Debug Mode

Enable debug logging to see detailed A/B testing decisions:

```python
import logging
logging.getLogger('app.services.ab_testing').setLevel(logging.DEBUG)
```

## Future Enhancements

- Statistical significance testing
- Multi-armed bandit algorithms
- Automated experiment scheduling
- Integration with feature flags
- Advanced metrics (p-values, confidence intervals)</content>
  <parameter name="filePath">c:\Users\GIGABYTE\projects\News_Topic_Classification\docs\AB_TESTING.md
