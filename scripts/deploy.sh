#!/bin/bash
# Deployment script for News Topic Classification service

set -e

echo "🚀 Starting deployment of News Topic Classification service"

# Configuration
SERVICE_NAME="news-classifier"
DOCKER_IMAGE="news-classifier:latest"
NAMESPACE="${NAMESPACE:-default}"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t $DOCKER_IMAGE .

# Save image for deployment
echo "💾 Saving Docker image..."
docker save $DOCKER_IMAGE > ${SERVICE_NAME}.tar

# Deploy to Kubernetes (if available)
if command -v kubectl &> /dev/null; then
    echo "☸️  Deploying to Kubernetes..."

    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Apply Kubernetes manifests
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $SERVICE_NAME
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $SERVICE_NAME
  template:
    metadata:
      labels:
        app: $SERVICE_NAME
    spec:
      containers:
      - name: $SERVICE_NAME
        image: $DOCKER_IMAGE
        ports:
        - containerPort: 3000
        env:
        - name: MODEL_DIR
          value: "/app/models"
        - name: TRANSFORMER_MODEL_DIR
          value: "/app/models/transformer_huffpost"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: $SERVICE_NAME
  namespace: $NAMESPACE
spec:
  selector:
    app: $SERVICE_NAME
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP
EOF

    echo "⏳ Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$SERVICE_NAME -n $NAMESPACE

    echo "✅ Deployment completed successfully!"
    echo "🌐 Service available at: http://$SERVICE_NAME.$NAMESPACE.svc.cluster.local"

else
    echo "🐳 Running locally with Docker..."
    docker run -d \
        --name $SERVICE_NAME \
        -p 3000:3000 \
        -e MODEL_DIR=/app/models \
        -e TRANSFORMER_MODEL_DIR=/app/models/transformer_huffpost \
        $DOCKER_IMAGE

    echo "✅ Service running locally at: http://localhost:3000"
fi

echo "🎉 Deployment complete!"