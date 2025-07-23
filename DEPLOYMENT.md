# Cloud Deployment Guide

This guide explains how to deploy the forecasting pipeline to various cloud platforms.

## 🚀 Quick Start

### 1. Environment Setup

1. **Copy configuration:**
   ```bash
   cp config.example.ini config.ini
   ```

2. **Update configuration with your values:**
   - Database credentials
   - Email settings
   - Cloud storage credentials
   - MLflow/Prefect server URLs

### 2. Cloud Platform Options

#### Option A: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Access services:**
   - Dashboard: http://localhost:8501
   - MLflow: http://localhost:5000
   - Prefect: http://localhost:4200

#### Option B: Kubernetes Deployment

1. **Create secrets:**
   ```bash
   kubectl create secret generic forecasting-secrets \
     --from-literal=mlflow-tracking-uri=http://mlflow-server:5000 \
     --from-literal=prefect-api-url=http://prefect-server:4200/api \
     --from-literal=database-url=postgresql://user:pass@db:5432/forecast
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

#### Option C: Cloud Platforms

##### AWS ECS/Fargate
```bash
# Build and push Docker image
docker build -t your-registry/forecasting-pipeline .
docker push your-registry/forecasting-pipeline

# Deploy using AWS CLI or console
```

##### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy forecasting-pipeline \
  --image your-registry/forecasting-pipeline \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

##### Azure Container Instances
```bash
# Deploy to Azure
az container create \
  --resource-group your-rg \
  --name forecasting-pipeline \
  --image your-registry/forecasting-pipeline \
  --ports 8501 5000 4200
```

## 🔧 Configuration

### Environment Variables

Set these environment variables in your cloud deployment:

```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/database

# MLflow
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# Prefect
PREFECT_API_URL=http://your-prefect-server:4200/api

# Email (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_FROM=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_TO=recipient@example.com

# Cloud Storage (AWS S3 example)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-forecasting-bucket
```

### Cloud Storage Setup

1. **AWS S3:**
   ```bash
   aws s3 mb s3://your-forecasting-bucket
   aws s3 sync data/ s3://your-forecasting-bucket/data/
   ```

2. **Google Cloud Storage:**
   ```bash
   gsutil mb gs://your-forecasting-bucket
   gsutil -m cp -r data/ gs://your-forecasting-bucket/
   ```

3. **Azure Blob Storage:**
   ```bash
   az storage container create --name forecasting-data
   az storage blob upload-batch --source data/ --destination forecasting-data
   ```

## 📊 Monitoring

### Health Checks

The application includes health checks at:
- Dashboard: `/_stcore/health`
- MLflow: `/health`
- Prefect: `/api/health`

### Logging

Logs are available at:
- Application logs: `/app/logs/`
- Container logs: `docker logs <container-name>`
- Kubernetes logs: `kubectl logs <pod-name>`

### Alerts

Configure alerts for:
- Pipeline failures
- Model performance degradation
- Data quality issues
- System resource usage

## 🔒 Security

### Secrets Management

1. **Use cloud-native secrets:**
   - AWS Secrets Manager
   - Google Secret Manager
   - Azure Key Vault

2. **Never commit secrets to git:**
   - All `.env` files are in `.gitignore`
   - Use environment variables in production

3. **Network security:**
   - Use VPC/private subnets
   - Configure security groups/firewall rules
   - Enable SSL/TLS encryption

### Access Control

1. **IAM/RBAC:**
   - Least privilege access
   - Service accounts for applications
   - Regular access reviews

2. **Data protection:**
   - Encrypt data at rest and in transit
   - Regular backups
   - Data retention policies

## 📈 Scaling

### Auto-scaling

Configure auto-scaling based on:
- CPU usage
- Memory usage
- Request volume
- Custom metrics

### Load Balancing

Use load balancers for:
- High availability
- SSL termination
- Health checks
- Traffic distribution

## 🚨 Troubleshooting

### Common Issues

1. **Database connection errors:**
   - Check connection strings
   - Verify network connectivity
   - Ensure database is running

2. **Model training failures:**
   - Check resource limits
   - Verify data quality
   - Review error logs

3. **Notification failures:**
   - Check email/Slack credentials
   - Verify network access
   - Review notification logs

### Debug Commands

```bash
# Check application status
curl http://localhost:8501/_stcore/health

# View logs
docker logs <container-name>
kubectl logs <pod-name>

# Check resource usage
docker stats
kubectl top pods
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Prefect Documentation](https://docs.prefect.io/)
- [Streamlit Documentation](https://docs.streamlit.io/) 