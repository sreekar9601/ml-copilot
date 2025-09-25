# Production Considerations

This document outlines important security, performance, and operational considerations for deploying the ML Documentation Copilot in production.

## 🔒 Security

### CORS Configuration
- **Current**: Environment-variable controlled CORS origins
- **Default**: `["*"]` (development only)
- **Production**: Set `ALLOWED_ORIGINS` to specific domains
- **Example**: `ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com`

### Trusted Hosts
- **Protection**: Prevents HTTP Host header attacks
- **Configuration**: Set `ALLOWED_HOSTS` environment variable
- **Example**: `ALLOWED_HOSTS=yourdomain.com,*.yourdomain.com,your-app.railway.app`

### Rate Limiting
- **Implementation**: In-memory rate limiting per IP
- **Default**: 60 requests per 60 seconds
- **Configuration**: 
  - `RATE_LIMIT_REQUESTS=60` (requests allowed)
  - `RATE_LIMIT_WINDOW=60` (time window in seconds)
- **Limitations**: Single-instance only (resets on restart)
- **Enterprise**: Consider Redis-backed rate limiting for multi-instance deployments

### API Key Protection
- **Google Gemini API**: Store in Railway environment variables (encrypted)
- **Qdrant API**: Store in Railway environment variables (encrypted)
- **Never**: Commit API keys to git repositories
- **Rotation**: Regularly rotate API keys

### Headers Security
- **CORS Headers**: Restricted to essential headers only
- **Methods**: Limited to GET and POST
- **Credentials**: Enabled for authenticated requests

## 🚀 Performance

### Caching Strategies
- **Current**: No caching implemented
- **Recommendations**:
  - Redis for query result caching
  - CDN for static assets
  - Database query optimization

### Database Performance
- **Qdrant Cloud**: Automatically managed and scaled
- **SQLite**: File-based, suitable for read-heavy workloads
- **Monitoring**: Track query response times

### Vector Search Optimization
- **Qdrant**: Optimized for similarity search
- **Batch Size**: Configure based on memory availability
- **Index Settings**: Use Qdrant's HNSW algorithm (default)

### API Response Times
- **Target**: < 2 seconds for /ask endpoint
- **Monitoring**: Log response times for optimization
- **Chunking**: Optimize chunk size vs. relevance trade-off

## 📊 Monitoring & Observability

### Health Checks
- **Railway**: `/health` endpoint with 60s start period
- **Docker**: Built-in healthcheck every 30s
- **Status**: Basic availability check

### Logging
- **Structured Logging**: JSON format for production
- **Log Levels**: INFO for normal operations, DEBUG for troubleshooting
- **Sensitive Data**: Never log API keys or user queries

### Metrics to Track
- **Request Volume**: Requests per minute/hour
- **Response Times**: p50, p95, p99 latencies
- **Error Rates**: 4xx, 5xx response codes
- **Database Performance**: Query times, connection counts
- **Rate Limiting**: Blocked requests, rate limit hits

### Error Handling
- **Graceful Degradation**: Return helpful error messages
- **Fallbacks**: ChromaDB if Qdrant unavailable
- **Retries**: Implement exponential backoff for external APIs

## 🔄 Deployment & Operations

### Environment Management
- **Development**: Local `.env` file
- **Production**: Railway environment variables
- **Staging**: Separate Railway project with staging configs

### Database Management
- **Qdrant Cloud**: Managed service with automatic backups
- **SQLite**: Included in container, backed up with Railway volumes
- **Data Migration**: Version control for schema changes

### Scaling Considerations
- **Horizontal Scaling**: Multiple Railway instances
- **Load Balancing**: Railway handles automatically
- **Session Storage**: Stateless API design
- **Database Connections**: Pool connections efficiently

### Backup & Recovery
- **Qdrant**: Automatic cloud backups
- **SQLite**: Railway volume snapshots
- **Configuration**: Environment variable backups
- **Recovery Time**: < 1 hour for full restoration

## 🛡️ Security Best Practices

### Input Validation
- **Query Length**: Limit to reasonable size (1000 chars)
- **Sanitization**: FastAPI automatic validation
- **Injection Protection**: Parameterized queries

### Authentication & Authorization
- **Current**: No authentication implemented
- **Enterprise Options**:
  - API key authentication
  - JWT tokens
  - OAuth 2.0 integration
  - Role-based access control (RBAC)

### Data Privacy
- **User Queries**: Not stored permanently
- **Logs**: Exclude sensitive information
- **GDPR**: Consider data retention policies
- **Anonymization**: Hash IP addresses in logs

## 📈 Cost Optimization

### Google Gemini API
- **Pricing**: Pay-per-token
- **Optimization**: Limit context size, cache responses
- **Monitoring**: Track token usage and costs

### Qdrant Cloud
- **Free Tier**: 1GB storage (330k vectors)
- **Paid Plans**: $10+/month for larger datasets
- **Optimization**: Monitor storage usage

### Railway
- **Pricing**: Pay-per-use compute
- **Optimization**: 
  - Right-size memory allocation
  - Scale down during low traffic
  - Use sleep mode for dev environments

## 🔧 Configuration Examples

### Development Environment
```env
GOOGLE_API_KEY=your_dev_api_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
ALLOWED_HOSTS=localhost,127.0.0.1
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Production Environment
```env
GOOGLE_API_KEY=your_prod_api_key
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_key
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
ALLOWED_HOSTS=yourdomain.com,*.yourdomain.com,your-app.railway.app
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

## 🚨 Incident Response

### Common Issues
1. **API Key Expiration**: Rotate keys immediately
2. **Rate Limit Exceeded**: Adjust limits or scale horizontally
3. **Database Connectivity**: Check Qdrant Cloud status
4. **High Latency**: Investigate database query performance

### Alerting
- **Response Times** > 5 seconds
- **Error Rates** > 5%
- **Rate Limit** > 80% utilization
- **Database** connection failures

### Runbooks
- **Service Restart**: Railway dashboard or CLI
- **Environment Update**: Railway variables update
- **Database Reset**: Qdrant collection recreation
- **Rollback**: Git revert + Railway redeploy

## 📋 Production Checklist

### Pre-Deployment
- [ ] API keys configured and tested
- [ ] CORS origins restricted to production domains
- [ ] Rate limiting configured appropriately
- [ ] Database connections tested
- [ ] Health checks passing
- [ ] Security headers configured

### Post-Deployment
- [ ] Monitoring dashboards configured
- [ ] Alerting rules set up
- [ ] Performance baselines established
- [ ] Backup procedures tested
- [ ] Incident response plan documented
- [ ] Team access and permissions configured

### Ongoing Maintenance
- [ ] Regular API key rotation
- [ ] Monthly cost review
- [ ] Quarterly security audit
- [ ] Performance optimization review
- [ ] Database maintenance tasks
- [ ] Documentation updates

---

📧 **Need Help?** Contact your DevOps team or refer to [Railway Documentation](https://docs.railway.app) and [Qdrant Documentation](https://qdrant.tech/documentation/).
