# Performance Monitoring and Interactive Search

This document describes the performance monitoring and interactive search capabilities added to the Pixel Cortex IT Support Agent system.

## Overview

The system now includes:

1. **Performance Monitoring Service** - Real-time monitoring and optimization of LLM and audio processing performance
2. **Interactive Search Service** - LLM-enhanced search across conversations, tickets, policies, and knowledge graph
3. **REST API Endpoints** - Full API access to monitoring and search functionality

## Performance Monitoring

### Features

- **Real-time Metrics Collection**: CPU, memory, LLM response times, cache hit rates, audio processing throughput
- **Automated Alerting**: Configurable thresholds with warning and critical alerts
- **Auto-optimization**: Automatic performance optimizations based on system metrics
- **Health Scoring**: Overall system health score (0-100) with trend analysis
- **Benchmarking**: Comprehensive performance benchmarking of all services
- **Data Export**: Export performance data for external analysis

### Key Metrics Tracked

| Metric | Description | Warning Threshold | Critical Threshold |
|--------|-------------|-------------------|-------------------|
| Response Time | LLM inference time (ms) | 5,000ms | 10,000ms |
| Memory Usage | System memory usage (%) | 80% | 90% |
| CPU Usage | System CPU usage (%) | 85% | 95% |
| Cache Hit Rate | LLM cache hit rate (%) | <30% | <10% |
| Error Rate | Processing error rate (%) | 5% | 15% |
| Throughput | Audio processing rate (req/sec) | <0.5 | <0.2 |

### Auto-optimization Features

1. **Memory Management**: Automatically unload unused models when memory usage is high
2. **Response Time Optimization**: Reduce token limits for faster inference
3. **Cache Optimization**: Clear old cache entries to improve hit rates
4. **Model Preloading**: Preload models based on usage patterns

### API Endpoints

```bash
# Get real-time metrics
GET /api/performance/metrics

# Get comprehensive summary
GET /api/performance/summary

# Get resource usage trends
GET /api/performance/trends?hours_back=24

# Get optimization recommendations
GET /api/performance/recommendations

# Apply optimization
POST /api/performance/optimize/{optimization_type}

# Start/stop monitoring
POST /api/performance/monitoring/start?interval_seconds=30
POST /api/performance/monitoring/stop

# Run benchmark
POST /api/performance/benchmark

# Get/clear alerts
GET /api/performance/alerts?severity=critical
DELETE /api/performance/alerts?severity=warning

# Export performance data
GET /api/performance/export?hours_back=24

# Get system health
GET /api/performance/health
```

## Interactive Search

### Features

- **LLM-Enhanced Query Processing**: Automatic query enhancement and intent detection
- **Multi-source Search**: Search across conversations, tickets, policies, and knowledge graph
- **Semantic Similarity**: Advanced relevance scoring using LLM
- **Smart Highlighting**: Extract relevant highlights from search results
- **Search Analytics**: User search patterns and behavior analysis
- **Search Suggestions**: Intelligent search suggestions based on context

### Search Types

1. **Conversation Search**: Semantic search through conversation history
2. **Ticket Search**: Enhanced ticket history search with filtering
3. **Policy Search**: Policy document search with compliance analysis
4. **Knowledge Graph Search**: Concept-based graph traversal
5. **Unified Search**: Cross-source search with unified ranking

### Search Enhancement

The system automatically enhances user queries by:

- **Intent Detection**: Classifying search intent (troubleshooting, policy lookup, etc.)
- **Keyword Extraction**: Identifying key terms and concepts
- **Query Expansion**: Suggesting related search terms
- **Relevance Scoring**: LLM-powered relevance calculation

### API Endpoints

```bash
# Intelligent multi-source search
POST /api/search/search
{
  "query": "VPN connection issues",
  "search_types": ["unified"],
  "filters": {"start_date": "2024-01-01"},
  "limit": 10
}

# Source-specific searches
GET /api/search/conversations?query=password&limit=10
GET /api/search/tickets?query=email&status=open
GET /api/search/policies?query=security&category=IT
GET /api/search/knowledge-graph?query=networking

# Search assistance
GET /api/search/suggestions?query=VPN&limit=5
GET /api/search/search-types

# User search management
GET /api/search/history?limit=20
GET /api/search/analytics?days_back=30
GET /api/search/stats
POST /api/search/save-search

# Quick access and trends
GET /api/search/quick-access
GET /api/search/trending?days_back=7
```

## Integration with Existing System

### LLM Service Integration

The performance monitor tracks:
- Model loading status
- Inference response times
- Cache hit rates
- Memory usage impact

The search service uses:
- Response generation for query enhancement
- Relevance scoring for results
- Concept extraction for knowledge graph search

### Audio Service Integration

Performance monitoring includes:
- Audio processing times
- Transcription accuracy
- Error rates
- Throughput metrics

### Knowledge Graph Integration

Search service leverages:
- Concept extraction for graph traversal
- Relationship analysis for relevance
- Graph-based result ranking

## Configuration

### Performance Thresholds

Configure monitoring thresholds via API:

```bash
POST /api/performance/thresholds
{
  "response_time": {"warning": 4000, "critical": 8000},
  "memory_usage": {"warning": 75, "critical": 85}
}
```

### Search Behavior

Search behavior is automatically optimized based on:
- User search patterns
- Result relevance feedback
- System performance constraints

## Monitoring and Maintenance

### Starting Services

Services start automatically with the application:

```python
# In main.py startup event
performance_monitor.start_monitoring(interval_seconds=30)
```

### Health Checks

Monitor system health:

```bash
# Overall system health
GET /api/performance/health

# Detailed metrics
GET /api/performance/metrics

# Service status
GET /api/performance/status
```

### Performance Optimization

1. **Manual Optimization**: Apply specific optimizations via API
2. **Auto-optimization**: Enable/disable automatic optimizations
3. **Threshold Tuning**: Adjust alert thresholds based on system characteristics

### Search Performance

Monitor search performance through:
- Search analytics endpoints
- User search pattern analysis
- Query enhancement effectiveness

## Best Practices

### Performance Monitoring

1. **Set Appropriate Thresholds**: Adjust thresholds based on your hardware
2. **Enable Auto-optimization**: Let the system automatically optimize performance
3. **Regular Benchmarking**: Run benchmarks to track performance trends
4. **Monitor Trends**: Use trend analysis to predict performance issues

### Search Optimization

1. **Use Unified Search**: Default to unified search for best results
2. **Leverage Suggestions**: Use search suggestions to guide users
3. **Monitor Search Patterns**: Analyze user search patterns for insights
4. **Optimize Queries**: Use query enhancement for better results

## Troubleshooting

### Performance Issues

1. **High Memory Usage**:
   - Check if models are loaded unnecessarily
   - Apply memory optimization
   - Reduce model size or enable quantization

2. **Slow Response Times**:
   - Reduce max_tokens for faster inference
   - Enable caching
   - Consider model optimization

3. **Low Cache Hit Rate**:
   - Review cache strategy
   - Increase cache TTL
   - Improve cache key design

### Search Issues

1. **Poor Search Results**:
   - Check LLM service status
   - Review query enhancement
   - Adjust relevance thresholds

2. **Search Service Errors**:
   - Verify database connections
   - Check service dependencies
   - Review error logs

### Monitoring Disabled Features

If LLM or audio services are unavailable:
- Performance metrics fall back to system-only monitoring
- Search falls back to keyword-based matching
- System continues to operate with reduced functionality

## Development and Testing

### Running Tests

```bash
# Run new service tests
pytest tests/test_integration_new_services.py -v

# Run all tests
pytest tests/ -v
```

### Adding New Metrics

1. Add metric to `PerformanceMetric` enum
2. Implement collection in `collect_current_metrics()`
3. Add thresholds in monitor initialization
4. Update health score calculation

### Extending Search

1. Add new search type to `SearchType` class
2. Implement search method in `InteractiveSearchService`
3. Add API endpoint in search router
4. Update documentation

## Security Considerations

### Performance Monitoring

- Monitor performance data access
- Secure performance optimization endpoints
- Validate threshold updates
- Protect sensitive metrics

### Search Service

- Ensure user data isolation
- Validate search permissions
- Sanitize search queries
- Protect conversation privacy

## Future Enhancements

### Performance Monitoring

- [ ] Distributed monitoring across multiple instances
- [ ] Machine learning-based anomaly detection
- [ ] Advanced optimization algorithms
- [ ] Performance prediction models

### Interactive Search

- [ ] Cross-user search analytics
- [ ] Advanced NLP for query understanding
- [ ] Federated search across external sources
- [ ] Real-time search suggestions

## Conclusion

The performance monitoring and interactive search features provide:

1. **Proactive Performance Management**: Continuous monitoring with automatic optimization
2. **Intelligent Search Capabilities**: LLM-enhanced search across all data sources
3. **User Experience Enhancement**: Better search results and system responsiveness
4. **Operational Insights**: Comprehensive analytics and trend analysis

These features ensure the Pixel Cortex system operates efficiently while providing powerful search capabilities for IT support workflows.
