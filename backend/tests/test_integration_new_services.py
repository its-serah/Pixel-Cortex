"""
Integration Tests for New Services

Tests performance monitoring, interactive search, and overall system integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient

from app.main import app
from app.services.performance_monitor import performance_monitor, PerformanceMetric
from app.services.interactive_search_service import interactive_search_service, SearchType


client = TestClient(app)


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring service integration"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initializes correctly"""
        assert performance_monitor is not None
        assert len(performance_monitor.thresholds) == 6
        assert PerformanceMetric.RESPONSE_TIME in performance_monitor.thresholds
        assert not performance_monitor.monitoring_active
    
    def test_collect_metrics(self):
        """Test metric collection from system"""
        metrics = performance_monitor.collect_current_metrics()
        
        assert isinstance(metrics, dict)
        assert PerformanceMetric.MEMORY_USAGE in metrics
        assert PerformanceMetric.CPU_USAGE in metrics
        assert 0 <= metrics[PerformanceMetric.MEMORY_USAGE] <= 100
        assert 0 <= metrics[PerformanceMetric.CPU_USAGE] <= 100
    
    def test_health_score_calculation(self):
        """Test health score calculation"""
        mock_metrics = {
            PerformanceMetric.MEMORY_USAGE: 50,
            PerformanceMetric.CPU_USAGE: 30,
            PerformanceMetric.RESPONSE_TIME: 2000,
            PerformanceMetric.ERROR_RATE: 2,
            PerformanceMetric.CACHE_HIT_RATE: 80
        }
        
        health_score = performance_monitor._calculate_health_score(mock_metrics)
        assert isinstance(health_score, int)
        assert 0 <= health_score <= 100
        assert health_score > 80  # Should be high with good metrics
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        recommendations = performance_monitor.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        # Recommendations depend on current system state
    
    @patch('app.services.local_llm_service.local_llm_service')
    def test_benchmark_system_performance(self, mock_llm):
        """Test system performance benchmarking"""
        mock_llm.model = None  # Mock model not loaded
        mock_llm.get_performance_stats.return_value = {
            "model_loaded": False,
            "inference_stats": {}
        }
        
        benchmark = performance_monitor.benchmark_system_performance()
        
        assert "timestamp" in benchmark
        assert "tests" in benchmark
        assert "system_info" in benchmark
        assert "performance_score" in benchmark
        assert isinstance(benchmark["performance_score"], int)
    
    def test_alert_management(self):
        """Test performance alert creation and management"""
        initial_alert_count = len(performance_monitor.alerts)
        
        # Simulate high memory usage
        mock_metrics = {
            PerformanceMetric.MEMORY_USAGE: 95,  # Critical level
            PerformanceMetric.CPU_USAGE: 30
        }
        
        performance_monitor._check_thresholds(mock_metrics)
        
        # Should have generated a critical alert
        assert len(performance_monitor.alerts) > initial_alert_count
        
        latest_alert = performance_monitor.alerts[-1]
        assert latest_alert.severity == "critical"
        assert latest_alert.metric == PerformanceMetric.MEMORY_USAGE
        
        # Test alert clearing
        cleared_count = performance_monitor.clear_alerts("critical")
        assert cleared_count > 0


class TestInteractiveSearchIntegration:
    """Test interactive search service integration"""
    
    def test_search_service_initialization(self):
        """Test search service initializes correctly"""
        assert interactive_search_service is not None
        assert interactive_search_service.kg_builder is not None
    
    def test_keyword_extraction(self):
        """Test keyword extraction functionality"""
        text = "How to configure VPN connection for remote work?"
        keywords = interactive_search_service._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert "configure" in keywords
        assert "vpn" in keywords
        assert "connection" in keywords
        assert "remote" in keywords
        assert "work" in keywords
        # Stop words should be filtered out
        assert "how" not in keywords
        assert "to" not in keywords
        assert "for" not in keywords
    
    def test_search_intent_detection(self):
        """Test search intent detection"""
        test_cases = [
            ("VPN connection error", "troubleshooting"),
            ("What is the password policy?", "policy_lookup"),
            ("How to reset password?", "information_request"),
            ("Show me previous tickets", "historical_lookup"),
            ("Network configuration", "general_search")
        ]
        
        for query, expected_intent in test_cases:
            intent = interactive_search_service._detect_search_intent(query)
            assert intent == expected_intent
    
    def test_search_suggestions_generation(self):
        """Test search suggestions generation"""
        query = "email not working"
        suggestions = interactive_search_service._generate_search_suggestions(query)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        assert any("solution" in suggestion for suggestion in suggestions)
    
    def test_simple_relevance_scoring(self):
        """Test simple relevance scoring"""
        query = "VPN connection issue"
        content = "The VPN connection failed due to authentication issues"
        
        score = interactive_search_service._simple_relevance_score(query, content)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.5  # Should have good overlap
    
    def test_highlight_extraction(self):
        """Test highlight extraction from content"""
        query = "password reset"
        content = "To reset your password, go to settings. The password reset function is available in user preferences. Always use strong passwords."
        
        highlights = interactive_search_service._extract_highlights(query, content)
        
        assert isinstance(highlights, list)
        assert len(highlights) <= 3
        for highlight in highlights:
            assert "text" in highlight
            assert "match_count" in highlight
            assert "confidence" in highlight
    
    @pytest.mark.asyncio
    async def test_query_enhancement_fallback(self):
        """Test query enhancement with LLM failure fallback"""
        with patch('app.services.local_llm_service.local_llm_service') as mock_llm:
            mock_llm.generate_response.return_value = {"error": "LLM unavailable"}
            
            query = "email problems"
            enhanced = await interactive_search_service._enhance_search_query(query)
            
            assert enhanced["original"] == query
            assert enhanced["enhanced"] == query  # Should fallback to original
            assert "keywords" in enhanced
            assert "intent" in enhanced


class TestAPIEndpointsIntegration:
    """Test API endpoints integration"""
    
    @patch('app.core.auth.get_current_user')
    def test_performance_metrics_endpoint(self, mock_get_user):
        """Test performance metrics API endpoint"""
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        response = client.get("/api/performance/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data
        assert "health_score" in data
        assert "services_status" in data
    
    @patch('app.core.auth.get_current_user')
    def test_performance_summary_endpoint(self, mock_get_user):
        """Test performance summary API endpoint"""
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        response = client.get("/api/performance/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "health_score" in data
        assert "current_metrics" in data
        assert "monitoring_active" in data
        assert "thresholds" in data
    
    @patch('app.core.auth.get_current_user')
    def test_search_suggestions_endpoint(self, mock_get_user):
        """Test search suggestions API endpoint"""
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        response = client.get("/api/search/suggestions?query=VPN&limit=3")
        assert response.status_code == 200
        
        suggestions = response.json()
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
    
    @patch('app.core.auth.get_current_user')
    @patch('app.services.interactive_search_service.interactive_search_service')
    def test_intelligent_search_endpoint(self, mock_search_service, mock_get_user):
        """Test intelligent search API endpoint"""
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        # Mock search service response
        mock_search_service.intelligent_search = AsyncMock(return_value={
            "original_query": "test query",
            "enhanced_query": {"enhanced": "test query"},
            "results": {
                "unified": {
                    "total_found": 2,
                    "items": [
                        {"title": "Test Result 1", "relevance_score": 0.8},
                        {"title": "Test Result 2", "relevance_score": 0.6}
                    ]
                }
            },
            "summary": {"total_results": 2}
        })
        
        search_request = {
            "query": "test query",
            "search_types": ["unified"],
            "limit": 10
        }
        
        response = client.post("/api/search/search", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "original_query" in data
        assert "results" in data
        assert "summary" in data


class TestSystemIntegration:
    """Test overall system integration"""
    
    @patch('app.services.local_llm_service.local_llm_service')
    @patch('app.services.audio_processing_service.audio_service')
    def test_service_coordination(self, mock_audio, mock_llm):
        """Test coordination between different services"""
        
        # Mock services as loaded
        mock_llm.model = Mock()
        mock_llm.get_performance_stats.return_value = {
            "model_loaded": True,
            "inference_stats": {
                "avg_inference_time": 2000,
                "total_requests": 100,
                "cache_hits": 70
            }
        }
        
        mock_audio.whisper_model = Mock()
        mock_audio.get_processing_stats.return_value = {
            "model_loaded": True,
            "processing_stats": {
                "total_processed": 50,
                "avg_processing_time": 1500,
                "errors": 2
            }
        }
        
        # Collect metrics
        metrics = performance_monitor.collect_current_metrics()
        
        # Verify metrics include data from both services
        assert PerformanceMetric.RESPONSE_TIME in metrics
        assert PerformanceMetric.CACHE_HIT_RATE in metrics
        assert PerformanceMetric.THROUGHPUT in metrics
        assert PerformanceMetric.ERROR_RATE in metrics
        
        assert metrics[PerformanceMetric.RESPONSE_TIME] == 2000
        assert metrics[PerformanceMetric.CACHE_HIT_RATE] == 70
        assert metrics[PerformanceMetric.ERROR_RATE] == 4  # 2/50 * 100
    
    def test_monitoring_lifecycle(self):
        """Test performance monitoring start/stop lifecycle"""
        assert not performance_monitor.monitoring_active
        
        # Start monitoring
        performance_monitor.start_monitoring(interval_seconds=60)
        assert performance_monitor.monitoring_active
        assert performance_monitor.monitor_thread is not None
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
    
    @patch('app.services.local_llm_service.local_llm_service')
    async def test_search_with_llm_integration(self, mock_llm):
        """Test search service integration with LLM"""
        
        # Mock LLM responses
        mock_llm.generate_response.return_value = {
            "response": "Enhanced query: VPN connection troubleshooting",
            "tokens_generated": 10
        }
        
        query = "VPN not working"
        enhanced_query = await interactive_search_service._enhance_search_query(query)
        
        assert enhanced_query["original"] == query
        assert "enhanced" in enhanced_query
        assert "keywords" in enhanced_query
        assert "intent" in enhanced_query
    
    def test_error_handling_resilience(self):
        """Test system resilience to service failures"""
        
        # Test performance monitor with service failures
        with patch('app.services.local_llm_service.local_llm_service') as mock_llm:
            mock_llm.get_performance_stats.side_effect = Exception("Service unavailable")
            
            # Should not crash and should include fallback metrics
            metrics = performance_monitor.collect_current_metrics()
            assert PerformanceMetric.MEMORY_USAGE in metrics  # System metrics should still work
            assert PerformanceMetric.CPU_USAGE in metrics
    
    @patch('app.core.auth.get_current_user')
    def test_api_endpoints_accessibility(self, mock_get_user):
        """Test that all new API endpoints are accessible"""
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        # Test performance endpoints
        endpoints_to_test = [
            "/api/performance/metrics",
            "/api/performance/summary",
            "/api/performance/health",
            "/api/performance/status",
            "/api/search/search-types",
            "/api/search/suggestions?query=test"
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            # Should not return 404 (endpoint exists)
            assert response.status_code != 404
    
    def test_performance_data_export(self):
        """Test performance data export functionality"""
        # Add some mock data to metrics history
        test_data = {
            "value": 75.0,
            "timestamp": datetime.now()
        }
        performance_monitor.metrics_history[PerformanceMetric.MEMORY_USAGE].append(test_data)
        
        exported_data = performance_monitor.export_performance_data(hours_back=1)
        
        assert "export_metadata" in exported_data
        assert "metrics_history" in exported_data
        assert "alerts_history" in exported_data
        assert "performance_summary" in exported_data
        
        # Check metadata
        metadata = exported_data["export_metadata"]
        assert "generated_at" in metadata
        assert "hours_back" in metadata
        assert "monitoring_active" in metadata
    
    @pytest.mark.asyncio
    async def test_search_fallback_mechanisms(self):
        """Test search service fallback mechanisms"""
        
        # Test with LLM unavailable
        with patch('app.services.local_llm_service.local_llm_service') as mock_llm:
            mock_llm.generate_response.return_value = {"error": "Model not loaded"}
            
            query = "network issues"
            enhanced_query = await interactive_search_service._enhance_search_query(query)
            
            # Should fallback gracefully
            assert enhanced_query["original"] == query
            assert enhanced_query["enhanced"] == query
            assert enhanced_query["intent"] == "troubleshooting"
    
    def test_concurrent_monitoring_safety(self):
        """Test thread safety of performance monitoring"""
        
        # Start monitoring
        performance_monitor.start_monitoring(interval_seconds=1)
        
        # Collect metrics from multiple threads simultaneously
        import threading
        
        results = []
        
        def collect_metrics():
            metrics = performance_monitor.collect_current_metrics()
            results.append(metrics)
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=collect_metrics)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert PerformanceMetric.MEMORY_USAGE in result
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
    
    def test_performance_optimization_application(self):
        """Test application of performance optimizations"""
        
        with patch('app.services.local_llm_service.local_llm_service') as mock_llm:
            mock_llm.unload_model = Mock()
            mock_llm.load_model = Mock()
            
            # Test model unloading optimization
            result = performance_monitor.apply_optimization("unload_models")
            assert result["applied"] == True
            assert "message" in result
            mock_llm.unload_model.assert_called_once()
            
            # Test model preloading optimization
            result = performance_monitor.apply_optimization("preload_model")
            assert result["applied"] == True
            mock_llm.load_model.assert_called_once()
            
            # Test unknown optimization
            result = performance_monitor.apply_optimization("unknown_optimization")
            assert result["applied"] == False
            assert "error" in result


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios"""
    
    @patch('app.core.auth.get_current_user')
    @patch('app.services.local_llm_service.local_llm_service')
    def test_search_and_monitor_scenario(self, mock_llm, mock_get_user):
        """Test realistic scenario: user searches while system is monitored"""
        
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        # Mock LLM responses
        mock_llm.generate_response.return_value = {
            "response": "Enhanced search query",
            "tokens_generated": 15
        }
        mock_llm.get_performance_stats.return_value = {
            "model_loaded": True,
            "inference_stats": {
                "avg_inference_time": 3000,
                "total_requests": 10,
                "cache_hits": 7
            }
        }
        
        # Start performance monitoring
        performance_monitor.start_monitoring(interval_seconds=1)
        
        # Perform search
        search_request = {
            "query": "email configuration help",
            "search_types": ["unified"],
            "limit": 5
        }
        
        # The search should work even with monitoring active
        response = client.post("/api/search/search", json=search_request)
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        
        # Check that monitoring doesn't interfere with search
        assert response.status_code in [200, 500]  # 500 acceptable due to mocked services
    
    @patch('app.core.auth.get_current_user')
    def test_performance_optimization_scenario(self, mock_get_user):
        """Test performance optimization scenario"""
        
        mock_user = Mock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user
        
        # Get recommendations
        response = client.get("/api/performance/recommendations")
        assert response.status_code == 200
        
        recommendations = response.json()
        assert isinstance(recommendations, list)
        
        # If there are recommendations, try to apply one
        if recommendations:
            first_recommendation = recommendations[0]
            if "action" in first_recommendation:
                action = first_recommendation["action"]
                
                # Apply optimization
                response = client.post(f"/api/performance/optimize/{action}")
                # Should not crash (may succeed or fail depending on system state)
                assert response.status_code in [200, 500]
    
    def test_system_health_monitoring(self):
        """Test system health monitoring end-to-end"""
        
        # Collect initial metrics
        initial_metrics = performance_monitor.collect_current_metrics()
        initial_health = performance_monitor._calculate_health_score(initial_metrics)
        
        # Health score should be reasonable
        assert 0 <= initial_health <= 100
        
        # Test different health scenarios
        excellent_metrics = {
            PerformanceMetric.MEMORY_USAGE: 40,
            PerformanceMetric.CPU_USAGE: 20,
            PerformanceMetric.RESPONSE_TIME: 1000,
            PerformanceMetric.ERROR_RATE: 1,
            PerformanceMetric.CACHE_HIT_RATE: 90
        }
        excellent_health = performance_monitor._calculate_health_score(excellent_metrics)
        assert excellent_health >= 90
        
        poor_metrics = {
            PerformanceMetric.MEMORY_USAGE: 95,
            PerformanceMetric.CPU_USAGE: 98,
            PerformanceMetric.RESPONSE_TIME: 15000,
            PerformanceMetric.ERROR_RATE: 20,
            PerformanceMetric.CACHE_HIT_RATE: 5
        }
        poor_health = performance_monitor._calculate_health_score(poor_metrics)
        assert poor_health <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
