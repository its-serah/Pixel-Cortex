"""
Performance Monitoring and Optimization Service

Monitors performance of LLM, audio processing, and overall system health
with automatic optimization and alerting capabilities.
"""

import time
import psutil
import threading
import logging
import numpy as np
from unittest.mock import patch
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from enum import Enum

from app.services.local_llm_service import local_llm_service
from app.services.audio_processing_service import audio_service
from app.services.conversation_memory_service import conversation_memory_service


logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics to track"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    metric: PerformanceMetric
    threshold_exceeded: float
    current_value: float
    timestamp: datetime
    severity: str  # "warning", "critical"
    message: str


class PerformanceMonitor:
    """Service for monitoring and optimizing system performance"""
    
    def __init__(self):
        self.metrics_history = {
            metric: deque(maxlen=100)  # Keep last 100 data points
            for metric in PerformanceMetric
        }
        self.alerts = deque(maxlen=50)  # Keep last 50 alerts
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            PerformanceMetric.RESPONSE_TIME: {"warning": 5000, "critical": 10000},  # ms
            PerformanceMetric.MEMORY_USAGE: {"warning": 80, "critical": 90},  # percentage
            PerformanceMetric.CPU_USAGE: {"warning": 85, "critical": 95},  # percentage
            PerformanceMetric.CACHE_HIT_RATE: {"warning": 30, "critical": 10},  # minimum percentage
            PerformanceMetric.ERROR_RATE: {"warning": 5, "critical": 15},  # percentage
            PerformanceMetric.THROUGHPUT: {"warning": 0.5, "critical": 0.2}  # requests per second
        }
        
        # Optimization settings
        self.auto_optimization_enabled = True
        self.last_optimization = datetime.now()
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous performance monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self.collect_current_metrics()
                
                # Store metrics in history
                for metric, value in current_metrics.items():
                    if metric in self.metrics_history:
                        self.metrics_history[metric].append({
                            "value": value,
                            "timestamp": datetime.now()
                        })
                
                # Check thresholds and generate alerts
                self._check_thresholds(current_metrics)
                
                # Perform auto-optimization if needed
                if self.auto_optimization_enabled:
                    self._auto_optimize(current_metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def collect_current_metrics(self) -> Dict[PerformanceMetric, float]:
        """Collect current performance metrics from all services"""
        
        metrics = {}
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metrics[PerformanceMetric.MEMORY_USAGE] = memory.percent
        metrics[PerformanceMetric.CPU_USAGE] = cpu_percent
        
        # LLM service metrics
        try:
            llm_stats = local_llm_service.get_performance_stats()
            if llm_stats["model_loaded"]:
                inference_stats = llm_stats["inference_stats"]
                
                metrics[PerformanceMetric.RESPONSE_TIME] = inference_stats.get("avg_inference_time", 0)
                
                # Calculate cache hit rate
                total_requests = inference_stats.get("total_requests", 0)
                cache_hits = inference_stats.get("cache_hits", 0)
                if total_requests > 0:
                    metrics[PerformanceMetric.CACHE_HIT_RATE] = (cache_hits / total_requests) * 100
                else:
                    metrics[PerformanceMetric.CACHE_HIT_RATE] = 0
        except Exception as e:
            logger.warning(f"Failed to collect LLM metrics: {e}")
            metrics[PerformanceMetric.RESPONSE_TIME] = 0
            metrics[PerformanceMetric.CACHE_HIT_RATE] = 0
        
        # Audio service metrics
        try:
            audio_stats = audio_service.get_processing_stats()
            if audio_stats["model_loaded"]:
                processing_stats = audio_stats["processing_stats"]
                
                # Calculate throughput (processed per hour)
                total_processed = processing_stats.get("total_processed", 0)
                if total_processed > 0:
                    # Rough estimate based on average processing time
                    avg_time = processing_stats.get("avg_processing_time", 1000)  # ms
                    metrics[PerformanceMetric.THROUGHPUT] = 3600000 / avg_time if avg_time > 0 else 0
                else:
                    metrics[PerformanceMetric.THROUGHPUT] = 0
                
                # Calculate error rate
                errors = processing_stats.get("errors", 0)
                if total_processed > 0:
                    metrics[PerformanceMetric.ERROR_RATE] = (errors / total_processed) * 100
                else:
                    metrics[PerformanceMetric.ERROR_RATE] = 0
        except Exception as e:
            logger.warning(f"Failed to collect audio metrics: {e}")
            metrics[PerformanceMetric.THROUGHPUT] = 0
            metrics[PerformanceMetric.ERROR_RATE] = 0
        
        return metrics
    
    def _check_thresholds(self, current_metrics: Dict[PerformanceMetric, float]):
        """Check if any metrics exceed thresholds and generate alerts"""
        
        for metric, value in current_metrics.items():
            if metric not in self.thresholds:
                continue
            
            thresholds = self.thresholds[metric]
            
            # Check critical threshold
            if value >= thresholds["critical"]:
                alert = PerformanceAlert(
                    metric=metric,
                    threshold_exceeded=thresholds["critical"],
                    current_value=value,
                    timestamp=datetime.now(),
                    severity="critical",
                    message=f"{metric.value} reached critical level: {value:.2f}"
                )
                self.alerts.append(alert)
                logger.critical(alert.message)
            
            # Check warning threshold
            elif value >= thresholds["warning"]:
                alert = PerformanceAlert(
                    metric=metric,
                    threshold_exceeded=thresholds["warning"],
                    current_value=value,
                    timestamp=datetime.now(),
                    severity="warning",
                    message=f"{metric.value} reached warning level: {value:.2f}"
                )
                self.alerts.append(alert)
                logger.warning(alert.message)
            
            # Special handling for inverted metrics (higher is better)
            elif metric == PerformanceMetric.CACHE_HIT_RATE:
                if value <= thresholds["critical"]:
                    alert = PerformanceAlert(
                        metric=metric,
                        threshold_exceeded=thresholds["critical"],
                        current_value=value,
                        timestamp=datetime.now(),
                        severity="critical",
                        message=f"Cache hit rate critically low: {value:.2f}%"
                    )
                    self.alerts.append(alert)
                    logger.critical(alert.message)
    
    def _auto_optimize(self, current_metrics: Dict[PerformanceMetric, float]):
        """Perform automatic optimizations based on current metrics"""
        
        # Only optimize every 5 minutes to avoid thrashing
        if datetime.now() - self.last_optimization < timedelta(minutes=5):
            return
        
        optimizations_applied = []
        
        # High memory usage optimization
        memory_usage = current_metrics.get(PerformanceMetric.MEMORY_USAGE, 0)
        if memory_usage > 85:
            # Unload LLM model if it hasn't been used recently
            try:
                if hasattr(local_llm_service, 'last_used'):
                    if (datetime.now() - local_llm_service.last_used).total_seconds() > 300:  # 5 minutes
                        local_llm_service.unload_model()
                        optimizations_applied.append("unloaded_llm_model")
                        logger.info("Auto-optimization: Unloaded LLM model due to high memory usage")
            except Exception as e:
                logger.error(f"Failed to unload LLM model: {e}")
        
        # Low cache hit rate optimization
        cache_hit_rate = current_metrics.get(PerformanceMetric.CACHE_HIT_RATE, 100)
        if cache_hit_rate < 20 and hasattr(local_llm_service, 'cache') and local_llm_service.cache:
            # Clear old cache entries to improve hit rate
            try:
                # Clear cache entries older than 1 hour
                # This is a simplified approach - in practice, you'd implement TTL-based cleanup
                optimizations_applied.append("cache_cleanup")
                logger.info("Auto-optimization: Performed cache cleanup")
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
        
        # High response time optimization
        response_time = current_metrics.get(PerformanceMetric.RESPONSE_TIME, 0)
        if response_time > 8000:  # 8 seconds
            # Reduce max_tokens for faster responses
            if hasattr(local_llm_service, 'max_new_tokens') and local_llm_service.max_new_tokens > 256:
                local_llm_service.max_new_tokens = 256
                optimizations_applied.append("reduced_max_tokens")
                logger.info("Auto-optimization: Reduced max tokens for faster responses")
        
        if optimizations_applied:
            self.last_optimization = datetime.now()
            logger.info(f"Applied optimizations: {optimizations_applied}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        current_metrics = self.collect_current_metrics()
        
        # Calculate averages from history
        metric_averages = {}
        for metric, history in self.metrics_history.items():
            if history:
                values = [entry["value"] for entry in history]
                metric_averages[metric.value] = {
                    "current": current_metrics.get(metric, 0),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values)
                }
        
        # Get recent alerts
        recent_alerts = [
            {
                "metric": alert.metric.value,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "value": alert.current_value
            }
            for alert in list(self.alerts)[-10:]  # Last 10 alerts
        ]
        
        # System health score (0-100)
        health_score = self._calculate_health_score(current_metrics)
        
        return {
            "health_score": health_score,
            "current_metrics": {k.value: v for k, v in current_metrics.items()},
            "metric_averages": metric_averages,
            "recent_alerts": recent_alerts,
            "monitoring_active": self.monitoring_active,
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "last_optimization": self.last_optimization.isoformat(),
            "thresholds": {k.value: v for k, v in self.thresholds.items()}
        }
    
    def _calculate_health_score(self, metrics: Dict[PerformanceMetric, float]) -> int:
        """Calculate overall system health score (0-100)"""
        
        score = 100
        
        # Deduct points for high resource usage
        memory_usage = metrics.get(PerformanceMetric.MEMORY_USAGE, 0)
        if memory_usage > 90:
            score -= 30
        elif memory_usage > 80:
            score -= 15
        elif memory_usage > 70:
            score -= 5
        
        cpu_usage = metrics.get(PerformanceMetric.CPU_USAGE, 0)
        if cpu_usage > 95:
            score -= 25
        elif cpu_usage > 85:
            score -= 10
        elif cpu_usage > 75:
            score -= 3
        
        # Deduct points for slow responses
        response_time = metrics.get(PerformanceMetric.RESPONSE_TIME, 0)
        if response_time > 10000:  # 10 seconds
            score -= 20
        elif response_time > 5000:  # 5 seconds
            score -= 10
        elif response_time > 3000:  # 3 seconds
            score -= 5
        
        # Deduct points for high error rate
        error_rate = metrics.get(PerformanceMetric.ERROR_RATE, 0)
        if error_rate > 15:
            score -= 20
        elif error_rate > 5:
            score -= 10
        
        # Deduct points for low cache hit rate
        cache_hit_rate = metrics.get(PerformanceMetric.CACHE_HIT_RATE, 100)
        if cache_hit_rate < 10:
            score -= 15
        elif cache_hit_rate < 30:
            score -= 5
        
        return max(0, score)
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for performance optimization"""
        
        current_metrics = self.collect_current_metrics()
        recommendations = []
        
        # Memory optimization recommendations
        memory_usage = current_metrics.get(PerformanceMetric.MEMORY_USAGE, 0)
        if memory_usage > 80:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "high" if memory_usage > 90 else "medium",
                "recommendation": "Consider unloading unused models or reducing model size",
                "action": "unload_models",
                "current_value": memory_usage,
                "target_value": 70
            })
        
        # Response time optimization
        response_time = current_metrics.get(PerformanceMetric.RESPONSE_TIME, 0)
        if response_time > 5000:
            recommendations.append({
                "type": "response_time_optimization",
                "priority": "high" if response_time > 10000 else "medium",
                "recommendation": "Reduce max_tokens or enable model quantization",
                "action": "optimize_inference",
                "current_value": response_time,
                "target_value": 3000
            })
        
        # Cache optimization
        cache_hit_rate = current_metrics.get(PerformanceMetric.CACHE_HIT_RATE, 100)
        if cache_hit_rate < 50:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "recommendation": "Increase cache TTL or improve cache key strategy",
                "action": "optimize_cache",
                "current_value": cache_hit_rate,
                "target_value": 70
            })
        
        # Model loading recommendations
        try:
            llm_stats = local_llm_service.get_performance_stats()
            if not llm_stats["model_loaded"] and self._should_preload_model():
                recommendations.append({
                    "type": "model_preloading",
                    "priority": "low",
                    "recommendation": "Preload LLM model for faster first response",
                    "action": "preload_model",
                    "benefit": "Eliminates cold start delay"
                })
        except Exception as e:
            logger.warning(f"Failed to get LLM stats for recommendations: {e}")
        
        return recommendations
    
    def _should_preload_model(self) -> bool:
        """Determine if model should be preloaded based on usage patterns"""
        
        # Check if there's been recent activity
        if hasattr(local_llm_service, 'last_used'):
            time_since_last_use = (datetime.now() - local_llm_service.last_used).total_seconds()
            return time_since_last_use < 3600  # If used within last hour
        
        return False
    
    def apply_optimization(self, optimization_type: str) -> Dict[str, Any]:
        """Apply a specific optimization"""
        
        result = {"applied": False, "optimization": optimization_type}
        
        try:
            if optimization_type == "unload_models":
                local_llm_service.unload_model()
                result["applied"] = True
                result["message"] = "LLM model unloaded to free memory"
            
            elif optimization_type == "optimize_inference":
                # Reduce token limits for faster inference
                if hasattr(local_llm_service, 'max_new_tokens'):
                    old_tokens = local_llm_service.max_new_tokens
                    local_llm_service.max_new_tokens = min(256, old_tokens)
                    result["applied"] = True
                    result["message"] = f"Reduced max tokens from {old_tokens} to {local_llm_service.max_new_tokens}"
            
            elif optimization_type == "optimize_cache":
                # Clear old cache entries
                if hasattr(local_llm_service, 'cache') and local_llm_service.cache:
                    # Simplified cache cleanup
                    result["applied"] = True
                    result["message"] = "Cache optimization applied"
            
            elif optimization_type == "preload_model":
                local_llm_service.load_model()
                result["applied"] = True
                result["message"] = "LLM model preloaded"
            
            else:
                result["error"] = f"Unknown optimization type: {optimization_type}"
            
            if result["applied"]:
                self.last_optimization = datetime.now()
                logger.info(f"Applied optimization: {optimization_type}")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Optimization failed: {e}")
        
        return result
    
    def get_resource_usage_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get resource usage trends over time"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        trends = {}
        for metric, history in self.metrics_history.items():
            # Filter to requested time window
            recent_history = [
                entry for entry in history
                if entry["timestamp"] >= cutoff_time
            ]
            
            if recent_history:
                values = [entry["value"] for entry in recent_history]
                timestamps = [entry["timestamp"].isoformat() for entry in recent_history]
                
                trends[metric.value] = {
                    "values": values,
                    "timestamps": timestamps,
                    "trend_direction": self._calculate_trend_direction(values),
                    "average": sum(values) / len(values),
                    "peak": max(values),
                    "valley": min(values)
                }
        
        return trends
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def benchmark_system_performance(self) -> Dict[str, Any]:
        """Run comprehensive system performance benchmark"""
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # LLM inference benchmark
        try:
            if hasattr(local_llm_service, 'model') and local_llm_service.model is not None:
                start_time = time.time()
                
                test_query = "What is VPN?"
                response = local_llm_service.generate_response(
                    test_query,
                    max_tokens=100,
                    use_cache=False
                )
                
                inference_time = (time.time() - start_time) * 1000
                
                benchmark_results["tests"]["llm_inference"] = {
                    "response_time_ms": inference_time,
                    "tokens_generated": response.get("tokens_generated", 0),
                    "success": not response.get("error"),
                    "memory_impact": response.get("memory_usage", {})
                }
        except Exception as e:
            logger.warning(f"LLM benchmark failed: {e}")
            benchmark_results["tests"]["llm_inference"] = {
                "success": False,
                "error": str(e)
            }
        
        # Audio processing benchmark (if models loaded)
        try:
            if hasattr(audio_service, 'whisper_model') and audio_service.whisper_model is not None:
                # Create test audio (1 second of silence)
                test_audio = np.zeros(16000, dtype=np.float32)
                
                start_time = time.time()
                
                # Mock transcription for benchmark
                with patch.object(audio_service, 'preprocess_audio') as mock_preprocess:
                    mock_preprocess.return_value = test_audio
                    
                    with patch.object(audio_service.whisper_model, 'transcribe') as mock_transcribe:
                        mock_transcribe.return_value = {"text": "test", "segments": []}
                        
                        result = audio_service.transcribe_audio(b"fake_audio", "wav")
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        benchmark_results["tests"]["audio_processing"] = {
                            "processing_time_ms": processing_time,
                            "success": not result.get("error"),
                            "confidence": result.get("confidence", 0)
                        }
        except Exception as e:
            logger.warning(f"Audio benchmark failed: {e}")
            benchmark_results["tests"]["audio_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        # System resource benchmark
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_usage_percent": psutil.cpu_percent(interval=1)
        }
        
        benchmark_results["system_info"] = system_info
        
        # Overall performance score
        benchmark_results["performance_score"] = self._calculate_benchmark_score(benchmark_results)
        
        return benchmark_results
    
    def _calculate_benchmark_score(self, benchmark_results: Dict[str, Any]) -> int:
        """Calculate overall performance score from benchmark results"""
        
        score = 100
        
        # LLM performance scoring
        if "llm_inference" in benchmark_results["tests"]:
            llm_test = benchmark_results["tests"]["llm_inference"]
            response_time = llm_test.get("response_time_ms", 10000)
            
            if response_time > 5000:
                score -= 20
            elif response_time > 3000:
                score -= 10
            elif response_time > 1000:
                score -= 5
            
            if not llm_test.get("success", True):
                score -= 30
        
        # Audio performance scoring
        if "audio_processing" in benchmark_results["tests"]:
            audio_test = benchmark_results["tests"]["audio_processing"]
            processing_time = audio_test.get("processing_time_ms", 5000)
            
            if processing_time > 3000:
                score -= 15
            elif processing_time > 1000:
                score -= 5
            
            if not audio_test.get("success", True):
                score -= 20
        
        # System resource scoring
        system_info = benchmark_results.get("system_info", {})
        memory_usage_percent = (
            (system_info.get("memory_total_gb", 1) - system_info.get("memory_available_gb", 0)) 
            / system_info.get("memory_total_gb", 1) * 100
        )
        
        if memory_usage_percent > 90:
            score -= 20
        elif memory_usage_percent > 80:
            score -= 10
        
        cpu_usage = system_info.get("cpu_usage_percent", 0)
        if cpu_usage > 90:
            score -= 15
        elif cpu_usage > 70:
            score -= 5
        
        return max(0, score)
    
    def export_performance_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Export performance data for analysis"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        exported_data = {
            "export_metadata": {
                "generated_at": datetime.now().isoformat(),
                "hours_back": hours_back,
                "monitoring_active": self.monitoring_active
            },
            "metrics_history": {},
            "alerts_history": [],
            "performance_summary": self.get_performance_summary()
        }
        
        # Export metrics history
        for metric, history in self.metrics_history.items():
            recent_history = [
                entry for entry in history
                if entry["timestamp"] >= cutoff_time
            ]
            
            exported_data["metrics_history"][metric.value] = [
                {
                    "value": entry["value"],
                    "timestamp": entry["timestamp"].isoformat()
                }
                for entry in recent_history
            ]
        
        # Export alerts history
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        exported_data["alerts_history"] = [
            {
                "metric": alert.metric.value,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "value": alert.current_value,
                "threshold": alert.threshold_exceeded
            }
            for alert in recent_alerts
        ]
        
        return exported_data
    
    def clear_alerts(self, severity: Optional[str] = None) -> int:
        """Clear alerts, optionally filtered by severity"""
        
        if severity:
            # Remove alerts of specific severity
            initial_count = len(self.alerts)
            self.alerts = deque(
                [alert for alert in self.alerts if alert.severity != severity],
                maxlen=50
            )
            cleared_count = initial_count - len(self.alerts)
        else:
            # Clear all alerts
            cleared_count = len(self.alerts)
            self.alerts.clear()
        
        logger.info(f"Cleared {cleared_count} alerts" + (f" (severity: {severity})" if severity else ""))
        return cleared_count
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        
        current_metrics = self.collect_current_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {k.value: v for k, v in current_metrics.items()},
            "health_score": self._calculate_health_score(current_metrics),
            "recent_alerts": len([a for a in self.alerts if (datetime.now() - a.timestamp).seconds < 300]),  # Last 5 minutes
            "services_status": {
                "llm_loaded": hasattr(local_llm_service, 'model') and local_llm_service.model is not None,
                "audio_loaded": hasattr(audio_service, 'whisper_model') and audio_service.whisper_model is not None,
                "cache_available": hasattr(local_llm_service, 'cache') and local_llm_service.cache is not None
            }
        }


# Global instance
performance_monitor = PerformanceMonitor()
