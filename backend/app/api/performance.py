"""
Performance Monitoring API Router

Provides REST endpoints for monitoring and optimizing system performance.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.performance_monitor import performance_monitor


router = APIRouter()


@router.get("/metrics", summary="Get real-time performance metrics")
async def get_real_time_metrics() -> Dict[str, Any]:
    """
    Get current real-time performance metrics including:
    - System resource usage (CPU, memory)
    - LLM response times and cache hit rates
    - Audio processing throughput
    - Overall health score
    """
    try:
        return performance_monitor.get_real_time_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect metrics: {str(e)}")


@router.get("/summary", summary="Get comprehensive performance summary")
async def get_performance_summary() -> Dict[str, Any]:
    """
    Get detailed performance summary including:
    - Historical metric averages
    - Recent alerts
    - Current monitoring status
    - Performance thresholds
    """
    try:
        return performance_monitor.get_performance_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get("/trends", summary="Get resource usage trends")
async def get_resource_trends(
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back (1-168)")
) -> Dict[str, Any]:
    """
    Get resource usage trends over the specified time period.
    Includes trend direction analysis and historical peaks/valleys.
    """
    try:
        return performance_monitor.get_resource_usage_trends(hours_back)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/recommendations", summary="Get optimization recommendations")
async def get_optimization_recommendations() -> List[Dict[str, Any]]:
    """
    Get intelligent recommendations for performance optimization
    based on current system metrics and usage patterns.
    """
    try:
        return performance_monitor.get_optimization_recommendations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/optimize/{optimization_type}", summary="Apply performance optimization")
async def apply_optimization(optimization_type: str) -> Dict[str, Any]:
    """
    Apply a specific performance optimization.
    
    Available optimization types:
    - unload_models: Unload LLM models to free memory
    - optimize_inference: Reduce token limits for faster responses
    - optimize_cache: Clear old cache entries
    - preload_model: Preload LLM model for faster responses
    """
    try:
        return performance_monitor.apply_optimization(optimization_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/monitoring/start", summary="Start performance monitoring")
async def start_monitoring(
    interval_seconds: int = Query(30, ge=10, le=300, description="Monitoring interval in seconds")
) -> Dict[str, Any]:
    """
    Start continuous performance monitoring with specified interval.
    """
    try:
        performance_monitor.start_monitoring(interval_seconds)
        return {
            "status": "started",
            "interval_seconds": interval_seconds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/monitoring/stop", summary="Stop performance monitoring")
async def stop_monitoring() -> Dict[str, Any]:
    """
    Stop continuous performance monitoring.
    """
    try:
        performance_monitor.stop_monitoring()
        return {
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.post("/benchmark", summary="Run system performance benchmark")
async def run_benchmark(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Run comprehensive system performance benchmark including:
    - LLM inference speed test
    - Audio processing speed test
    - System resource analysis
    - Overall performance scoring
    """
    try:
        # Run benchmark in background to avoid blocking
        benchmark_results = performance_monitor.benchmark_system_performance()
        return benchmark_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/alerts", summary="Get performance alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (warning, critical)")
) -> List[Dict[str, Any]]:
    """
    Get performance alerts, optionally filtered by severity.
    """
    try:
        if severity and severity not in ["warning", "critical"]:
            raise HTTPException(status_code=400, detail="Severity must be 'warning' or 'critical'")
        
        alerts = list(performance_monitor.alerts)
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return [
            {
                "metric": alert.metric.value,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "value": alert.current_value,
                "threshold": alert.threshold_exceeded
            }
            for alert in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.delete("/alerts", summary="Clear performance alerts")
async def clear_alerts(
    severity: Optional[str] = Query(None, description="Clear alerts of specific severity")
) -> Dict[str, Any]:
    """
    Clear performance alerts, optionally filtered by severity.
    """
    try:
        if severity and severity not in ["warning", "critical"]:
            raise HTTPException(status_code=400, detail="Severity must be 'warning' or 'critical'")
        
        cleared_count = performance_monitor.clear_alerts(severity)
        
        return {
            "cleared_count": cleared_count,
            "severity_filter": severity,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear alerts: {str(e)}")


@router.get("/export", summary="Export performance data")
async def export_performance_data(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data to export (1-168)")
) -> Dict[str, Any]:
    """
    Export performance data for external analysis.
    Includes metrics history, alerts, and performance summaries.
    """
    try:
        return performance_monitor.export_performance_data(hours_back)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/health", summary="Get overall system health")
async def get_system_health() -> Dict[str, Any]:
    """
    Get overall system health status with simplified health indicators.
    """
    try:
        metrics = performance_monitor.collect_current_metrics()
        health_score = performance_monitor._calculate_health_score(metrics)
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"
        
        # Count recent alerts
        recent_critical_alerts = len([
            a for a in performance_monitor.alerts 
            if a.severity == "critical" and (datetime.now() - a.timestamp).seconds < 3600
        ])
        
        recent_warning_alerts = len([
            a for a in performance_monitor.alerts 
            if a.severity == "warning" and (datetime.now() - a.timestamp).seconds < 3600
        ])
        
        return {
            "health_score": health_score,
            "health_status": health_status,
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": performance_monitor.monitoring_active,
            "recent_alerts": {
                "critical": recent_critical_alerts,
                "warning": recent_warning_alerts
            },
            "key_metrics": {
                "memory_usage": metrics.get(performance_monitor.PerformanceMetric.MEMORY_USAGE, 0),
                "cpu_usage": metrics.get(performance_monitor.PerformanceMetric.CPU_USAGE, 0),
                "response_time": metrics.get(performance_monitor.PerformanceMetric.RESPONSE_TIME, 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.post("/thresholds", summary="Update performance thresholds")
async def update_thresholds(thresholds: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Update performance monitoring thresholds.
    
    Expected format:
    {
        "response_time": {"warning": 5000, "critical": 10000},
        "memory_usage": {"warning": 80, "critical": 90},
        ...
    }
    """
    try:
        updated_metrics = []
        
        for metric_name, threshold_values in thresholds.items():
            # Find matching PerformanceMetric enum
            matching_metric = None
            for metric in performance_monitor.PerformanceMetric:
                if metric.value == metric_name:
                    matching_metric = metric
                    break
            
            if matching_metric:
                # Validate threshold values
                if "warning" not in threshold_values or "critical" not in threshold_values:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Threshold for {metric_name} must include 'warning' and 'critical' values"
                    )
                
                # Update thresholds
                performance_monitor.thresholds[matching_metric] = threshold_values
                updated_metrics.append(metric_name)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown metric: {metric_name}"
                )
        
        return {
            "updated_metrics": updated_metrics,
            "timestamp": datetime.now().isoformat(),
            "current_thresholds": {
                k.value: v for k, v in performance_monitor.thresholds.items()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")


@router.get("/status", summary="Get monitoring service status")
async def get_monitoring_status() -> Dict[str, Any]:
    """
    Get current status of the performance monitoring service.
    """
    try:
        return {
            "monitoring_active": performance_monitor.monitoring_active,
            "auto_optimization_enabled": performance_monitor.auto_optimization_enabled,
            "last_optimization": performance_monitor.last_optimization.isoformat(),
            "total_alerts": len(performance_monitor.alerts),
            "metrics_tracked": len(performance_monitor.metrics_history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/auto-optimization/toggle", summary="Toggle auto-optimization")
async def toggle_auto_optimization(enabled: bool) -> Dict[str, Any]:
    """
    Enable or disable automatic performance optimization.
    """
    try:
        performance_monitor.auto_optimization_enabled = enabled
        
        return {
            "auto_optimization_enabled": enabled,
            "timestamp": datetime.now().isoformat(),
            "message": f"Auto-optimization {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle auto-optimization: {str(e)}")
