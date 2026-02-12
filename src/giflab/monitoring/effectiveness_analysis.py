"""
Cache effectiveness analysis and optimization recommendations.

This module provides comprehensive analysis of cache performance data:
- Cross-system effectiveness analysis combining metrics from multiple sources
- Optimization recommendations based on statistical analysis
- Performance regression detection and alerting
- Decision support for cache configuration and deployment

Phase 3.2 Implementation: Transform cache effectiveness data into actionable insights.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .baseline_framework import BaselineStatistics, get_baseline_framework
from .cache_effectiveness import (
    CacheEffectivenessStats,
    get_cache_effectiveness_monitor,
)
from .memory_monitor import get_cache_memory_tracker

logger = logging.getLogger(__name__)

class CacheRecommendation(Enum):
    """Cache deployment recommendations."""
    ENABLE_PRODUCTION = "enable_production"          # Strong evidence for enabling caching
    ENABLE_WITH_MONITORING = "enable_with_monitoring"  # Moderate evidence, monitor closely
    SELECTIVE_ENABLE = "selective_enable"             # Enable for specific operations only
    KEEP_DISABLED = "keep_disabled"                   # Evidence suggests caching not beneficial
    INSUFFICIENT_DATA = "insufficient_data"           # Need more data for decision
    PERFORMANCE_REGRESSION = "performance_regression" # Caching causing performance issues

@dataclass
class CacheEffectivenessAnalysis:
    """Comprehensive cache effectiveness analysis results."""
    # Overall assessment
    recommendation: CacheRecommendation
    confidence_score: float  # 0.0 to 1.0
    
    # Key metrics summary
    overall_hit_rate: float = 0.0
    average_performance_improvement: float = 0.0
    memory_efficiency_score: float = 0.0
    
    # Detailed analysis
    cache_stats_summary: dict[str, Any] = field(default_factory=dict)
    baseline_comparison_summary: dict[str, Any] = field(default_factory=dict)
    memory_analysis: dict[str, Any] = field(default_factory=dict)
    
    # Recommendations and insights
    optimization_recommendations: list[str] = field(default_factory=list)
    performance_insights: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    
    # Configuration suggestions
    suggested_cache_sizes: dict[str, float] = field(default_factory=dict)
    suggested_eviction_thresholds: dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    analysis_timestamp: float = field(default_factory=time.time)
    data_collection_period_hours: float = 0.0
    total_operations_analyzed: int = 0

class CacheEffectivenessAnalyzer:
    """Analyzes cache effectiveness data and generates optimization recommendations."""
    
    def __init__(self,
                 min_confidence_threshold: float = 0.7,
                 min_analysis_period_hours: float = 1.0):
        self.min_confidence_threshold = min_confidence_threshold
        self.min_analysis_period_hours = min_analysis_period_hours
        
        # Analysis thresholds
        self.excellent_hit_rate_threshold = 0.8
        self.good_hit_rate_threshold = 0.6
        self.poor_hit_rate_threshold = 0.3
        
        self.excellent_performance_improvement = 0.3
        self.good_performance_improvement = 0.15
        self.acceptable_performance_improvement = 0.05
        
        logger.info("Cache effectiveness analyzer initialized")
    
    def analyze_cache_effectiveness(self) -> CacheEffectivenessAnalysis:
        """Perform comprehensive cache effectiveness analysis."""
        logger.info("Starting comprehensive cache effectiveness analysis")
        
        # Collect data from all monitoring systems
        effectiveness_monitor = get_cache_effectiveness_monitor()
        baseline_framework = get_baseline_framework()
        memory_tracker = get_cache_memory_tracker()
        
        # Get effectiveness statistics
        cache_stats = effectiveness_monitor.get_all_cache_stats()
        system_summary = effectiveness_monitor.get_system_effectiveness_summary()
        
        # Get baseline performance comparisons
        baseline_stats = baseline_framework.get_all_baseline_statistics()
        baseline_report = baseline_framework.generate_performance_report()
        
        # Get memory analysis
        memory_summary = memory_tracker.get_system_effectiveness_summary()
        
        # Perform comprehensive analysis
        analysis = self._perform_comprehensive_analysis(
            cache_stats=cache_stats,
            system_summary=system_summary,
            baseline_stats=baseline_stats,
            baseline_report=baseline_report,
            memory_summary=memory_summary
        )
        
        logger.info(f"Cache effectiveness analysis completed: {analysis.recommendation.value} (confidence: {analysis.confidence_score:.2f})")
        
        return analysis
    
    def _perform_comprehensive_analysis(self,
                                       cache_stats: dict[str, CacheEffectivenessStats],
                                       system_summary: dict[str, Any],
                                       baseline_stats: dict[str, BaselineStatistics],
                                       baseline_report: dict[str, Any],
                                       memory_summary: dict[str, Any]) -> CacheEffectivenessAnalysis:
        """Perform detailed analysis and generate recommendations."""
        analysis = CacheEffectivenessAnalysis(
            recommendation=CacheRecommendation.INSUFFICIENT_DATA,
            confidence_score=0.0
        )
        
        # Extract key metrics
        overall_hit_rate = system_summary.get("overall_hit_rate", 0.0)
        total_operations = system_summary.get("total_operations", 0)
        monitoring_duration = system_summary.get("monitoring_duration_hours", 0.0)
        
        avg_performance_improvement = baseline_report.get("performance_analysis", {}).get("average_improvement", 0.0)
        
        analysis.overall_hit_rate = overall_hit_rate
        analysis.average_performance_improvement = avg_performance_improvement
        analysis.data_collection_period_hours = monitoring_duration
        analysis.total_operations_analyzed = total_operations
        
        # Check if we have sufficient data for analysis
        if not self._has_sufficient_data(cache_stats, baseline_stats, monitoring_duration):
            analysis.recommendation = CacheRecommendation.INSUFFICIENT_DATA
            analysis.confidence_score = 0.1
            analysis.optimization_recommendations = self._get_insufficient_data_recommendations()
            return analysis
        
        # Analyze cache hit rates
        hit_rate_analysis = self._analyze_hit_rates(cache_stats, overall_hit_rate)
        
        # Analyze performance impact
        performance_analysis = self._analyze_performance_impact(baseline_stats, avg_performance_improvement)
        
        # Analyze memory efficiency
        memory_analysis = self._analyze_memory_efficiency(cache_stats, memory_summary)
        
        # Combine analyses for overall recommendation
        recommendation, confidence = self._generate_overall_recommendation(
            hit_rate_analysis, performance_analysis, memory_analysis
        )
        
        analysis.recommendation = recommendation
        analysis.confidence_score = confidence
        analysis.memory_efficiency_score = memory_analysis.get("efficiency_score", 0.0)
        
        # Generate detailed summaries
        analysis.cache_stats_summary = self._summarize_cache_stats(cache_stats)
        analysis.baseline_comparison_summary = self._summarize_baseline_comparisons(baseline_stats)
        analysis.memory_analysis = memory_analysis
        
        # Generate optimization recommendations
        analysis.optimization_recommendations = self._generate_optimization_recommendations(
            hit_rate_analysis, performance_analysis, memory_analysis, cache_stats
        )
        
        # Generate performance insights
        analysis.performance_insights = self._generate_performance_insights(
            cache_stats, baseline_stats, system_summary
        )
        
        # Identify risk factors
        analysis.risk_factors = self._identify_risk_factors(
            cache_stats, memory_analysis, performance_analysis
        )
        
        # Generate configuration suggestions
        analysis.suggested_cache_sizes = self._suggest_cache_sizes(cache_stats, memory_analysis)
        analysis.suggested_eviction_thresholds = self._suggest_eviction_thresholds(memory_analysis)
        
        return analysis
    
    def _has_sufficient_data(self,
                            cache_stats: dict[str, CacheEffectivenessStats],
                            baseline_stats: dict[str, BaselineStatistics],
                            monitoring_duration: float) -> bool:
        """Check if we have sufficient data for meaningful analysis."""
        # Need minimum monitoring duration
        if monitoring_duration < self.min_analysis_period_hours:
            return False
        
        # Need some cache operations
        total_operations = sum(stats.total_operations for stats in cache_stats.values())
        if total_operations < 100:  # Minimum operations threshold
            return False
        
        # Need at least some baseline comparisons
        valid_baselines = sum(1 for stats in baseline_stats.values() if stats.min_samples_met)
        if valid_baselines == 0:
            return False
        
        return True
    
    def _analyze_hit_rates(self,
                          cache_stats: dict[str, CacheEffectivenessStats],
                          overall_hit_rate: float) -> dict[str, Any]:
        """Analyze cache hit rate effectiveness."""
        if not cache_stats:
            return {"overall_assessment": "no_data", "hit_rate_score": 0.0}
        
        hit_rates = [stats.hit_rate for stats in cache_stats.values() if stats.total_operations > 0]
        
        if not hit_rates:
            return {"overall_assessment": "no_data", "hit_rate_score": 0.0}
        
        # Calculate hit rate score (0.0 to 1.0)
        if overall_hit_rate >= self.excellent_hit_rate_threshold:
            assessment = "excellent"
            score = 1.0
        elif overall_hit_rate >= self.good_hit_rate_threshold:
            assessment = "good"
            score = 0.75
        elif overall_hit_rate >= self.poor_hit_rate_threshold:
            assessment = "poor"
            score = 0.4
        else:
            assessment = "very_poor"
            score = 0.1
        
        return {
            "overall_assessment": assessment,
            "hit_rate_score": score,
            "overall_hit_rate": overall_hit_rate,
            "cache_type_hit_rates": {cache_type: stats.hit_rate for cache_type, stats in cache_stats.items()},
            "hit_rate_variance": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0.0
        }
    
    def _analyze_performance_impact(self,
                                   baseline_stats: dict[str, BaselineStatistics],
                                   avg_improvement: float) -> dict[str, Any]:
        """Analyze performance impact of caching."""
        if not baseline_stats:
            return {"overall_assessment": "no_data", "performance_score": 0.0}
        
        valid_comparisons = [stats for stats in baseline_stats.values() if stats.statistical_significance]
        
        if not valid_comparisons:
            return {"overall_assessment": "insufficient_data", "performance_score": 0.0}
        
        # Calculate performance score
        if avg_improvement >= self.excellent_performance_improvement:
            assessment = "excellent"
            score = 1.0
        elif avg_improvement >= self.good_performance_improvement:
            assessment = "good"
            score = 0.75
        elif avg_improvement >= self.acceptable_performance_improvement:
            assessment = "acceptable"
            score = 0.5
        elif avg_improvement >= 0.0:
            assessment = "marginal"
            score = 0.25
        else:
            assessment = "negative"
            score = 0.0
        
        return {
            "overall_assessment": assessment,
            "performance_score": score,
            "average_improvement": avg_improvement,
            "valid_comparisons": len(valid_comparisons),
            "improvement_variance": statistics.stdev([stats.performance_improvement for stats in valid_comparisons]) if len(valid_comparisons) > 1 else 0.0,
            "operation_improvements": {stats.operation_type: stats.performance_improvement for stats in valid_comparisons}
        }
    
    def _analyze_memory_efficiency(self,
                                  cache_stats: dict[str, CacheEffectivenessStats],
                                  memory_summary: dict[str, Any]) -> dict[str, Any]:
        """Analyze memory efficiency of caching."""
        if "effectiveness_monitoring" in memory_summary and memory_summary["effectiveness_monitoring"] == "disabled":
            return {"overall_assessment": "monitoring_disabled", "efficiency_score": 0.5}
        
        # Calculate memory utilization metrics
        total_cache_memory = sum(stats.total_data_cached_mb for stats in cache_stats.values())
        total_evictions = sum(stats.evictions for stats in cache_stats.values())
        total_puts = sum(stats.puts for stats in cache_stats.values())
        
        # Memory efficiency score based on eviction rate and utilization
        if total_puts > 0:
            eviction_rate = total_evictions / total_puts
            if eviction_rate < 0.1:  # Less than 10% eviction rate
                efficiency_score = 1.0
                assessment = "excellent"
            elif eviction_rate < 0.3:  # Less than 30% eviction rate
                efficiency_score = 0.7
                assessment = "good"
            elif eviction_rate < 0.5:  # Less than 50% eviction rate
                efficiency_score = 0.4
                assessment = "acceptable"
            else:
                efficiency_score = 0.1
                assessment = "poor"
        else:
            efficiency_score = 0.0
            assessment = "no_data"
        
        return {
            "overall_assessment": assessment,
            "efficiency_score": efficiency_score,
            "total_cache_memory_mb": total_cache_memory,
            "total_evictions": total_evictions,
            "total_puts": total_puts,
            "eviction_rate": total_evictions / total_puts if total_puts > 0 else 0.0,
            "cache_type_eviction_rates": {
                cache_type: stats.evictions / stats.puts if stats.puts > 0 else 0.0
                for cache_type, stats in cache_stats.items()
            }
        }
    
    def _generate_overall_recommendation(self,
                                        hit_rate_analysis: dict[str, Any],
                                        performance_analysis: dict[str, Any],
                                        memory_analysis: dict[str, Any]) -> tuple[CacheRecommendation, float]:
        """Generate overall recommendation and confidence score."""
        # Calculate weighted scores
        hit_rate_score = hit_rate_analysis.get("hit_rate_score", 0.0)
        performance_score = performance_analysis.get("performance_score", 0.0)
        memory_score = memory_analysis.get("efficiency_score", 0.0)
        
        # Weights: performance is most important, then hit rate, then memory efficiency
        overall_score = (performance_score * 0.5) + (hit_rate_score * 0.35) + (memory_score * 0.15)
        
        # Generate recommendation based on combined score
        if overall_score >= 0.8:
            recommendation = CacheRecommendation.ENABLE_PRODUCTION
            confidence = min(0.95, overall_score)
        elif overall_score >= 0.6:
            recommendation = CacheRecommendation.ENABLE_WITH_MONITORING
            confidence = overall_score
        elif overall_score >= 0.4:
            recommendation = CacheRecommendation.SELECTIVE_ENABLE
            confidence = overall_score
        elif overall_score >= 0.2:
            recommendation = CacheRecommendation.KEEP_DISABLED
            confidence = 1.0 - overall_score  # High confidence in keeping disabled
        else:
            # Check for performance regression
            if performance_analysis.get("average_improvement", 0.0) < -0.05:
                recommendation = CacheRecommendation.PERFORMANCE_REGRESSION
                confidence = 0.8
            else:
                recommendation = CacheRecommendation.KEEP_DISABLED
                confidence = 0.7
        
        return recommendation, confidence
    
    def _generate_optimization_recommendations(self,
                                             hit_rate_analysis: dict[str, Any],
                                             performance_analysis: dict[str, Any],
                                             memory_analysis: dict[str, Any],
                                             cache_stats: dict[str, CacheEffectivenessStats]) -> list[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # Hit rate optimization
        overall_hit_rate = hit_rate_analysis.get("overall_hit_rate", 0.0)
        if overall_hit_rate < self.good_hit_rate_threshold:
            recommendations.append(f"Low hit rate ({overall_hit_rate:.1%}). Consider reviewing cache key strategies or increasing cache size.")
        
        # Performance optimization
        avg_improvement = performance_analysis.get("average_improvement", 0.0)
        if 0 < avg_improvement < self.acceptable_performance_improvement:
            recommendations.append("Marginal performance improvement detected. Optimize cache implementation or evaluate cost/benefit.")
        
        # Memory optimization
        high_eviction_caches = [
            cache_type for cache_type, rate in memory_analysis.get("cache_type_eviction_rates", {}).items()
            if rate > 0.3
        ]
        if high_eviction_caches:
            recommendations.append(f"High eviction rates detected for: {', '.join(high_eviction_caches)}. Consider increasing cache sizes or optimizing eviction policies.")
        
        # Cache-specific recommendations
        for cache_type, stats in cache_stats.items():
            if stats.cache_turnover_rate > 0.8:
                recommendations.append(f"{cache_type}: High turnover rate ({stats.cache_turnover_rate:.1%}). Review cache size and TTL settings.")
            
            if stats.hit_rate < 0.3 and stats.total_operations > 100:
                recommendations.append(f"{cache_type}: Very low hit rate ({stats.hit_rate:.1%}). Consider disabling or redesigning cache strategy.")
        
        return recommendations
    
    def _generate_performance_insights(self,
                                     cache_stats: dict[str, CacheEffectivenessStats],
                                     baseline_stats: dict[str, BaselineStatistics],
                                     system_summary: dict[str, Any]) -> list[str]:
        """Generate performance insights from analysis."""
        insights = []
        
        # System-wide insights
        total_ops = system_summary.get("total_operations", 0)
        monitoring_hours = system_summary.get("monitoring_duration_hours", 0)
        if monitoring_hours > 0:
            ops_per_hour = total_ops / monitoring_hours
            insights.append(f"Processing rate: {ops_per_hour:.0f} operations/hour over {monitoring_hours:.1f} hours")
        
        # Best performing cache types
        if cache_stats:
            best_hit_rate = max((stats.hit_rate for stats in cache_stats.values() if stats.total_operations > 10), default=0)
            if best_hit_rate > 0:
                best_cache = next((cache_type for cache_type, stats in cache_stats.items() if stats.hit_rate == best_hit_rate), None)
                insights.append(f"Best performing cache: {best_cache} ({best_hit_rate:.1%} hit rate)")
        
        # Performance improvement insights
        if baseline_stats:
            best_improvement = max((stats.performance_improvement for stats in baseline_stats.values() if stats.statistical_significance), default=0)
            if best_improvement > 0:
                best_operation = next((stats.operation_type for stats in baseline_stats.values() if stats.performance_improvement == best_improvement), None)
                insights.append(f"Highest performance gain: {best_operation} ({best_improvement:.1%} improvement)")
        
        return insights
    
    def _identify_risk_factors(self,
                              cache_stats: dict[str, CacheEffectivenessStats],
                              memory_analysis: dict[str, Any],
                              performance_analysis: dict[str, Any]) -> list[str]:
        """Identify potential risk factors with caching."""
        risks = []
        
        # Memory pressure risks
        if memory_analysis.get("efficiency_score", 0.0) < 0.4:
            risks.append("High memory pressure detected. Cache evictions may impact performance.")
        
        # Performance regression risks
        if performance_analysis.get("average_improvement", 0.0) < 0:
            risks.append("Negative performance impact detected. Caching may be adding overhead.")
        
        # Data staleness risks (high TTL with high hit rates)
        for cache_type, stats in cache_stats.items():
            if stats.hit_rate > 0.8 and stats.cache_turnover_rate < 0.1:
                risks.append(f"{cache_type}: Very high hit rate with low turnover may indicate stale data issues.")
        
        # Insufficient baseline data
        if performance_analysis.get("valid_comparisons", 0) < 2:
            risks.append("Limited baseline comparisons available. Recommendations may be less reliable.")
        
        return risks
    
    def _suggest_cache_sizes(self,
                            cache_stats: dict[str, CacheEffectivenessStats],
                            memory_analysis: dict[str, Any]) -> dict[str, float]:
        """Suggest optimal cache sizes based on analysis."""
        suggestions = {}
        
        for cache_type, stats in cache_stats.items():
            current_size = stats.total_data_cached_mb
            eviction_rate = stats.evictions / stats.puts if stats.puts > 0 else 0
            
            if eviction_rate > 0.5:  # High eviction rate - suggest larger cache
                suggested_size = current_size * 1.5
            elif eviction_rate > 0.3:  # Moderate eviction rate - modest increase
                suggested_size = current_size * 1.2
            elif eviction_rate < 0.1 and stats.hit_rate < 0.6:  # Low eviction but poor hit rate - may be too large
                suggested_size = current_size * 0.8
            else:  # Maintain current size
                suggested_size = current_size
            
            suggestions[cache_type] = max(suggested_size, 10.0)  # Minimum 10MB
        
        return suggestions
    
    def _suggest_eviction_thresholds(self, memory_analysis: dict[str, Any]) -> dict[str, float]:
        """Suggest optimal eviction thresholds based on memory analysis."""
        current_efficiency = memory_analysis.get("efficiency_score", 0.5)
        
        # Adjust thresholds based on efficiency
        if current_efficiency > 0.8:  # High efficiency - can be more aggressive
            return {
                "warning_threshold": 0.75,
                "critical_threshold": 0.85,
                "emergency_threshold": 0.95
            }
        elif current_efficiency > 0.6:  # Good efficiency - moderate thresholds
            return {
                "warning_threshold": 0.70,
                "critical_threshold": 0.80,
                "emergency_threshold": 0.95
            }
        else:  # Poor efficiency - conservative thresholds
            return {
                "warning_threshold": 0.60,
                "critical_threshold": 0.75,
                "emergency_threshold": 0.90
            }
    
    def _summarize_cache_stats(self, cache_stats: dict[str, CacheEffectivenessStats]) -> dict[str, Any]:
        """Summarize cache statistics for the analysis report."""
        if not cache_stats:
            return {"total_cache_types": 0}
        
        return {
            "total_cache_types": len(cache_stats),
            "total_operations": sum(stats.total_operations for stats in cache_stats.values()),
            "total_hits": sum(stats.hits for stats in cache_stats.values()),
            "total_misses": sum(stats.misses for stats in cache_stats.values()),
            "total_evictions": sum(stats.evictions for stats in cache_stats.values()),
            "cache_type_summaries": {
                cache_type: {
                    "hit_rate": stats.hit_rate,
                    "operations": stats.total_operations,
                    "eviction_rate": stats.evictions / stats.puts if stats.puts > 0 else 0.0,
                    "memory_mb": stats.total_data_cached_mb
                }
                for cache_type, stats in cache_stats.items()
            }
        }
    
    def _summarize_baseline_comparisons(self, baseline_stats: dict[str, BaselineStatistics]) -> dict[str, Any]:
        """Summarize baseline performance comparisons."""
        if not baseline_stats:
            return {"total_comparisons": 0}
        
        valid_comparisons = [stats for stats in baseline_stats.values() if stats.statistical_significance]
        
        return {
            "total_comparisons": len(baseline_stats),
            "statistically_significant": len(valid_comparisons),
            "operation_comparisons": {
                stats.operation_type: {
                    "performance_improvement": stats.performance_improvement,
                    "cached_samples": stats.cached_samples,
                    "non_cached_samples": stats.non_cached_samples,
                    "statistical_significance": stats.statistical_significance
                }
                for stats in baseline_stats.values()
            }
        }
    
    def _get_insufficient_data_recommendations(self) -> list[str]:
        """Get recommendations when insufficient data is available."""
        return [
            "Continue monitoring cache operations to collect more performance data.",
            "Enable A/B testing mode in baseline framework for comparative analysis.",
            "Ensure cache operations are being recorded properly by checking integration points.",
            f"Allow at least {self.min_analysis_period_hours} hours of monitoring before making decisions.",
            "Consider running controlled performance tests with synthetic workloads."
        ]

# Global analyzer instance
_effectiveness_analyzer: CacheEffectivenessAnalyzer | None = None

def get_effectiveness_analyzer() -> CacheEffectivenessAnalyzer:
    """Get singleton cache effectiveness analyzer."""
    global _effectiveness_analyzer
    if _effectiveness_analyzer is None:
        _effectiveness_analyzer = CacheEffectivenessAnalyzer()
    return _effectiveness_analyzer

def analyze_cache_effectiveness() -> CacheEffectivenessAnalysis:
    """Perform comprehensive cache effectiveness analysis."""
    analyzer = get_effectiveness_analyzer()
    return analyzer.analyze_cache_effectiveness()
