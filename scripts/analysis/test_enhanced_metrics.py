#!/usr/bin/env python3
"""
Test the enhanced metrics system to validate the implementation.
"""

import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from giflab.enhanced_metrics import (
    calculate_composite_quality,
    calculate_efficiency_metric,
    get_enhanced_weights_info,
    process_metrics_with_enhanced_quality
)
from giflab.config import DEFAULT_METRICS_CONFIG
import pandas as pd


def test_enhanced_weights():
    """Test that enhanced weights sum to 1.0 and are properly distributed."""
    print("=== Testing Enhanced Weights ===")
    
    weights_info = get_enhanced_weights_info()
    
    print(f"Core structural (40%): {weights_info['core_structural']['total']:.3f}")
    print(f"Signal quality (25%): {weights_info['signal_quality']['total']:.3f}")
    print(f"Advanced structural (20%): {weights_info['advanced_structural']['total']:.3f}")
    print(f"Perceptual quality (10%): {weights_info['perceptual_quality']['total']:.3f}")
    print(f"Temporal consistency (5%): {weights_info['temporal_consistency']['total']:.3f}")
    print(f"Grand total: {weights_info['grand_total']:.6f}")
    
    assert abs(weights_info['grand_total'] - 1.0) < 1e-6, f"Weights don't sum to 1.0: {weights_info['grand_total']}"
    print("✓ Weights validation passed")


def test_sample_calculation():
    """Test enhanced composite quality calculation with sample data."""
    print("\n=== Testing Sample Calculation ===")
    
    # Sample metrics from the existing results
    sample_metrics = {
        'ssim_mean': 0.8,
        'ms_ssim_mean': 0.75,
        'psnr_mean': 0.6,
        'mse_mean': 0.05,
        'fsim_mean': 0.85,
        'gmsd_mean': 0.1,
        'chist_mean': 0.9,
        'edge_similarity_mean': 0.8,
        'texture_similarity_mean': 0.75,
        'sharpness_similarity_mean': 0.7,
        # Wave 7: composite reads the honest single-stream ``_compressed`` key.
        'temporal_consistency_compressed': 0.95,
        'compression_ratio': 2.5
    }
    
    # Test composite quality
    composite_quality = calculate_composite_quality(sample_metrics)
    print(f"Composite quality: {composite_quality:.4f}")
    
    # Test efficiency calculation
    efficiency = calculate_efficiency_metric(sample_metrics['compression_ratio'], composite_quality)
    print(f"Efficiency: {efficiency:.4f}")
    
    # Test full processing
    processed = process_metrics_with_enhanced_quality(sample_metrics.copy())
    print(f"Full processing result keys: {list(processed.keys())}")
    
    assert 0.0 <= composite_quality <= 1.0, f"Composite quality out of range: {composite_quality}"
    assert efficiency > 0, f"Efficiency should be positive: {efficiency}"
    print("✓ Sample calculation passed")


def test_real_data_validation():
    """Test with real data from enhanced results."""
    print("\n=== Testing Real Data Validation ===")
    
    # Load the enhanced results
    results_path = Path("test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv")
    
    if not results_path.exists():
        print("Enhanced results file not found, skipping real data test")
        return
    
    df = pd.read_csv(results_path)
    
    # Check that new columns exist
    required_cols = ['composite_quality', 'efficiency']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check value ranges
    successful_rows = df[df['success'] == True]
    
    composite_quality = successful_rows['composite_quality']
    efficiency = successful_rows['efficiency']
    
    print(f"Composite quality range: {composite_quality.min():.3f} - {composite_quality.max():.3f}")
    print(f"Efficiency range: {efficiency.min():.3f} - {efficiency.max():.3f}")
    
    # Validation checks
    assert composite_quality.min() >= 0.0, "Composite quality has negative values"
    assert composite_quality.max() <= 1.0, "Composite quality exceeds 1.0"
    assert efficiency.min() >= 0.0, "Efficiency has negative values"
    
    # Show quality statistics
    print(f"Mean composite quality: {composite_quality.mean():.3f}")
    print(f"Standard deviation: {composite_quality.std():.3f}")
    
    print("✓ Real data validation passed")


def test_quality_distribution():
    """Show composite quality distribution analysis."""
    print("\n=== Quality Distribution Analysis ===")
    
    results_path = Path("test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv")
    
    if not results_path.exists():
        print("Enhanced results file not found, skipping analysis")
        return
    
    df = pd.read_csv(results_path)
    successful_rows = df[df['success'] == True]
    
    # Show composite quality distribution
    composite_quality = successful_rows['composite_quality']
    
    print(f"Composite quality statistics:")
    print(composite_quality.describe())
    
    # Show top performers
    print(f"\nTop 5 pipelines by composite quality:")
    top_quality = successful_rows.nlargest(5, 'composite_quality')[['pipeline_id', 'composite_quality', 'efficiency']]
    print(top_quality.to_string(index=False))
    
    print("✓ Quality distribution analysis completed")


def main():
    """Run all tests."""
    try:
        test_enhanced_weights()
        test_sample_calculation()
        test_real_data_validation()
        test_quality_distribution()
        print("\n🎉 All tests passed! Composite quality metrics system is working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)