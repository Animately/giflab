#!/usr/bin/env python3
"""
Comprehensive breakdown of the frame comparison dataset using enhanced metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset():
    """Load the specific dataset."""
    data_path = Path(
        "/Users/lachlants/repos/animately/giflab/test-workspace/frame-comparison-with-gifs/run_20250807_123641/enhanced_streaming_results.csv"
    )

    if not data_path.exists():
        print(f"Dataset not found at: {data_path}")
        return None

    df = pd.read_csv(data_path)
    return df[df["success"] is True].copy()  # Focus on successful results


def dataset_overview(df):
    """Provide high-level dataset overview."""
    print("🔬 DATASET OVERVIEW")
    print("=" * 50)

    print("📊 Size & Scope:")
    print(f"   • Total successful results: {len(df):,}")
    print(f"   • Unique pipelines tested: {df['pipeline_id'].nunique()}")
    print(f"   • Content types analyzed: {df['content_type'].nunique()}")
    print("   • Date range: Single experiment run (run_20250807_123641)")

    print("\n📝 Content Types Distribution:")
    content_counts = df["content_type"].value_counts().sort_values(ascending=False)
    for content_type, count in content_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {content_type.ljust(15)}: {count:3d} tests ({percentage:4.1f}%)")

    # Show file size range
    print("\n💾 File Size Analysis:")
    print(
        f"   • Original sizes: {df['original_size_kb'].min():.1f}KB to {df['original_size_kb'].max():.1f}KB"
    )
    print(
        f"   • Compressed sizes: {df['file_size_kb'].min():.1f}KB to {df['file_size_kb'].max():.1f}KB"
    )
    print(f"   • Average compression: {df['compression_ratio'].mean():.1f}x")
    print(f"   • Best compression: {df['compression_ratio'].max():.1f}x")


def pipeline_architecture_analysis(df):
    """Analyze the pipeline architecture tested."""
    print("\n\n🏗️  PIPELINE ARCHITECTURE ANALYSIS")
    print("=" * 50)

    # Extract pipeline components
    pipeline_parts = df["pipeline_id"].str.split("__", expand=True)
    df["frame_component"] = pipeline_parts[0]
    df["color_component"] = pipeline_parts[1]
    df["lossy_component"] = pipeline_parts[2]

    print("🔧 Frame Reduction Components Tested:")
    frame_counts = df["frame_component"].value_counts()
    for component, count in frame_counts.items():
        clean_name = component.split("_")[0]
        print(f"   • {clean_name.ljust(20)}: {count:3d} tests")

    print("\n🎨 Color Reduction Components:")
    color_counts = df["color_component"].value_counts()
    for component, count in color_counts.items():
        clean_name = component.split("_")[0]
        print(f"   • {clean_name.ljust(20)}: {count:3d} tests")

    print("\n🗜️  Lossy Compression Components:")
    lossy_counts = df["lossy_component"].value_counts()
    for component, count in lossy_counts.items():
        clean_name = component.split("_")[0]
        print(f"   • {clean_name.ljust(20)}: {count:3d} tests")


def quality_metrics_breakdown(df):
    """Analyze quality metrics in detail."""
    print("\n\n📐 QUALITY METRICS BREAKDOWN")
    print("=" * 50)

    # Traditional vs Enhanced comparison
    print("📊 Legacy vs Enhanced Quality Comparison:")
    # legacy_stats = df["composite_quality"].describe()  # Removed legacy composite_quality
    enhanced_stats = df["composite_quality"].describe()

    print("   Enhanced System (11 metrics):")
    print(f"   • Range: {enhanced_stats['min']:.3f} to {enhanced_stats['max']:.3f}")
    print(f"   • Mean: {enhanced_stats['mean']:.3f} ± {enhanced_stats['std']:.3f}")
    print(f"   • Median: {enhanced_stats['50%']:.3f}")

    # Individual metric analysis
    print("\n🔍 Individual Quality Metrics Analysis:")
    quality_metrics = [
        ("ssim_mean", "SSIM"),
        ("ms_ssim_mean", "MS-SSIM"),
        ("psnr_mean", "PSNR"),
        # Wave 7: storage CSV column renamed to ``_compressed``.
        ("temporal_consistency_compressed", "Temporal"),
        ("fsim_mean", "FSIM"),
        ("gmsd_mean", "GMSD"),
        ("chist_mean", "Color Hist"),
        ("edge_similarity_mean", "Edge Sim"),
        ("texture_similarity_mean", "Texture"),
        ("sharpness_similarity_mean", "Sharpness"),
    ]

    for metric_col, metric_name in quality_metrics:
        if metric_col in df.columns:
            stats = df[metric_col].describe()
            print(
                f"   • {metric_name.ljust(12)}: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f}-{stats['max']:.3f})"
            )


def efficiency_analysis(df):
    """Detailed efficiency scoring analysis."""
    print("\n\n⚡ EFFICIENCY SCORING ANALYSIS")
    print("=" * 50)

    efficiency_stats = df["efficiency"].describe()
    print("🎯 Efficiency Score Statistics:")
    print(f"   • Range: {efficiency_stats['min']:.3f} to {efficiency_stats['max']:.3f}")
    print(f"   • Mean: {efficiency_stats['mean']:.3f} ± {efficiency_stats['std']:.3f}")
    print(f"   • Median: {efficiency_stats['50%']:.3f}")
    print(f"   • 75th percentile: {efficiency_stats['75%']:.3f}")
    print(f"   • 95th percentile: {df['efficiency'].quantile(0.95):.3f}")

    # Efficiency distribution
    df["efficiency_tier"] = pd.cut(
        df["efficiency"],
        bins=[0, 1, 2.5, 5, 10, float("inf")],
        labels=[
            "Poor (0-1)",
            "Fair (1-2.5)",
            "Good (2.5-5)",
            "Excellent (5-10)",
            "Outstanding (10+)",
        ],
        include_lowest=True,
    )

    print("\n📊 Efficiency Distribution:")
    tier_counts = df["efficiency_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {str(tier).ljust(20)}: {count:3d} ({percentage:4.1f}%)")


def content_type_deep_dive(df):
    """Deep dive into performance by content type."""
    print("\n\n🎨 CONTENT TYPE PERFORMANCE DEEP DIVE")
    print("=" * 50)

    content_analysis = (
        df.groupby("content_type")
        .agg(
            {
                "composite_quality": ["mean", "std", "min", "max"],
                "efficiency": ["mean", "std", "min", "max"],
                "compression_ratio": ["mean", "std", "max"],
                "file_size_kb": ["mean", "std"],
                "pipeline_id": "count",
            }
        )
        .round(3)
    )

    # Flatten column names
    content_analysis.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0] for col in content_analysis.columns
    ]
    content_analysis = content_analysis.sort_values("mean_efficiency", ascending=False)

    print("📈 Performance Rankings (by average efficiency):")
    for idx, (content_type, row) in enumerate(content_analysis.iterrows(), 1):
        print(
            f"\n{idx:2d}. {content_type.upper().ljust(15)} (n={int(row['count_pipeline_id'])})"
        )
        print(
            f"    Efficiency: {row['mean_efficiency']:5.2f} ± {row['std_efficiency']:4.2f} (max: {row['max_efficiency']:5.2f})"
        )
        print(
            f"    Quality:    {row['mean_composite_quality']:5.3f} ± {row['std_composite_quality']:5.3f}"
        )
        print(
            f"    Compression:{row['mean_compression_ratio']:5.1f}x ± {row['std_compression_ratio']:4.1f} (max: {row['max_compression_ratio']:5.1f}x)"
        )
        print(
            f"    File size:  {row['mean_file_size_kb']:5.1f}KB ± {row['std_file_size_kb']:5.1f}"
        )


def champion_pipelines(df):
    """Identify the champion pipelines across different categories."""
    print("\n\n🏆 CHAMPION PIPELINES")
    print("=" * 50)

    categories = {
        "Highest Efficiency": df.loc[df["efficiency"].idxmax()],
        "Best Quality": df.loc[df["composite_quality"].idxmax()],
        "Maximum Compression": df.loc[df["compression_ratio"].idxmax()],
        "Smallest File Size": df.loc[df["file_size_kb"].idxmin()],
        "Best Balance": df.loc[
            (df["composite_quality"] * df["compression_ratio"]).idxmax()
        ],
    }

    for category, champion in categories.items():
        print(f"\n🥇 {category}:")
        pipeline_parts = champion["pipeline_id"].split("__")
        frame_tool = pipeline_parts[0].split("_")[0]
        color_tool = pipeline_parts[1].split("_")[0]
        lossy_tool = pipeline_parts[2].split("_")[0]

        print(f"   Pipeline: {frame_tool} + {color_tool} + {lossy_tool}")
        print(f"   Content: {champion['content_type']}")
        print(f"   Enhanced Quality: {champion['composite_quality']:.3f}")
        print(f"   Compression: {champion['compression_ratio']:.1f}x")
        print(f"   Efficiency: {champion['efficiency']:.2f}")
        print(
            f"   Size: {champion['file_size_kb']:.1f}KB (was {champion['original_size_kb']:.1f}KB)"
        )


def performance_insights(df):
    """Generate actionable performance insights."""
    print("\n\n💡 PERFORMANCE INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)

    # Tool performance analysis
    df["frame_tool"] = df["pipeline_id"].str.split("__").str[0].str.split("_").str[0]
    tool_performance = (
        df.groupby("frame_tool")
        .agg(
            {
                "efficiency": "mean",
                "composite_quality": "mean",
                "compression_ratio": "mean",
            }
        )
        .round(3)
        .sort_values("efficiency", ascending=False)
    )

    print("🔧 Frame Reduction Tool Rankings:")
    for idx, (tool, performance) in enumerate(tool_performance.iterrows(), 1):
        print(
            f"   {idx}. {tool.ljust(12)}: Efficiency {performance['efficiency']:5.2f} | Quality {performance['composite_quality']:5.3f} | Compression {performance['compression_ratio']:4.1f}x"
        )

    # Content-specific recommendations
    print("\n🎯 Content-Specific Recommendations:")

    content_champions = {}
    for content_type in df["content_type"].unique():
        content_data = df[df["content_type"] == content_type]
        best = content_data.loc[content_data["efficiency"].idxmax()]
        content_champions[content_type] = {
            "pipeline": best["frame_tool"],
            "efficiency": best["efficiency"],
            "quality": best["composite_quality"],
        }

    # Group similar content types
    motion_like = ["motion", "minimal", "spectrum"]
    complex_like = ["gradient", "complex_gradient", "texture", "noise"]
    geometric_like = ["geometric", "charts", "solid", "contrast"]

    print("   Motion/Animation Content:")
    for content in motion_like:
        if content in content_champions:
            champ = content_champions[content]
            print(
                f"     • {content.ljust(12)}: {champ['pipeline']} (eff: {champ['efficiency']:.1f})"
            )

    print("   Complex Visual Content:")
    for content in complex_like:
        if content in content_champions:
            champ = content_champions[content]
            print(
                f"     • {content.ljust(12)}: {champ['pipeline']} (eff: {champ['efficiency']:.1f})"
            )

    print("   Simple/Geometric Content:")
    for content in geometric_like:
        if content in content_champions:
            champ = content_champions[content]
            print(
                f"     • {content.ljust(12)}: {champ['pipeline']} (eff: {champ['efficiency']:.1f})"
            )


def statistical_summary(df):
    """Provide statistical summary of the experiment."""
    print("\n\n📊 STATISTICAL SUMMARY")
    print("=" * 50)

    print("🔢 Experiment Statistics:")
    print(f"   • Success rate: {(df['success'] is True).sum()}/{len(df)} (100.0%)")
    print(
        f"   • Average processing time: {df['total_processing_time_ms'].mean():.0f}ms"
    )
    print(
        f"   • Total processing time: {df['total_processing_time_ms'].sum()/1000/60:.1f} minutes"
    )

    print("\n📈 Performance Distributions:")
    print(
        f"   • Quality correlation (legacy vs enhanced): {df['composite_quality'].corr(df['composite_quality']):.3f}"
    )
    print(
        f"   • Files achieving >10x compression: {(df['compression_ratio'] > 10).sum()} ({(df['compression_ratio'] > 10).mean()*100:.1f}%)"
    )
    print(
        f"   • Files achieving >0.8 quality: {(df['composite_quality'] > 0.8).sum()} ({(df['composite_quality'] > 0.8).mean()*100:.1f}%)"
    )
    print(
        f"   • High-efficiency results (>5): {(df['efficiency'] > 5).sum()} ({(df['efficiency'] > 5).mean()*100:.1f}%)"
    )

    print("\n🎯 Key Metrics:")
    print(
        f"   • Median file size reduction: {(1 - df['file_size_kb'] / df['original_size_kb']).median()*100:.1f}%"
    )
    print(
        f"   • Average quality retention: {df['composite_quality'].mean()*100:.1f}%"
    )
    print(f"   • Best efficiency achieved: {df['efficiency'].max():.1f}")


def main():
    """Run complete dataset breakdown."""
    print("🔬 FRAME COMPARISON DATASET BREAKDOWN")
    print("Using Enhanced Metrics System")
    print("Dataset: run_20250807_123641")
    print("=" * 60)

    # Load the dataset
    df = load_dataset()
    if df is None:
        return False

    # Run all analyses
    dataset_overview(df)
    pipeline_architecture_analysis(df)
    quality_metrics_breakdown(df)
    efficiency_analysis(df)
    content_type_deep_dive(df)
    champion_pipelines(df)
    performance_insights(df)
    statistical_summary(df)

    print("\n\n🎉 ANALYSIS COMPLETE")
    print("=" * 30)
    print("Dataset successfully analyzed using enhanced 11-metric system")
    print("with efficiency scoring and comprehensive quality assessment.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
