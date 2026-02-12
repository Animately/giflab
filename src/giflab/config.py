"""Configuration settings for GifLab."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompressionConfig:
    """Configuration for GIF compression variants."""

    # Frame keep ratios to test
    FRAME_KEEP_RATIOS: list[float] | None = None

    # Color palette sizes to test
    COLOR_KEEP_COUNTS: list[int] | None = None

    # Lossy compression levels to test
    LOSSY_LEVELS: list[int] | None = None

    # Supported engines
    ENGINES: list[str] | None = None

    def __post_init__(self) -> None:
        if self.FRAME_KEEP_RATIOS is None:
            self.FRAME_KEEP_RATIOS = [1.0]  # Frame reduction handled separately

        if self.COLOR_KEEP_COUNTS is None:
            # Prediction-required granularity: 5 color counts
            self.COLOR_KEEP_COUNTS = [256, 128, 64, 32, 16]

        if self.LOSSY_LEVELS is None:
            # Import from single source of truth
            from giflab.prediction.schemas import LOSSY_LEVELS as _LOSSY_LEVELS

            self.LOSSY_LEVELS = list(_LOSSY_LEVELS)

        if self.ENGINES is None:
            self.ENGINES = ["gifsicle", "animately"]


@dataclass
class MetricsConfig:
    """Configuration for quality metrics calculation."""

    # SSIM calculation modes: fast, optimized, full, comprehensive
    SSIM_MODE: str = "comprehensive"

    # Maximum frames to sample for SSIM calculation (balance accuracy/speed)
    SSIM_MAX_FRAMES: int = 30

    # Frame alignment: content-based (most robust visual matching)
    # No configuration needed - always uses content-based alignment

    # Enable comprehensive multi-metric calculation
    USE_COMPREHENSIVE_METRICS: bool = True

    # Enable temporal consistency analysis
    TEMPORAL_CONSISTENCY_ENABLED: bool = True

    # Enable raw (un-normalised) metric values alongside normalised ones
    RAW_METRICS: bool = False

    # Enable positional sampling (first, middle, last frame analysis)
    ENABLE_POSITIONAL_SAMPLING: bool = True
    POSITIONAL_METRICS: list[str] | None = None  # Will be set in __post_init__

    # Composite quality weights (must sum to 1.0)
    # Legacy 4-metric weights (for backward compatibility)
    SSIM_WEIGHT: float = 0.30
    MS_SSIM_WEIGHT: float = 0.35
    PSNR_WEIGHT: float = 0.25
    TEMPORAL_WEIGHT: float = 0.10

    # Enhanced 11-metric composite quality weights (comprehensive approach)
    # Core structural similarity metrics (40% total)
    ENHANCED_SSIM_WEIGHT: float = 0.18
    ENHANCED_MS_SSIM_WEIGHT: float = 0.22

    # Signal quality metrics (25% total)
    ENHANCED_PSNR_WEIGHT: float = 0.15
    ENHANCED_MSE_WEIGHT: float = 0.10

    # Advanced structural metrics (20% total)
    ENHANCED_FSIM_WEIGHT: float = 0.08
    ENHANCED_EDGE_WEIGHT: float = 0.07
    ENHANCED_GMSD_WEIGHT: float = 0.05

    # Perceptual quality metrics (10% total)
    ENHANCED_CHIST_WEIGHT: float = 0.04
    ENHANCED_SHARPNESS_WEIGHT: float = 0.03
    ENHANCED_TEXTURE_WEIGHT: float = 0.00  # Reduced to make room for LPIPS

    # Temporal consistency (5% total)
    ENHANCED_TEMPORAL_WEIGHT: float = 0.05

    # Deep perceptual metrics (3% total)
    ENHANCED_LPIPS_WEIGHT: float = 0.01  # Reduced to make room for SSIMULACRA2
    ENHANCED_SSIMULACRA2_WEIGHT: float = 0.02  # Modern perceptual metric

    # Enable enhanced composite quality calculation
    USE_ENHANCED_COMPOSITE_QUALITY: bool = True

    # Deep perceptual metrics configuration
    ENABLE_DEEP_PERCEPTUAL: bool = True
    DEEP_PERCEPTUAL_DEVICE: str = "auto"
    LPIPS_DOWNSCALE_SIZE: int = 512
    LPIPS_MAX_FRAMES: int = 100

    # SSIMULACRA2 configuration
    ENABLE_SSIMULACRA2: bool = True
    SSIMULACRA2_PATH: str = "/opt/homebrew/bin/ssimulacra2"
    SSIMULACRA2_MAX_FRAMES: int = 30
    SSIMULACRA2_QUALITY_THRESHOLD: float = 0.7  # Trigger for borderline quality

    # PSNR normalisation upper bound (dB) – values ≥ this map to 1.0
    PSNR_MAX_DB: float = 50.0

    # Edge-similarity Canny thresholds
    EDGE_CANNY_THRESHOLD1: int = 50
    EDGE_CANNY_THRESHOLD2: int = 150

    def __post_init__(self) -> None:
        # Validate legacy weights sum to 1.0 with proper floating point tolerance
        legacy_total = (
            self.SSIM_WEIGHT
            + self.MS_SSIM_WEIGHT
            + self.PSNR_WEIGHT
            + self.TEMPORAL_WEIGHT
        )
        tolerance = 1e-6  # More restrictive tolerance for configuration validation
        if abs(legacy_total - 1.0) > tolerance:
            raise ValueError(
                f"Legacy composite quality weights must sum to 1.0 (±{tolerance}), got {legacy_total:.10f}"
            )

        # Validate enhanced weights sum to 1.0
        enhanced_total = (
            self.ENHANCED_SSIM_WEIGHT
            + self.ENHANCED_MS_SSIM_WEIGHT
            + self.ENHANCED_PSNR_WEIGHT
            + self.ENHANCED_MSE_WEIGHT
            + self.ENHANCED_FSIM_WEIGHT
            + self.ENHANCED_EDGE_WEIGHT
            + self.ENHANCED_GMSD_WEIGHT
            + self.ENHANCED_CHIST_WEIGHT
            + self.ENHANCED_SHARPNESS_WEIGHT
            + self.ENHANCED_TEXTURE_WEIGHT
            + self.ENHANCED_TEMPORAL_WEIGHT
            + self.ENHANCED_LPIPS_WEIGHT
            + self.ENHANCED_SSIMULACRA2_WEIGHT
        )
        if abs(enhanced_total - 1.0) > tolerance:
            raise ValueError(
                f"Enhanced composite quality weights must sum to 1.0 (±{tolerance}), got {enhanced_total:.10f}"
            )

        # Validate all weights are non-negative
        legacy_weights = [
            self.SSIM_WEIGHT,
            self.MS_SSIM_WEIGHT,
            self.PSNR_WEIGHT,
            self.TEMPORAL_WEIGHT,
        ]
        enhanced_weights = [
            self.ENHANCED_SSIM_WEIGHT,
            self.ENHANCED_MS_SSIM_WEIGHT,
            self.ENHANCED_PSNR_WEIGHT,
            self.ENHANCED_MSE_WEIGHT,
            self.ENHANCED_FSIM_WEIGHT,
            self.ENHANCED_EDGE_WEIGHT,
            self.ENHANCED_GMSD_WEIGHT,
            self.ENHANCED_CHIST_WEIGHT,
            self.ENHANCED_SHARPNESS_WEIGHT,
            self.ENHANCED_TEXTURE_WEIGHT,
            self.ENHANCED_TEMPORAL_WEIGHT,
            self.ENHANCED_LPIPS_WEIGHT,
            self.ENHANCED_SSIMULACRA2_WEIGHT,
        ]
        all_weights = legacy_weights + enhanced_weights
        if any(w < 0 for w in all_weights):
            raise ValueError(
                f"All weights must be non-negative, got negatives in {all_weights}"
            )

        # Note: Frame alignment is always content-based (most robust approach)

        # Validate SSIM mode
        valid_modes = {"fast", "optimized", "full", "comprehensive"}
        if self.SSIM_MODE not in valid_modes:
            raise ValueError(f"Invalid SSIM mode: {self.SSIM_MODE}")

        # Validate frame limit is reasonable
        if self.SSIM_MAX_FRAMES <= 0 or self.SSIM_MAX_FRAMES > 1000:
            raise ValueError(
                f"SSIM_MAX_FRAMES must be between 1 and 1000, got {self.SSIM_MAX_FRAMES}"
            )

        # Validate PSNR max dB
        if self.PSNR_MAX_DB <= 0:
            raise ValueError("PSNR_MAX_DB must be positive")

        # Validate Canny thresholds
        if self.EDGE_CANNY_THRESHOLD1 <= 0 or self.EDGE_CANNY_THRESHOLD2 <= 0:
            raise ValueError("Canny thresholds must be positive")
        if self.EDGE_CANNY_THRESHOLD1 >= self.EDGE_CANNY_THRESHOLD2:
            raise ValueError("EDGE_CANNY_THRESHOLD1 must be < EDGE_CANNY_THRESHOLD2")

        # Set default positional metrics if not provided
        if self.POSITIONAL_METRICS is None:
            self.POSITIONAL_METRICS = ["ssim", "mse", "fsim", "chist"]


# Frame cache configuration (added for performance optimization)
# EXPERIMENTAL: Caching disabled by default due to performance degradation found in Phase 4.3 benchmarking
ENABLE_EXPERIMENTAL_CACHING = False

FRAME_CACHE = {
    "enabled": ENABLE_EXPERIMENTAL_CACHING,  # Enable frame caching
    "memory_limit_mb": 500,  # Maximum memory usage for in-memory cache
    "disk_path": None,  # Path to disk cache (None for default ~/.giflab_cache)
    "disk_limit_mb": 2000,  # Maximum disk cache size
    "ttl_seconds": 86400,  # Time-to-live for cache entries (24 hours)
    # Resized frame cache configuration
    "resize_cache_enabled": True,  # Enable resized frame caching
    "resize_cache_memory_mb": 200,  # Memory limit for resized frame cache
    "resize_cache_ttl_seconds": 3600,  # TTL for resized frames (1 hour)
    "enable_buffer_pooling": True,  # Enable memory buffer pooling for efficiency
}

# Frame sampling configuration for efficient validation
FRAME_SAMPLING = {
    "enabled": True,  # Enable frame sampling for large GIFs
    "min_frames_threshold": 30,  # Don't sample if fewer frames
    "default_strategy": "adaptive",  # Options: uniform, adaptive, progressive, scene_aware, full
    "confidence_level": 0.95,  # Target confidence level for statistical sampling
    # Strategy-specific configuration
    "uniform": {
        "sampling_rate": 0.3,  # Sample 30% of frames uniformly
    },
    "adaptive": {
        "base_rate": 0.2,  # Base sampling rate for low-motion areas
        "motion_threshold": 0.1,  # Threshold for detecting significant motion
        "max_rate": 0.8,  # Maximum sampling rate for high-motion areas
    },
    "progressive": {
        "initial_rate": 0.1,  # Initial sampling rate
        "increment_rate": 0.1,  # Rate increase per iteration
        "max_iterations": 5,  # Maximum sampling iterations
        "target_ci_width": 0.1,  # Target confidence interval width
    },
    "scene_aware": {
        "scene_threshold": 0.3,  # Threshold for scene change detection
        "boundary_window": 2,  # Frames to sample around scene boundaries
        "min_scene_samples": 3,  # Minimum samples per detected scene
        "base_rate": 0.3,  # Base sampling rate within scenes
    },
    "verbose": False,  # Enable verbose logging for sampling operations
}

# Validation cache configuration for caching metric results
VALIDATION_CACHE = {
    "enabled": True,  # Enable validation result caching
    "memory_limit_mb": 100,  # Maximum memory usage for in-memory cache
    "disk_path": None,  # Path to disk cache (None for default ~/.giflab_cache/validation_cache.db)
    "disk_limit_mb": 1000,  # Maximum disk cache size
    "ttl_seconds": 172800,  # Time-to-live for cache entries (48 hours)
    "cache_ssim": True,  # Cache SSIM calculations
    "cache_ms_ssim": True,  # Cache MS-SSIM calculations
    "cache_lpips": True,  # Cache LPIPS calculations
    "cache_gradient_color": True,  # Cache gradient color metrics
    "cache_ssimulacra2": True,  # Cache SSIMulacra2 metrics
}

# Performance monitoring configuration for optimization systems
MONITORING = {
    "enabled": True,  # Enable performance monitoring
    "backend": "sqlite",  # Backend type: "memory", "sqlite", or "statsd"
    "buffer_size": 10000,  # In-memory ring buffer size for recent metrics
    "flush_interval": 10.0,  # Seconds between automatic metric flushes
    "sampling_rate": 1.0,  # Fraction of metrics to collect (1.0 = all)
    # SQLite backend configuration
    "sqlite": {
        "db_path": None,  # Path to metrics database (None for ~/.giflab_cache/metrics.db)
        "retention_days": 7.0,  # Days to retain metrics before cleanup
        "max_size_mb": 100.0,  # Maximum database size in MB
    },
    # StatsD backend configuration (optional, requires statsd library)
    "statsd": {
        "host": "localhost",  # StatsD server host
        "port": 8125,  # StatsD server port
        "prefix": "giflab",  # Metric name prefix
    },
    # Alert thresholds for monitoring
    "alerts": {
        "cache_hit_rate_warning": 0.4,  # Warn if cache hit rate drops below 40%
        "cache_hit_rate_critical": 0.2,  # Critical if cache hit rate drops below 20%
        "memory_usage_warning": 0.8,  # Warn at 80% memory usage
        "memory_usage_critical": 0.95,  # Critical at 95% memory usage
        "eviction_rate_spike": 3.0,  # Alert if eviction rate > 3x normal
        "response_time_degradation": 1.5,  # Alert if response time > 1.5x baseline
    },
    # Per-system monitoring configuration
    "systems": {
        "frame_cache": True,  # Monitor FrameCache performance
        "validation_cache": True,  # Monitor ValidationCache performance
        "resize_cache": True,  # Monitor ResizedFrameCache performance
        "sampling": True,  # Monitor frame sampling performance
        "lazy_imports": True,  # Monitor lazy import patterns
        "metrics_calculation": True,  # Monitor core metrics calculation
        "memory_pressure": True,  # Monitor system memory pressure
    },
    # Memory pressure monitoring configuration
    "memory_pressure": {
        "enabled": True,  # Enable memory pressure monitoring
        "update_interval": 5.0,  # Seconds between memory checks
        "auto_eviction": True,  # Enable automatic cache eviction under pressure
        "eviction_policy": "conservative",  # Eviction policy: "conservative", "aggressive"
        # Memory pressure thresholds (fractions of system memory)
        "thresholds": {
            "warning": 0.70,  # Warning level at 70% system memory
            "critical": 0.80,  # Critical level at 80% system memory
            "emergency": 0.95,  # Emergency level at 95% system memory
        },
        # Eviction targets as fraction of process memory to free
        "eviction_targets": {
            "warning": 0.15,  # Free 15% of process memory at warning
            "critical": 0.30,  # Free 30% of process memory at critical
            "emergency": 0.50,  # Free 50% of process memory at emergency
        },
        # Hysteresis to prevent eviction thrashing
        "hysteresis": {
            "enable_delta": 0.05,  # Re-enable pressure monitoring 5% below threshold
            "eviction_cooldown": 30.0,  # Minimum seconds between evictions
        },
        # Integration with existing cache limits
        "respect_cache_limits": True,  # Don't evict below configured cache minimums
        "fallback_when_unavailable": True,  # Graceful degradation without psutil
    },
    # Phase 7: Performance regression monitoring configuration
    "performance": {
        "enabled": False,  # Disabled by default, enable with GIFLAB_ENABLE_PERFORMANCE_MONITORING=true
        "regression_threshold": 0.10,  # Alert on 10%+ performance regression
        "confidence_level": 0.95,  # 95% confidence level for statistical analysis
        "monitoring_interval": 3600,  # Seconds between continuous monitoring checks (1 hour)
        "max_history_days": 30,  # Days to retain performance history
        "baseline_min_samples": 3,  # Minimum samples needed to establish baseline
        # Scenarios for continuous monitoring (lightweight subset)
        "continuous_scenarios": [
            "continuous_small_gif",  # Small GIF for regular monitoring
        ],
        # Performance data storage
        "data_directory": "performance_data",  # Directory for baselines and history
        "baseline_file": "performance_baselines.json",  # Baseline storage file
        "history_directory": "performance_history",  # History storage directory
        # Alert configuration
        "alert_on_regression": True,  # Send alerts via AlertManager
        "alert_severities": {
            "minor_regression": 0.10,  # 10%+ regression = INFO
            "major_regression": 0.25,  # 25%+ regression = WARNING
            "critical_regression": 0.50,  # 50%+ regression = CRITICAL
        },
        # CI/CD integration
        "fail_ci_on_regression": True,  # Fail CI/CD on significant regression
        "ci_regression_threshold": 0.15,  # 15%+ regression fails CI/CD
        "ci_scenarios": [
            "small_gif_basic",
            "medium_gif_comprehensive",
        ],  # Scenarios for CI
        # Phase 6 optimization validation
        "validate_phase6": True,  # Monitor Phase 6 optimization effectiveness
        "phase6_baseline_speedup": 5.04,  # Expected Phase 6 speedup (5.04x)
        "phase6_regression_alert": 0.20,  # Alert if Phase 6 speedup drops 20%+
    },
    "verbose": False,  # Enable verbose monitoring logs
}


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""

    RAW_DIR: Path = Path("data/raw")
    RENDERS_DIR: Path = Path("data/renders")
    CSV_DIR: Path = Path("data/csv")
    BAD_GIFS_DIR: Path = Path("data/bad_gifs")
    TMP_DIR: Path = Path("data/tmp")
    SEED_DIR: Path = Path("seed")
    LOGS_DIR: Path = Path("logs")

    def __post_init__(self) -> None:
        """Validate path configuration after initialization."""
        # Import here to avoid circular imports
        from .input_validation import ValidationError, validate_config_paths

        try:
            # Convert dataclass to dict for validation
            config_dict = {
                "RAW_DIR": self.RAW_DIR,
                "RENDERS_DIR": self.RENDERS_DIR,
                "CSV_DIR": self.CSV_DIR,
                "BAD_GIFS_DIR": self.BAD_GIFS_DIR,
                "TMP_DIR": self.TMP_DIR,
                "SEED_DIR": self.SEED_DIR,
                "LOGS_DIR": self.LOGS_DIR,
            }

            # Validate paths for security
            validate_config_paths(config_dict)

        except ValidationError as e:
            raise ValueError(f"Invalid path configuration: {e}") from e


@dataclass
class EngineConfig:
    """Configuration for compression engine paths with environment variable overrides."""

    # Path to the gifsicle executable.
    # On macOS/Linux, "gifsicle" should work if installed via package manager
    # On Windows, you might need to provide the full path to the .exe
    # e.g., "C:/Program Files/gifsicle/gifsicle.exe"
    # Override with: GIFLAB_GIFSICLE_PATH
    GIFSICLE_PATH: str = "gifsicle"

    # Path to the animately executable.
    # Set this to the location of your animately binary.
    # Override with: GIFLAB_ANIMATELY_PATH
    ANIMATELY_PATH: str = (
        "/Users/lachlants/repos/animately/animately-engine-releases/launcher"
    )

    # Path to ImageMagick executable (magick or convert).
    # On most systems, "magick" should work (ImageMagick 7.x)
    # On older systems or specific setups, may need "convert"
    # Override with: GIFLAB_IMAGEMAGICK_PATH
    IMAGEMAGICK_PATH: str = "magick"

    # Path to FFmpeg executable.
    # Usually "ffmpeg" works if installed via package manager
    # Override with: GIFLAB_FFMPEG_PATH
    FFMPEG_PATH: str = "ffmpeg"

    # Path to FFprobe executable (companion to FFmpeg).
    # Usually "ffprobe" works if FFmpeg is properly installed
    # Override with: GIFLAB_FFPROBE_PATH
    FFPROBE_PATH: str = "ffprobe"

    # Path to gifski executable.
    # Install from: https://gif.ski/
    # Override with: GIFLAB_GIFSKI_PATH
    GIFSKI_PATH: str = "gifski"

    def __post_init__(self) -> None:
        """Apply environment variable overrides after initialization."""
        import os

        # Environment variable mapping
        env_overrides = {
            "GIFSICLE_PATH": "GIFLAB_GIFSICLE_PATH",
            "ANIMATELY_PATH": "GIFLAB_ANIMATELY_PATH",
            "IMAGEMAGICK_PATH": "GIFLAB_IMAGEMAGICK_PATH",
            "FFMPEG_PATH": "GIFLAB_FFMPEG_PATH",
            "FFPROBE_PATH": "GIFLAB_FFPROBE_PATH",
            "GIFSKI_PATH": "GIFLAB_GIFSKI_PATH",
        }

        # Apply overrides from environment variables
        for attr_name, env_var_name in env_overrides.items():
            env_value = os.getenv(env_var_name)
            if env_value:
                setattr(self, attr_name, env_value)


@dataclass
class ValidationConfig:
    """Configuration for wrapper output validation system."""

    # Enable/disable validation
    ENABLE_WRAPPER_VALIDATION: bool = True

    # Frame validation tolerances
    FRAME_RATIO_TOLERANCE: float = 0.1  # 10% tolerance for frame reduction ratios
    MIN_FRAMES_REQUIRED: int = 1  # Minimum frames in output

    # Color validation settings
    COLOR_COUNT_TOLERANCE: int = 5  # Allow up to 5 extra colors due to encoding
    MIN_COLOR_REDUCTION_PERCENT: float = 0.1  # Require at least 10% color reduction

    # Timing validation tolerances
    FPS_TOLERANCE: float = 0.2  # 20% tolerance for FPS changes
    MIN_FPS: float = 0.1  # Minimum valid FPS
    MAX_FPS: float = 60.0  # Maximum reasonable FPS

    # Advanced timing validation settings
    TIMING_VALIDATION_ENABLED: bool = True  # Enable comprehensive timing validation
    TIMING_VALIDATION_ALERT_ON_FAILURE: bool = (
        True  # Alert when timing validation fails
    )
    TIMING_GRID_MS: int = 10  # Timing grid size in milliseconds
    TIMING_MAX_DRIFT_MS: int = 100  # Maximum acceptable timing drift
    TIMING_DURATION_DIFF_THRESHOLD: int = 50  # Maximum total duration difference
    TIMING_ALIGNMENT_ACCURACY_MIN: float = 0.9  # Minimum alignment accuracy (0.0-1.0)

    # Performance and error handling
    VALIDATION_TIMEOUT_MS: int = 5000  # 5 second timeout per validation
    FAIL_ON_VALIDATION_ERROR: bool = False  # Don't break pipelines on validation errors
    LOG_VALIDATION_FAILURES: bool = True  # Log validation failures for debugging

    # File integrity checks
    MIN_FILE_SIZE_BYTES: int = 100  # Minimum valid GIF size
    MAX_FILE_SIZE_MB: float = 100.0  # Maximum reasonable output size

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate tolerances are positive
        if self.FRAME_RATIO_TOLERANCE <= 0 or self.FRAME_RATIO_TOLERANCE > 1:
            raise ValueError(
                f"FRAME_RATIO_TOLERANCE must be between 0 and 1, got {self.FRAME_RATIO_TOLERANCE}"
            )

        if self.FPS_TOLERANCE <= 0 or self.FPS_TOLERANCE > 1:
            raise ValueError(
                f"FPS_TOLERANCE must be between 0 and 1, got {self.FPS_TOLERANCE}"
            )

        if self.MIN_COLOR_REDUCTION_PERCENT < 0 or self.MIN_COLOR_REDUCTION_PERCENT > 1:
            raise ValueError(
                f"MIN_COLOR_REDUCTION_PERCENT must be between 0 and 1, got {self.MIN_COLOR_REDUCTION_PERCENT}"
            )

        # Validate frame requirements
        if self.MIN_FRAMES_REQUIRED < 1:
            raise ValueError(
                f"MIN_FRAMES_REQUIRED must be at least 1, got {self.MIN_FRAMES_REQUIRED}"
            )

        # Validate FPS bounds
        if self.MIN_FPS <= 0:
            raise ValueError(f"MIN_FPS must be positive, got {self.MIN_FPS}")
        if self.MAX_FPS <= self.MIN_FPS:
            raise ValueError(
                f"MAX_FPS must be greater than MIN_FPS, got MAX_FPS={self.MAX_FPS}, MIN_FPS={self.MIN_FPS}"
            )

        # Validate file size limits
        if self.MIN_FILE_SIZE_BYTES <= 0:
            raise ValueError(
                f"MIN_FILE_SIZE_BYTES must be positive, got {self.MIN_FILE_SIZE_BYTES}"
            )
        if self.MAX_FILE_SIZE_MB <= 0:
            raise ValueError(
                f"MAX_FILE_SIZE_MB must be positive, got {self.MAX_FILE_SIZE_MB}"
            )

        # Validate timeout
        if self.VALIDATION_TIMEOUT_MS <= 0:
            raise ValueError(
                f"VALIDATION_TIMEOUT_MS must be positive, got {self.VALIDATION_TIMEOUT_MS}"
            )

        # Validate timing validation settings
        if self.TIMING_GRID_MS <= 0:
            raise ValueError(
                f"TIMING_GRID_MS must be positive, got {self.TIMING_GRID_MS}"
            )
        if self.TIMING_MAX_DRIFT_MS < 0:
            raise ValueError(
                f"TIMING_MAX_DRIFT_MS must be non-negative, got {self.TIMING_MAX_DRIFT_MS}"
            )
        if self.TIMING_DURATION_DIFF_THRESHOLD < 0:
            raise ValueError(
                f"TIMING_DURATION_DIFF_THRESHOLD must be non-negative, got {self.TIMING_DURATION_DIFF_THRESHOLD}"
            )
        if (
            self.TIMING_ALIGNMENT_ACCURACY_MIN < 0
            or self.TIMING_ALIGNMENT_ACCURACY_MIN > 1
        ):
            raise ValueError(
                f"TIMING_ALIGNMENT_ACCURACY_MIN must be between 0 and 1, got {self.TIMING_ALIGNMENT_ACCURACY_MIN}"
            )


# Default configuration instances
DEFAULT_COMPRESSION_CONFIG = CompressionConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_ENGINE_CONFIG = EngineConfig()
DEFAULT_VALIDATION_CONFIG = ValidationConfig()
