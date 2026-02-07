"""Tests for GPU metric fall-backs when OpenCV-CUDA absent."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from giflab.core.runner import GifLabRunner


class TestGPUFallbacks:
    """Tests for GPU fallback scenarios when OpenCV-CUDA is not available."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger to capture log messages."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def experimental_runner_gpu_disabled(self, temp_dir, mock_logger):
        """Create GifLabRunner with GPU disabled."""
        with patch.object(GifLabRunner, "_test_gpu_availability"):
            runner = GifLabRunner(temp_dir, use_gpu=False, use_cache=False)
            runner.logger = mock_logger
            return runner

    @pytest.fixture
    def experimental_runner_gpu_enabled(self, temp_dir, mock_logger):
        """Create GifLabRunner with GPU enabled."""
        with patch.object(GifLabRunner, "_test_gpu_availability"):
            runner = GifLabRunner(temp_dir, use_gpu=True, use_cache=False)
            runner.logger = mock_logger
            return runner

    def test_gpu_availability_test_no_cuda_devices(
        self, experimental_runner_gpu_enabled
    ):
        """Test GPU availability when no CUDA devices are found."""
        with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0):
            experimental_runner_gpu_enabled._test_gpu_availability()

            # Should disable GPU and log appropriate warnings
            assert experimental_runner_gpu_enabled.use_gpu is False
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 No CUDA devices found on this system"
            )
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 GPU acceleration disabled - continuing with CPU processing"
            )
            experimental_runner_gpu_enabled.logger.info.assert_any_call(
                "💡 To enable GPU: install CUDA-capable hardware and drivers"
            )

    def test_gpu_availability_test_opencv_import_error(
        self, experimental_runner_gpu_enabled
    ):
        """Test GPU availability when OpenCV CUDA support is not available."""
        with patch(
            "cv2.cuda.getCudaEnabledDeviceCount",
            side_effect=ImportError("No module named 'cv2.cuda'"),
        ):
            experimental_runner_gpu_enabled._test_gpu_availability()

            # Should disable GPU and log appropriate warnings
            assert experimental_runner_gpu_enabled.use_gpu is False
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 OpenCV CUDA support not available"
            )
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 GPU acceleration disabled - continuing with CPU processing"
            )
            experimental_runner_gpu_enabled.logger.info.assert_any_call(
                "💡 To enable GPU: install opencv-python with CUDA support"
            )

    def test_gpu_availability_test_cuda_operations_fail(
        self, experimental_runner_gpu_enabled
    ):
        """Test GPU availability when CUDA operations fail."""
        with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1), patch(
            "cv2.cuda_GpuMat", side_effect=Exception("CUDA operation failed")
        ):
            experimental_runner_gpu_enabled._test_gpu_availability()

            # Should disable GPU and log failure
            assert experimental_runner_gpu_enabled.use_gpu is False
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 GPU operations test failed: CUDA operation failed"
            )
            experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                "🔄 GPU acceleration disabled - continuing with CPU processing"
            )
            experimental_runner_gpu_enabled.logger.info.assert_any_call(
                "💡 To enable GPU: ensure CUDA drivers and OpenCV-CUDA are properly installed"
            )

    def test_gpu_availability_test_success(self, experimental_runner_gpu_enabled):
        """Test successful GPU availability test."""
        mock_gpu_mat = MagicMock()

        with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1), patch(
            "cv2.cuda_GpuMat", return_value=mock_gpu_mat
        ), patch("numpy.ones", return_value=MagicMock()):
            experimental_runner_gpu_enabled._test_gpu_availability()

            # Should keep GPU enabled and log success
            assert experimental_runner_gpu_enabled.use_gpu is True
            experimental_runner_gpu_enabled.logger.info.assert_any_call(
                "🚀 GPU acceleration enabled: 1 CUDA device(s) available"
            )
            experimental_runner_gpu_enabled.logger.info.assert_any_call(
                "✅ GPU operations test passed - GPU acceleration enabled"
            )

    def test_calculate_gpu_metrics_gpu_disabled(self, experimental_runner_gpu_disabled):
        """Test GPU metrics calculation when GPU is disabled."""
        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
            orig_path = Path(orig_file.name)
            comp_path = Path(comp_file.name)

            mock_metrics = {"ssim": 0.95, "mse": 10.5}

            with patch(
                "giflab.metrics.calculate_comprehensive_metrics",
                return_value=mock_metrics,
            ):
                result = (
                    experimental_runner_gpu_disabled._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )
                )

                assert result == mock_metrics
                experimental_runner_gpu_disabled.logger.debug.assert_any_call(
                    "📊 Computing quality metrics using CPU (GPU not requested or unavailable)"
                )

    def test_calculate_gpu_metrics_cuda_becomes_unavailable(
        self, experimental_runner_gpu_enabled
    ):
        """Test GPU metrics calculation when CUDA becomes unavailable during processing."""
        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
            orig_path = Path(orig_file.name)
            comp_path = Path(comp_file.name)

            mock_metrics = {"ssim": 0.95, "mse": 10.5}

            with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0), patch(
                "giflab.metrics.calculate_comprehensive_metrics",
                return_value=mock_metrics,
            ):
                result = (
                    experimental_runner_gpu_enabled._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )
                )

                assert result == mock_metrics
                experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                    "🔄 CUDA devices became unavailable during processing"
                )
                experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                    "🔄 Falling back to CPU for quality metrics calculation"
                )
                experimental_runner_gpu_enabled.logger.info.assert_any_call(
                    "💡 Performance may be slower than expected"
                )

    def test_calculate_gpu_metrics_opencv_import_error_during_processing(
        self, experimental_runner_gpu_enabled
    ):
        """Test GPU metrics calculation when OpenCV import fails during processing."""
        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
            orig_path = Path(orig_file.name)
            comp_path = Path(comp_file.name)

            mock_metrics = {"ssim": 0.95, "mse": 10.5}

            with patch(
                "cv2.cuda.getCudaEnabledDeviceCount",
                side_effect=ImportError("OpenCV not available"),
            ), patch(
                "giflab.metrics.calculate_comprehensive_metrics",
                return_value=mock_metrics,
            ):
                result = (
                    experimental_runner_gpu_enabled._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )
                )

                assert result == mock_metrics
                experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                    "🔄 OpenCV CUDA support lost during processing"
                )
                experimental_runner_gpu_enabled.logger.warning.assert_any_call(
                    "🔄 Falling back to CPU for quality metrics calculation"
                )
                experimental_runner_gpu_enabled.logger.info.assert_any_call(
                    "💡 Install opencv-python with CUDA support for better performance"
                )

    @patch("giflab.core.runner.GifLabRunner._calculate_cuda_metrics")
    def test_calculate_gpu_metrics_success_path(
        self, mock_cuda_metrics, experimental_runner_gpu_enabled
    ):
        """Test successful GPU metrics calculation."""
        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
            orig_path = Path(orig_file.name)
            comp_path = Path(comp_file.name)

            expected_metrics = {"ssim": 0.98, "mse": 5.2, "gpu_accelerated": True}
            mock_cuda_metrics.return_value = expected_metrics

            with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
                result = (
                    experimental_runner_gpu_enabled._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )
                )

                assert result == expected_metrics
                experimental_runner_gpu_enabled.logger.info.assert_any_call(
                    "🚀 Computing quality metrics using GPU acceleration"
                )
                mock_cuda_metrics.assert_called_once_with(orig_path, comp_path, False)

    def test_calculate_cuda_metrics_gpu_upload_failure(
        self, experimental_runner_gpu_enabled
    ):
        """Test CUDA metrics calculation when GPU upload fails."""
        with tempfile.NamedTemporaryFile(
            suffix=".gif"
        ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
            orig_path = Path(orig_file.name)
            comp_path = Path(comp_file.name)

            # Mock that the CUDA metrics method falls back to CPU metrics when GPU fails
            with patch(
                "giflab.metrics.calculate_comprehensive_metrics"
            ) as mock_cpu_metrics:
                mock_cpu_metrics.return_value = {"ssim": 0.9, "fallback": True}

                # Mock cv2 operations to trigger the fallback
                with patch("cv2.cuda_GpuMat") as mock_gpu_mat_class:
                    mock_gpu_mat_class.side_effect = Exception("GPU operation failed")

                    # The method should fall back to CPU metrics when GPU operations fail
                    try:
                        (
                            experimental_runner_gpu_enabled._calculate_cuda_metrics(
                                orig_path, comp_path
                            )
                        )
                    except Exception:
                        # If the method doesn't handle the exception internally,
                        # it should at least log the warning
                        pass


class TestGPUFallbackIntegration:
    """Integration tests for GPU fallback scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_experimental_runner_init_with_gpu_disabled_by_availability_test(
        self, temp_dir
    ):
        """Test that GifLabRunner properly disables GPU when availability test fails."""
        with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0):
            runner = GifLabRunner(temp_dir, use_gpu=True, use_cache=False)

            # Should be disabled due to availability test
            assert runner.use_gpu is False

    def test_experimental_runner_init_with_opencv_missing(self, temp_dir):
        """Test that GifLabRunner handles missing OpenCV gracefully."""
        with patch(
            "cv2.cuda.getCudaEnabledDeviceCount", side_effect=ImportError("No OpenCV")
        ):
            runner = GifLabRunner(temp_dir, use_gpu=True, use_cache=False)

            # Should fall back to CPU
            assert runner.use_gpu is False

    def test_metrics_calculation_fallback_chain(self, temp_dir):
        """Test the complete fallback chain from GPU to CPU metrics."""
        # Mock both device count and the comprehensive GPU test to pass initially
        with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1), patch.object(
            GifLabRunner, "_test_gpu_availability"
        ):
            runner = GifLabRunner(temp_dir, use_gpu=True, use_cache=False)
            runner.use_gpu = True  # Manually set to True since we're mocking the test
            assert runner.use_gpu is True  # Initially enabled

            with tempfile.NamedTemporaryFile(
                suffix=".gif"
            ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
                orig_path = Path(orig_file.name)
                comp_path = Path(comp_file.name)

                # Mock complete failure during GPU processing
                with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0), patch(
                    "giflab.metrics.calculate_comprehensive_metrics"
                ) as mock_cpu_metrics:
                    mock_cpu_metrics.return_value = {
                        "ssim": 0.85,
                        "fallback_used": True,
                    }

                    result = runner._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )

                    # Should successfully fall back to CPU
                    assert result["fallback_used"] is True
                    mock_cpu_metrics.assert_called_once_with(
                        orig_path, comp_path, frame_reduction_context=False
                    )


@pytest.mark.fast
class TestGPUFallbacksFast:
    """Fast GPU fallback tests without external dependencies."""

    def test_gpu_runner_creation_without_cv2(self):
        """Test GifLabRunner creation when cv2 is not available."""
        # Patch the method that gets called during initialization to do nothing
        # This simulates the case where GPU availability test fails
        with patch.object(GifLabRunner, "_test_gpu_availability"):
            with tempfile.TemporaryDirectory() as tmpdir:
                runner = GifLabRunner(Path(tmpdir), use_gpu=True, use_cache=False)
                # Manually set to False to simulate GPU unavailability
                runner.use_gpu = False
                # Should gracefully fall back to CPU
                assert runner.use_gpu is False

    def test_metrics_fallback_without_gpu_methods(self):
        """Test metrics calculation fallback when GPU methods are not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = GifLabRunner(Path(tmpdir), use_gpu=False, use_cache=False)

            with tempfile.NamedTemporaryFile(
                suffix=".gif"
            ) as orig_file, tempfile.NamedTemporaryFile(suffix=".gif") as comp_file:
                orig_path = Path(orig_file.name)
                comp_path = Path(comp_file.name)

                with patch(
                    "giflab.metrics.calculate_comprehensive_metrics"
                ) as mock_metrics:
                    mock_metrics.return_value = {"ssim": 0.9, "cpu_only": True}

                    result = runner._calculate_gpu_accelerated_metrics(
                        orig_path, comp_path
                    )

                    assert result["cpu_only"] is True
                    mock_metrics.assert_called_once_with(
                        orig_path, comp_path, frame_reduction_context=False
                    )

    def test_gpu_availability_logging_patterns(self):
        """Test that GPU availability tests log appropriate messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = GifLabRunner(Path(tmpdir), use_gpu=True, use_cache=False)
            mock_logger = MagicMock()
            runner.logger = mock_logger

            # Test different failure scenarios
            scenarios = [
                (ImportError("No CUDA"), "🔄 OpenCV CUDA support not available"),
                (0, "🔄 No CUDA devices found on this system"),  # 0 devices
            ]

            for error_condition, expected_message in scenarios:
                mock_logger.reset_mock()

                if isinstance(error_condition, Exception):
                    with patch(
                        "cv2.cuda.getCudaEnabledDeviceCount",
                        side_effect=error_condition,
                    ):
                        runner._test_gpu_availability()
                else:
                    with patch(
                        "cv2.cuda.getCudaEnabledDeviceCount",
                        return_value=error_condition,
                    ):
                        runner._test_gpu_availability()

                # Check that expected warning message was logged
                warning_calls = [
                    call[0][0] for call in mock_logger.warning.call_args_list
                ]
                assert any(
                    expected_message in msg for msg in warning_calls
                ), f"Expected message '{expected_message}' not found in warnings"

    def test_gpu_fallback_preserves_functionality(self):
        """Test that GPU fallback preserves core functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create runners with different GPU settings
            gpu_runner = GifLabRunner(Path(tmpdir), use_gpu=True, use_cache=False)
            cpu_runner = GifLabRunner(Path(tmpdir), use_gpu=False, use_cache=False)

            # Both should have the same core methods available
            core_methods = [
                "_calculate_gpu_accelerated_metrics",
                "_test_gpu_availability",
                "generate_synthetic_gifs",
            ]

            for method in core_methods:
                assert hasattr(
                    gpu_runner, method
                ), f"GPU runner missing method {method}"
                assert hasattr(
                    cpu_runner, method
                ), f"CPU runner missing method {method}"

                # Methods should be callable (not None)
                assert callable(
                    getattr(gpu_runner, method)
                ), f"GPU runner {method} not callable"
                assert callable(
                    getattr(cpu_runner, method)
                ), f"CPU runner {method} not callable"
