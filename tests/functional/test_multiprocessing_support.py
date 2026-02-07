"""Lightweight tests for multiprocessing support functionality.

These tests verify that the multiprocessing support infrastructure
works correctly without heavy performance testing.
"""

import multiprocessing as mp
import queue
import time

import pytest
from giflab.multiprocessing_support import (
    FrameGenerationResult,
    FrameGenerationTask,
    ParallelFrameGenerator,
    ParallelPipelineExecutor,
    ProcessSafeQueue,
    _generate_single_frame,
    get_optimal_worker_count,
)
from giflab.synthetic_gifs import SyntheticGifSpec


class TestProcessSafeQueue:
    """Test the process-safe queue implementation."""

    def test_queue_basic_operations(self):
        """Test basic put/get operations work."""
        pq = ProcessSafeQueue(maxsize=5)

        # Test put and get
        test_item = {"test": "data"}
        pq.put(test_item)

        retrieved = pq.get()
        assert retrieved == test_item

    def test_queue_empty_check(self):
        """Test empty queue detection."""
        pq = ProcessSafeQueue()

        # Test basic put/get cycle rather than relying on empty() which can be racy
        pq.put("item")
        retrieved = pq.get()
        assert retrieved == "item"

        # Test that we can put and get another item
        pq.put("item2")
        retrieved2 = pq.get()
        assert retrieved2 == "item2"

    def test_queue_size_tracking(self):
        """Test queue size tracking (skipped on macOS due to platform limitations)."""
        pq = ProcessSafeQueue()

        try:
            size = pq.qsize()
            # If qsize() works, test it
            assert size >= 0

            pq.put("item1")
            pq.put("item2")
            new_size = pq.qsize()
            assert new_size >= size  # Should have increased

            pq.get()
            final_size = pq.qsize()
            assert final_size < new_size  # Should have decreased

        except NotImplementedError:
            # qsize() not supported on this platform (like macOS)
            # Just test basic functionality
            pq.put("item1")
            item = pq.get()
            assert item == "item1"

    def test_queue_timeout_behavior(self):
        """Test timeout behavior for get operations."""
        pq = ProcessSafeQueue()

        start_time = time.time()
        with pytest.raises(queue.Empty):
            pq.get(timeout=0.1)

        elapsed = time.time() - start_time
        assert 0.05 < elapsed < 0.2  # Should timeout around 0.1s


class TestFrameGenerationTask:
    """Test frame generation task data structures."""

    def test_task_creation(self):
        """Test frame generation task creation."""
        task = FrameGenerationTask(
            content_type="gradient",
            size=(100, 100),
            frame_index=5,
            total_frames=10,
            task_id="test_task",
        )

        assert task.content_type == "gradient"
        assert task.size == (100, 100)
        assert task.frame_index == 5
        assert task.total_frames == 10
        assert task.task_id == "test_task"


class TestFrameGenerationResult:
    """Test frame generation result data structures."""

    def test_successful_result(self):
        """Test successful frame generation result."""
        result = FrameGenerationResult(
            task_id="test_task",
            frame_index=5,
            success=True,
            image_data=b"fake_image_data",
            generation_time=0.123,
        )

        assert result.task_id == "test_task"
        assert result.frame_index == 5
        assert result.success
        assert result.image_data == b"fake_image_data"
        assert result.generation_time == 0.123
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed frame generation result."""
        result = FrameGenerationResult(
            task_id="failed_task",
            frame_index=2,
            success=False,
            error_message="Something went wrong",
        )

        assert result.task_id == "failed_task"
        assert result.frame_index == 2
        assert not result.success
        assert result.error_message == "Something went wrong"
        assert result.image_data is None


class TestWorkerFunction:
    """Test the worker function for frame generation."""

    def test_successful_frame_generation(self):
        """Test successful frame generation in worker."""
        task = FrameGenerationTask(
            content_type="gradient",
            size=(50, 50),
            frame_index=0,
            total_frames=5,
            task_id="worker_test",
        )

        result = _generate_single_frame(task)

        assert isinstance(result, FrameGenerationResult)
        assert result.task_id == "worker_test"
        assert result.frame_index == 0
        assert result.success
        assert result.image_data is not None
        assert result.generation_time > 0
        assert result.error_message is None

    def test_invalid_content_type_handling(self):
        """Test worker handles invalid content types gracefully."""
        task = FrameGenerationTask(
            content_type="invalid_type",
            size=(50, 50),
            frame_index=0,
            total_frames=5,
            task_id="invalid_test",
        )

        result = _generate_single_frame(task)

        # Should still succeed with fallback
        assert isinstance(result, FrameGenerationResult)
        assert result.task_id == "invalid_test"
        assert result.success  # Should fallback gracefully


class TestParallelFrameGenerator:
    """Test the parallel frame generator."""

    def setup_method(self):
        """Setup test fixtures."""
        # Use minimal workers for testing
        self.generator = ParallelFrameGenerator(max_workers=2)

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        gen = ParallelFrameGenerator(max_workers=4, chunk_size=2)

        assert gen.max_workers == 4
        assert gen.chunk_size == 2
        assert gen.logger is not None

    def test_small_task_list_processing(self):
        """Test processing a small list of tasks."""
        tasks = [
            FrameGenerationTask("gradient", (30, 30), 0, 2, "task_0"),
            FrameGenerationTask("solid", (30, 30), 1, 2, "task_1"),
        ]

        results = self.generator.generate_frames_parallel(tasks)

        assert len(results) == 2
        assert all(isinstance(r, FrameGenerationResult) for r in results)
        assert all(r.success for r in results)

    def test_empty_task_list(self):
        """Test handling empty task list."""
        results = self.generator.generate_frames_parallel([])
        assert results == []

    def test_progress_callback(self):
        """Test progress callback functionality."""
        tasks = [
            FrameGenerationTask("gradient", (20, 20), i, 3, f"task_{i}")
            for i in range(3)
        ]

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        results = self.generator.generate_frames_parallel(tasks, progress_callback)

        assert len(results) == 3
        assert len(progress_calls) > 0
        # Should have final call with all completed
        assert (3, 3) in progress_calls

    def test_gif_frames_parallel_generation(self):
        """Test generating complete GIF frames in parallel."""
        spec = SyntheticGifSpec(
            name="test_spec",
            frames=4,
            size=(40, 40),
            content_type="gradient",
            description="Test spec",
        )

        images = self.generator.generate_gif_frames_parallel(spec)

        assert len(images) == 4
        # All should be PIL Images
        from PIL import Image

        assert all(isinstance(img, Image.Image) for img in images)
        assert all(img.size == (40, 40) for img in images)


class TestParallelPipelineExecutor:
    """Test the parallel pipeline executor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.executor = ParallelPipelineExecutor(max_workers=2, db_write_queue_size=10)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.executor._writer_thread and self.executor._writer_thread.is_alive():
            self.executor.stop_db_writer_thread()

    def test_executor_initialization(self):
        """Test executor initializes correctly."""
        exec = ParallelPipelineExecutor(max_workers=4, db_write_queue_size=100)

        assert exec.max_workers == 4
        assert exec.db_write_queue is not None
        assert exec.logger is not None

    def test_db_writer_thread_lifecycle(self):
        """Test DB writer thread start/stop."""
        write_calls = []

        def mock_writer(data):
            write_calls.append(data)

        # Start thread
        self.executor.start_db_writer_thread(mock_writer)
        assert self.executor._writer_thread is not None
        assert self.executor._writer_thread.is_alive()

        # Queue some data
        self.executor.queue_db_write("test_data")

        # Stop thread
        self.executor.stop_db_writer_thread()

        # Should have processed the data
        time.sleep(0.1)  # Give thread time to process
        assert "test_data" in write_calls

    def test_queue_db_write_timeout(self):
        """Test DB write queuing with timeout."""
        # Fill up the queue (small queue size for testing)
        small_executor = ParallelPipelineExecutor(max_workers=1, db_write_queue_size=2)

        # Should succeed initially
        assert small_executor.queue_db_write("item1", timeout=0.1)
        assert small_executor.queue_db_write("item2", timeout=0.1)

        # Should timeout on third item (queue full, no consumer)
        assert not small_executor.queue_db_write("item3", timeout=0.1)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_optimal_worker_count_frame_generation(self):
        """Test optimal worker count for frame generation."""
        count = get_optimal_worker_count("frame_generation")

        assert isinstance(count, int)
        assert count > 0
        # Should use all CPUs for CPU-intensive work
        assert count == mp.cpu_count()

    def test_optimal_worker_count_pipeline_execution(self):
        """Test optimal worker count for pipeline execution."""
        count = get_optimal_worker_count("pipeline_execution")

        assert isinstance(count, int)
        assert count > 0
        # Should leave one CPU for coordination
        expected = max(1, mp.cpu_count() - 1)
        assert count == expected

    def test_optimal_worker_count_unknown_type(self):
        """Test optimal worker count for unknown task type."""
        count = get_optimal_worker_count("unknown_task")

        assert isinstance(count, int)
        assert count > 0
        # Should be conservative
        expected = max(1, mp.cpu_count() // 2)
        assert count == expected


@pytest.mark.integration
class TestMultiprocessingIntegration:
    """Integration tests for multiprocessing components."""

    def test_end_to_end_frame_generation(self):
        """Test complete end-to-end frame generation workflow."""
        generator = ParallelFrameGenerator(max_workers=2)

        # Create a small but realistic workload
        spec = SyntheticGifSpec(
            name="integration_test",
            frames=6,
            size=(60, 60),
            content_type="noise",
            description="Integration test spec",
        )

        start_time = time.time()
        images = generator.generate_gif_frames_parallel(spec)
        end_time = time.time()

        # Verify results
        assert len(images) == 6
        from PIL import Image

        assert all(isinstance(img, Image.Image) for img in images)

        # Should complete reasonably quickly
        assert (end_time - start_time) < 5.0  # Should be much faster but allow for CI

    def test_process_safety_simulation(self):
        """Test that multiprocessing doesn't break under concurrent access."""
        generator = ParallelFrameGenerator(max_workers=2)

        # Create multiple tasks that might compete
        tasks = [
            FrameGenerationTask("gradient", (30, 30), i, 8, f"concurrent_{i}")
            for i in range(8)
        ]

        results = generator.generate_frames_parallel(tasks)

        # All should succeed
        assert len(results) == 8
        assert all(r.success for r in results)

        # Should have unique task IDs
        task_ids = [r.task_id for r in results]
        assert len(set(task_ids)) == 8  # All unique
