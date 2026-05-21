#!/usr/bin/env python3
"""Debug script for gifsicle pipeline execution issue.

This script systematically tests each step of the problematic pipeline:
gifsicle-frame → ffmpeg-color → animately-advanced

to identify where the multi-frame GIF is being corrupted to single-frame.
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from giflab.tool_wrappers import (
    GifsicleFrameReducer,
    AnimatelyFrameReducer, 
    FFmpegColorReducer,
    AnimatelyAdvancedLossyCompressor
)

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_gif_info(gif_path: Path) -> Dict[str, Any]:
    """Get GIF information using gifsicle --info."""
    try:
        result = subprocess.run(
            ['gifsicle', '--info', str(gif_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {'error': f"Command failed: {result.stderr}"}
            
        info_lines = result.stdout.strip().split('\n')
        
        # Parse frame count and basic info
        frames = 0
        logical_screen = None
        
        for line in info_lines:
            if ' images' in line:
                # Extract frame count from "* file.gif X images"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'images' and i > 0:
                        try:
                            frames = int(parts[i-1])
                            break
                        except ValueError:
                            continue
            elif 'logical screen' in line:
                logical_screen = line.strip()
                
        file_size = gif_path.stat().st_size
        
        return {
            'frames': frames,
            'file_size_bytes': file_size,
            'file_size_kb': file_size / 1024,
            'logical_screen': logical_screen,
            'raw_output': result.stdout
        }
        
    except Exception as e:
        return {'error': str(e)}

def test_individual_step(step_name: str, tool_class, input_path: Path, 
                        output_path: Path, params: Dict[str, Any], 
                        logger: logging.Logger) -> Dict[str, Any]:
    """Test an individual pipeline step."""
    logger.info(f"\n{'='*50}")
    logger.info(f"TESTING STEP: {step_name}")
    logger.info(f"{'='*50}")
    
    # Get input info
    input_info = get_gif_info(input_path)
    logger.info(f"INPUT - Frames: {input_info.get('frames', 'unknown')}, "
               f"Size: {input_info.get('file_size_kb', 0):.1f}KB")
    
    # Apply the tool
    try:
        tool = tool_class()
        logger.info(f"Applying {tool_class.__name__} with params: {params}")
        
        result = tool.apply(input_path, output_path, params=params)
        
        # Get output info
        if output_path.exists():
            output_info = get_gif_info(output_path)
            logger.info(f"OUTPUT - Frames: {output_info.get('frames', 'unknown')}, "
                       f"Size: {output_info.get('file_size_kb', 0):.1f}KB")
            
            # Check for frame loss
            input_frames = input_info.get('frames', 0)
            output_frames = output_info.get('frames', 0)
            
            if input_frames > 0 and output_frames > 0:
                frame_ratio = output_frames / input_frames
                logger.info(f"FRAME RATIO: {output_frames}/{input_frames} = {frame_ratio:.2f}")
                
                if output_frames == 1 and input_frames > 1:
                    logger.error("🚨 CRITICAL: Multi-frame input reduced to single frame!")
                elif output_frames < input_frames:
                    logger.info(f"✅ Frame reduction applied: {input_frames} → {output_frames}")
                elif output_frames == input_frames:
                    logger.info(f"✅ Frame count preserved: {input_frames}")
                else:
                    logger.warning(f"⚠️  Unexpected frame increase: {input_frames} → {output_frames}")
            
            return {
                'success': True,
                'input_info': input_info,
                'output_info': output_info,
                'result': result,
                'frame_ratio': frame_ratio if input_frames > 0 else 0
            }
        else:
            logger.error(f"❌ Output file not created: {output_path}")
            return {'success': False, 'error': 'Output file not created'}
            
    except Exception as e:
        logger.error(f"❌ Step failed with exception: {e}")
        return {'success': False, 'error': str(e)}

def test_pipeline_combination(steps: List[Dict[str, Any]], input_path: Path,
                             debug_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Test a combination of pipeline steps."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING PIPELINE: {' → '.join([step['name'] for step in steps])}")
    logger.info(f"{'='*60}")
    
    current_input = input_path
    results = []
    
    for i, step in enumerate(steps):
        step_name = step['name']
        tool_class = step['tool_class']
        params = step['params']
        
        # Create output path for this step
        output_path = debug_dir / f"step_{i+1}_{step_name.replace('-', '_')}.gif"
        
        # Test this step
        result = test_individual_step(step_name, tool_class, current_input, 
                                    output_path, params, logger)
        
        results.append({
            'step': step_name,
            'result': result
        })
        
        # Check if step failed
        if not result.get('success', False):
            logger.error(f"❌ Pipeline failed at step: {step_name}")
            return {
                'success': False,
                'failed_at_step': step_name,
                'results': results
            }
        
        # Use output as input for next step
        current_input = output_path
    
    logger.info("✅ Pipeline completed successfully")
    return {
        'success': True,
        'results': results,
        'final_output': current_input
    }

def main():
    """Main debugging routine."""
    logger = setup_logging(verbose=True)
    
    logger.info("🐛 GifLab Pipeline Debugging Script")
    logger.info("Investigating gifsicle pipeline execution issue")
    
    # Set up paths
    test_gif = Path('/Users/lachlants/repos/animately/giflab/results/samples/synthetic/animation_heavy.gif')
    
    if not test_gif.exists():
        logger.error(f"❌ Test GIF not found: {test_gif}")
        return 1
    
    # Create debug directory
    debug_dir = Path('/tmp/giflab_debug')
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)
    
    logger.info(f"📁 Debug output directory: {debug_dir}")
    
    # Get initial GIF info
    initial_info = get_gif_info(test_gif)
    logger.info(f"🎬 Test GIF - Frames: {initial_info.get('frames', 'unknown')}, "
               f"Size: {initial_info.get('file_size_kb', 0):.1f}KB")
    
    # Phase 1: Test individual steps
    logger.info("\n🔍 PHASE 1: Testing individual pipeline steps")
    
    individual_tests = [
        {
            'name': 'gifsicle-frame',
            'tool_class': GifsicleFrameReducer,
            'params': {'ratio': 0.5},
            'output': debug_dir / 'individual_gifsicle_frame.gif'
        },
        {
            'name': 'ffmpeg-color',  
            'tool_class': FFmpegColorReducer,
            'params': {'colors': 32},
            'output': debug_dir / 'individual_ffmpeg_color.gif'
        },
        {
            'name': 'animately-advanced',
            'tool_class': AnimatelyAdvancedLossyCompressor,
            'params': {'lossy_level': 40},
            'output': debug_dir / 'individual_animately_lossy.gif'
        }
    ]
    
    individual_results = {}
    for test in individual_tests:
        result = test_individual_step(
            test['name'], 
            test['tool_class'], 
            test_gif, 
            test['output'],
            test['params'], 
            logger
        )
        individual_results[test['name']] = result
    
    # Phase 2: Test problematic pipeline
    logger.info("\n🔍 PHASE 2: Testing problematic pipeline combination")
    
    problematic_pipeline = [
        {
            'name': 'gifsicle-frame',
            'tool_class': GifsicleFrameReducer,
            'params': {'ratio': 0.5}
        },
        {
            'name': 'ffmpeg-color',
            'tool_class': FFmpegColorReducer, 
            'params': {'colors': 32}
        },
        {
            'name': 'animately-advanced',
            'tool_class': AnimatelyAdvancedLossyCompressor,
            'params': {'lossy_level': 40}
        }
    ]
    
    problematic_result = test_pipeline_combination(
        problematic_pipeline, test_gif, debug_dir, logger
    )
    
    # Phase 3: Test working pipeline for comparison
    logger.info("\n🔍 PHASE 3: Testing working pipeline for comparison")
    
    working_pipeline = [
        {
            'name': 'animately-frame',
            'tool_class': AnimatelyFrameReducer,
            'params': {'ratio': 0.5}
        },
        {
            'name': 'ffmpeg-color',
            'tool_class': FFmpegColorReducer,
            'params': {'colors': 32}
        },
        {
            'name': 'animately-advanced', 
            'tool_class': AnimatelyAdvancedLossyCompressor,
            'params': {'lossy_level': 40}
        }
    ]
    
    working_result = test_pipeline_combination(
        working_pipeline, test_gif, debug_dir, logger
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEBUGGING SUMMARY")
    logger.info("="*60)
    
    # Individual step results
    logger.info("\n📊 Individual Step Results:")
    for step_name, result in individual_results.items():
        if result.get('success'):
            input_frames = result['input_info'].get('frames', 0)
            output_frames = result['output_info'].get('frames', 0)
            logger.info(f"  {step_name}: {input_frames} → {output_frames} frames")
        else:
            logger.info(f"  {step_name}: FAILED - {result.get('error', 'unknown error')}")
    
    # Pipeline results
    logger.info(f"\n🔴 Problematic Pipeline: {'SUCCESS' if problematic_result.get('success') else 'FAILED'}")
    if not problematic_result.get('success'):
        logger.info(f"    Failed at step: {problematic_result.get('failed_at_step', 'unknown')}")
    
    logger.info(f"\n🟢 Working Pipeline: {'SUCCESS' if working_result.get('success') else 'FAILED'}")
    
    logger.info(f"\n📁 Debug files saved to: {debug_dir}")
    logger.info("🔍 Examine intermediate outputs to identify the exact failure point")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())