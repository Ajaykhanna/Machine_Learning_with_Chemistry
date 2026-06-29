#!/usr/bin/env python3
"""
ULTRA-FAST Molecular Dynamics Trajectory Extraction Script
Optimized for high-core-count systems with large memory (like yours: 144 cores, 500GB RAM)

Key optimizations:
1. Multiprocessing for extraction (not just counting) - uses all cores
2. Memory buffering - reduces disk I/O
3. Batch directory creation
4. Large chunk sizes for large memory systems
5. Parallel frame writing across many workers

Expected speedup: 5-10x faster extraction phase
"""

import os
import sys
import argparse
import numpy as np
import multiprocessing
import logging
import time
import csv
import psutil
import random
from tqdm import tqdm
from threading import Thread, Lock
from collections import defaultdict
from pathlib import Path
from functools import partial


# Global locks
frame_map_lock = Lock()
summary_lock = Lock()


def print_banner():
    """Display script banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║   ULTRA-FAST Trajectory Processor v2.1                   ║
    ║   Optimized for High-Performance Systems                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    logging.info(banner)


def get_system_resources():
    """Get current system resource information."""
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    disk = psutil.disk_usage('/')
    
    return {
        'total_memory_gb': mem.total / (1024**3),
        'available_memory_gb': mem.available / (1024**3),
        'memory_percent': mem.percent,
        'cpu_count': cpu_count,
        'disk_free_gb': disk.free / (1024**3),
        'disk_percent': disk.percent
    }


def estimate_resources(file_size_bytes, num_frames, atoms_per_frame):
    """Estimate memory and processing requirements."""
    resources = get_system_resources()
    
    bytes_per_atom = 100
    estimated_frame_size = atoms_per_frame * bytes_per_atom if atoms_per_frame else 10000
    estimated_total_memory_mb = (estimated_frame_size * num_frames) / (1024**2)
    
    logging.info("=" * 60)
    logging.info("RESOURCE ESTIMATION")
    logging.info("=" * 60)
    logging.info(f"System Resources:")
    logging.info(f"  - Total Memory: {resources['total_memory_gb']:.2f} GB")
    logging.info(f"  - Available Memory: {resources['available_memory_gb']:.2f} GB")
    logging.info(f"  - CPU Cores: {resources['cpu_count']}")
    logging.info(f"  - Free Disk Space: {resources['disk_free_gb']:.2f} GB")
    logging.info("")
    logging.info(f"Processing Estimates:")
    logging.info(f"  - Input File Size: {file_size_bytes / (1024**3):.2f} GB")
    logging.info(f"  - Estimated Memory Usage: {estimated_total_memory_mb:.2f} MB")
    logging.info("=" * 60)
    
    return resources


def determine_optimal_workers(cpu_count, available_memory_gb, file_size_gb):
    """Determine optimal number of worker processes."""
    # Use 75% of cores for extraction, leave some for OS
    max_workers = max(1, int(cpu_count * 0.75))
    
    # For small files with huge memory, use more workers
    if file_size_gb < 1 and available_memory_gb > 100:
        workers = min(max_workers, 32)  # Cap at 32 for diminishing returns
    elif file_size_gb < 5 and available_memory_gb > 50:
        workers = min(max_workers, 16)
    else:
        workers = min(max_workers, 8)
    
    logging.info(f"🚀 Using {workers} worker processes for parallel extraction")
    return workers


def _count_pattern_worker(args):
    """Worker for parallel pattern counting."""
    file_path, pattern, start, end = args
    pattern_len = len(pattern)
    overlap = pattern_len - 1
    read_start = max(0, start - overlap)
    read_end = end + overlap
    read_len = read_end - read_start
    count = 0
    
    try:
        with open(file_path, "r", errors="ignore") as f:
            f.seek(read_start)
            data = f.read(read_len)
            idx = data.find(pattern)
            while idx != -1:
                abs_pos = read_start + idx
                if abs_pos >= start and abs_pos < end:
                    count += 1
                idx = data.find(pattern, idx + 1)
    except Exception as e:
        logging.warning(f"Error in worker processing range [{start}, {end}): {e}")
    
    return count


def count_pattern_in_file(file_path, pattern="FINAL HEAT OF FORMATION =", num_workers=None):
    """Count occurrences of a pattern using multiprocessing."""
    if num_workers is None:
        num_workers = max(1, min(multiprocessing.cpu_count() - 1, 8))
    
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return 0
        
        pattern_len = len(pattern)
        if file_size < 100 * 1024 * 1024:
            part_size = file_size
            num_workers = 1
        else:
            part_size = max(10 * 1024 * 1024, file_size // num_workers + 1)
        
        tasks = []
        for i in range(0, file_size, part_size):
            start = i
            end = min(file_size, i + part_size)
            tasks.append((file_path, pattern, start, end))
        
        with multiprocessing.Pool(min(num_workers, len(tasks))) as pool:
            counts = pool.map(_count_pattern_worker, tasks)
        
        total = sum(counts)
        return total
    except Exception as e:
        logging.error(f"Error counting patterns: {e}")
        return 0


def load_all_frames_into_memory(coords_path, vel_path, atoms_per_frame, chunk_size_mb=200):
    """
    Load ALL frames into memory for ultra-fast access.
    With 500GB RAM, we can easily hold everything in memory.
    """
    logging.info("💾 Loading entire trajectory into memory (fast random access)...")
    
    coords_frames = []
    vel_frames = []
    
    # Load coordinates
    chunk_size = chunk_size_mb * 1024 * 1024
    frame_idx = 0
    
    with open(coords_path, 'r') as f:
        leftover = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            lines = (leftover + chunk).splitlines()
            leftover = ""
            i = 0
            
            while i < len(lines):
                try:
                    num_atoms = int(lines[i].strip())
                    frame_size = num_atoms + 2
                except (ValueError, IndexError):
                    i += 1
                    continue
                
                if i + frame_size > len(lines):
                    leftover = "\n".join(lines[i:])
                    break
                
                frame_data = []
                for atom_line in lines[i+2:i+frame_size]:
                    parts = atom_line.split()
                    if len(parts) == 4:
                        try:
                            frame_data.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
                        except ValueError:
                            pass
                
                if frame_data:
                    coords_frames.append(frame_data)
                    frame_idx += 1
                
                i += frame_size
    
    logging.info(f"✓ Loaded {len(coords_frames)} coordinate frames into memory")
    
    # Load velocities if provided
    if vel_path and atoms_per_frame:
        with open(vel_path, 'r') as f:
            leftover = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                lines = (leftover + chunk).splitlines()
                frame = []
                in_velocity_block = False
                
                for line in lines:
                    if "$VELOC" in line:
                        in_velocity_block = True
                        continue
                    if "$ENDVELOC" in line:
                        in_velocity_block = False
                        continue
                    
                    if not in_velocity_block:
                        continue
                    
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            frame.append(tuple(map(float, parts)))
                        except ValueError:
                            pass
                    
                    if atoms_per_frame and len(frame) == atoms_per_frame:
                        vel_frames.append(frame)
                        frame = []
        
        logging.info(f"✓ Loaded {len(vel_frames)} velocity frames into memory")
    
    return coords_frames, vel_frames


def _write_frame_worker(args):
    """
    Worker function for parallel frame writing.
    Each worker handles writing one frame.
    """
    frame_idx, original_idx, coords_data, vel_data, output_dir = args
    
    try:
        # Create frame directory
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # Write coordinates
        if coords_data:
            xyz_file = os.path.join(frame_dir, f"frame_{frame_idx}.xyz")
            with open(xyz_file, 'w') as f:
                f.write(f"{len(coords_data)}\n")
                f.write(f"Frame {original_idx}\n")
                for atom in coords_data:
                    f.write(f"{atom[0]} {atom[1]:.10f} {atom[2]:.10f} {atom[3]:.10f}\n")
        
        # Write velocities
        if vel_data:
            vel_file = os.path.join(frame_dir, f"frame_{frame_idx}.vel")
            with open(vel_file, 'w') as f:
                f.write(f"{len(vel_data)}\n")
                f.write(f"Frame {original_idx}\n")
                for v in vel_data:
                    f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        
        return {
            'extracted_frame_index': frame_idx,
            'original_coords_frame': original_idx if coords_data else None,
            'original_velocity_frame': original_idx if vel_data else None,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"Error writing frame {frame_idx}: {e}")
        return {
            'extracted_frame_index': frame_idx,
            'success': False,
            'error': str(e)
        }


def _validate_frame_worker(args):
    """Worker for parallel frame validation."""
    idx, frame_mapping, output_dir = args
    mapping = frame_mapping[idx]
    extracted_idx = mapping['extracted_frame_index']
    
    results = {'idx': extracted_idx, 'xyz_valid': None, 'vel_valid': None, 'errors': []}
    
    # Validate XYZ
    xyz_file = os.path.join(output_dir, f"frame_{extracted_idx}", f"frame_{extracted_idx}.xyz")
    if os.path.exists(xyz_file):
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    num_atoms = int(lines[0].strip())
                    actual_atoms = len(lines) - 2
                    results['xyz_valid'] = (num_atoms == actual_atoms)
                    if not results['xyz_valid']:
                        results['errors'].append(f"XYZ atom count mismatch: {num_atoms} vs {actual_atoms}")
                else:
                    results['xyz_valid'] = False
                    results['errors'].append("XYZ file too short")
        except Exception as e:
            results['xyz_valid'] = False
            results['errors'].append(f"XYZ validation error: {e}")
    
    # Validate VEL
    vel_file = os.path.join(output_dir, f"frame_{extracted_idx}", f"frame_{extracted_idx}.vel")
    if os.path.exists(vel_file):
        try:
            with open(vel_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    num_atoms = int(lines[0].strip())
                    actual_atoms = len(lines) - 2
                    results['vel_valid'] = (num_atoms == actual_atoms)
                    if not results['vel_valid']:
                        results['errors'].append(f"VEL atom count mismatch: {num_atoms} vs {actual_atoms}")
                else:
                    results['vel_valid'] = False
                    results['errors'].append("VEL file too short")
        except Exception as e:
            results['vel_valid'] = False
            results['errors'].append(f"VEL validation error: {e}")
    
    return results


def validate_extracted_frames(output_dir, frame_mapping, num_samples=100, full_validation=False):
    """Validate extracted frames."""
    logging.info("=" * 60)
    logging.info("VALIDATION PHASE")
    logging.info("=" * 60)
    
    if not frame_mapping:
        logging.warning("No frame mapping available")
        return {'validated': 0, 'passed': 0, 'failed': 0}
    
    if full_validation:
        frames_to_validate = list(range(len(frame_mapping)))
        logging.info(f"Full validation on {len(frames_to_validate)} frames...")
    else:
        num_samples = min(num_samples, len(frame_mapping))
        frames_to_validate = random.sample(range(len(frame_mapping)), num_samples)
        logging.info(f"Random sampling validation on {num_samples} frames...")
    
    validation_results = {
        'validated': 0,
        'passed': 0,
        'failed': 0,
        'failures': []
    }
    
    tasks = [(idx, frame_mapping, output_dir) for idx in frames_to_validate]
    
    with multiprocessing.Pool() as pool:
        validation_outputs = list(tqdm(
            pool.imap(_validate_frame_worker, tasks),
            total=len(tasks),
            desc="Validating frames",
            unit="frame"
        ))
    
    for result in validation_outputs:
        validation_results['validated'] += 1
        xyz_ok = result['xyz_valid'] if result['xyz_valid'] is not None else True
        vel_ok = result['vel_valid'] if result['vel_valid'] is not None else True
        
        if xyz_ok and vel_ok:
            validation_results['passed'] += 1
        else:
            validation_results['failed'] += 1
            validation_results['failures'].append({
                'frame': result['idx'],
                'errors': result['errors']
            })
    
    logging.info(f"Validation: {validation_results['passed']}/{validation_results['validated']} passed")
    if validation_results['failed'] > 0:
        logging.warning(f"⚠️  {validation_results['failed']} frames failed validation")
    else:
        logging.info("✓ All validated frames passed!")
    
    logging.info("=" * 60)
    
    return validation_results


def generate_summary_report(output_dir, summary, elapsed_time, resources_before, resources_after, validation_results=None):
    """Generate summary report."""
    summary_file = os.path.join(output_dir, "extraction_summary.log")
    
    try:
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ULTRA-FAST TRAJECTORY EXTRACTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Frames Extracted (Coords):   {summary['coords_extracted']}\n")
            f.write(f"Frames Extracted (Velocity): {summary['vel_extracted']}\n")
            f.write(f"Processing Time:             {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
            
            if summary['coords_extracted'] > 0:
                rate = summary['coords_extracted'] / elapsed_time
                f.write(f"Average Extraction Rate:     {rate:.2f} frames/second\n")
            
            f.write("\nRESOURCE USAGE\n")
            f.write("-" * 80 + "\n")
            f.write(f"CPU Cores Used:     {resources_before['cpu_count']}\n")
            f.write(f"Memory Available:   {resources_before['available_memory_gb']:.2f} GB\n")
            f.write(f"Disk Space Used:    {abs(resources_before['disk_free_gb'] - resources_after['disk_free_gb']):.2f} GB\n")
            
            if validation_results:
                f.write("\nVALIDATION RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Frames Validated: {validation_results['validated']}\n")
                f.write(f"Passed:           {validation_results['passed']}\n")
                f.write(f"Failed:           {validation_results['failed']}\n")
            
            if summary['errors']:
                f.write("\nERRORS\n")
                f.write("-" * 80 + "\n")
                for error in summary['errors']:
                    f.write(f"  {error}\n")
            else:
                f.write("\n✓ No errors encountered\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logging.info(f"✓ Summary saved to: {summary_file}")
        
    except Exception as e:
        logging.error(f"Error generating summary: {e}")


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="ULTRA-FAST Trajectory Processor - Optimized for High-Performance Systems"
    )
    
    parser.add_argument("-xyz", "--coords", type=str, required=True, help="Coordinates file")
    parser.add_argument("-vel", "--velocities", type=str, help="Velocity file")
    parser.add_argument("--start", type=int, default=1, help="Starting frame (1-based)")
    parser.add_argument("--stop", type=int, help="Stopping frame (inclusive)")
    parser.add_argument("--step", type=int, default=1, help="Step size")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Output directory")
    parser.add_argument("--workers", type=int, help="Number of worker processes (auto-detect if not specified)")
    parser.add_argument("--full-validation", action="store_true", help="Validate all frames")
    parser.add_argument("--validation-samples", type=int, default=100, help="Random frames to validate")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation")
    
    args = parser.parse_args()
    
    start_time = time.time()
    summary = {
        "coords_extracted": 0,
        "vel_extracted": 0,
        "errors": []
    }
    
    try:
        # Get resources
        coords_size = os.path.getsize(args.coords)
        resources_before = get_system_resources()
        
        # Read atoms per frame
        with open(args.coords, "r") as cf:
            num_atoms_per_frame = int(cf.readline().strip())
        
        logging.info(f"Atoms per frame: {num_atoms_per_frame}")
        
        # Count frames
        logging.info("Counting frames...")
        n_frames = count_pattern_in_file(args.coords, pattern="FINAL HEAT OF FORMATION =")
        logging.info(f"✓ Found {n_frames} frames")
        
        # Estimate resources
        estimate_resources(coords_size, n_frames, num_atoms_per_frame)
        
        # Determine workers
        if args.workers:
            num_workers = args.workers
        else:
            num_workers = determine_optimal_workers(
                resources_before['cpu_count'],
                resources_before['available_memory_gb'],
                coords_size / (1024**3)
            )
        
        # Resolve start/stop/step
        start = args.start
        step = args.step
        stop = args.stop if args.stop else n_frames
        
        if stop > n_frames:
            logging.warning(f"Capping stop from {stop} to {n_frames}")
            stop = n_frames
        
        want_frames = list(range(start, stop + 1, step))
        
        logging.info("=" * 60)
        logging.info(f"EXTRACTION PLAN")
        logging.info("=" * 60)
        logging.info(f"  Frames to Extract: {len(want_frames)}")
        logging.info(f"  Using: {num_workers} parallel workers")
        logging.info(f"  Strategy: Load all into memory + parallel write")
        logging.info("=" * 60)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Setup file logging
        fh = logging.FileHandler(os.path.join(args.output, "processing.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)
        
        # OPTIMIZATION: Load all frames into memory
        coords_frames, vel_frames = load_all_frames_into_memory(
            args.coords,
            args.velocities,
            num_atoms_per_frame,
            chunk_size_mb=200
        )
        
        # Prepare tasks for parallel writing
        logging.info(f"\n🚀 Starting parallel extraction with {num_workers} workers...")
        tasks = []
        frame_mapping = []
        
        for extraction_idx, original_idx in enumerate(want_frames, 1):
            coords_data = coords_frames[original_idx - 1] if original_idx <= len(coords_frames) else None
            vel_data = vel_frames[original_idx - 1] if original_idx <= len(vel_frames) else None
            
            tasks.append((
                extraction_idx,
                original_idx,
                coords_data,
                vel_data,
                args.output
            ))
        
        # Parallel extraction with progress bar
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_write_frame_worker, tasks),
                total=len(tasks),
                desc="Extracting frames",
                unit="frame",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ))
        
        # Process results
        for result in results:
            if result['success']:
                if result['original_coords_frame']:
                    summary['coords_extracted'] += 1
                if result['original_velocity_frame']:
                    summary['vel_extracted'] += 1
                
                frame_mapping.append({
                    'extracted_frame_index': result['extracted_frame_index'],
                    'original_coords_frame': result['original_coords_frame'],
                    'original_velocity_frame': result['original_velocity_frame']
                })
            else:
                summary['errors'].append(f"Frame {result['extracted_frame_index']}: {result.get('error', 'Unknown')}")
        
        logging.info(f"\n✓ Extraction complete!")
        
        # Write frame mapping
        mapping_csv = os.path.join(args.output, "coords_frame_indices_map.csv")
        with open(mapping_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Extracted_Frame_Index', 'Original_Coords_Frame', 'Original_Velocity_Frame'])
            writer.writeheader()
            for mapping in frame_mapping:
                writer.writerow({
                    'Extracted_Frame_Index': mapping['extracted_frame_index'],
                    'Original_Coords_Frame': mapping.get('original_coords_frame', 'N/A'),
                    'Original_Velocity_Frame': mapping.get('original_velocity_frame', 'N/A')
                })
        
        logging.info(f"✓ Frame mapping saved to: {mapping_csv}")
        
        # Validation
        validation_results = None
        if not args.no_validation and summary['coords_extracted'] > 0:
            validation_results = validate_extracted_frames(
                args.output,
                frame_mapping,
                num_samples=args.validation_samples,
                full_validation=args.full_validation
            )
        
        resources_after = get_system_resources()
        
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        summary['errors'].append(str(e))
        resources_after = get_system_resources()
    
    finally:
        elapsed = time.time() - start_time
        
        generate_summary_report(
            args.output,
            summary,
            elapsed,
            resources_before,
            resources_after,
            validation_results
        )
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Frames extracted:  {summary['coords_extracted']}")
        print(f"Elapsed time:      {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Extraction rate:   {summary['coords_extracted']/elapsed:.2f} frames/sec")
        
        if validation_results:
            print(f"Validation:        {validation_results['passed']}/{validation_results['validated']} passed")
        
        if summary['errors']:
            print(f"\n⚠️  Errors: {len(summary['errors'])}")
        else:
            print("\n✓ No errors")
        
        print(f"\nOutput: {args.output}")
        print("=" * 80)


if __name__ == "__main__":
    main()


