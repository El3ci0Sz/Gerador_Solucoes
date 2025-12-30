# scripts/runner.py

"""
Main script for executing mapping generation.

Available commands:
- single: Single execution with specific parameters.
- benchmark_v2: Configurable full benchmark.
"""

import os
import sys

# Add parent directory to path immediately
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from mapping_generator.cli import create_parser
from mapping_generator.generation.controller import GenerationTask
from mapping_generator.utils.logger_setup import setup_logger

def run_single_generation(args):
    """
    Executes a single generation task based on CLI arguments.
    """
    print("=" * 60)
    print("SINGLE EXECUTION - Single Generation")
    print("=" * 60)
    
    task_params = _build_task_params_from_args(args)
    
    try:
        # Unpack parameters for Task
        task = GenerationTask(**task_params)
        success = task.run()
        
        if success:
            print("\nGeneration completed successfully!")
            if task.generator and task.file_saver:
                print(f"Results at: {os.path.abspath(task.file_saver.output_dir)}")
        else:
            print("\nGeneration failed.")
            
    except Exception as e:
        print(f"\nError during generation: {e}")
        logging.error(f"Task error: {e}", exc_info=True)

def _build_task_params_from_args(args) -> dict:
    """Constructs parameter dictionary from arguments."""
    params = {
        'tec': args.tec,
        'gen_mode': args.gen_mode,
        'k': args.k_graphs,
        'output_dir': args.output_dir,
        'no_images': args.no_images,
        'retries_multiplier': args.retries_multiplier,
        'visualize': args.visualize
    }
    
    if args.tec == 'cgra':
        params['arch_sizes'] = [tuple(args.arch_size)]
        params['cgra_params'] = {'bits': args.bits}
        params['graph_range'] = tuple(args.graph_range)
        params['k_range'] = tuple(args.k_range)
        params['no_extend_io'] = args.no_extend_io
        params['max_path_length'] = args.max_path_length
        params['ii'] = args.ii
        params['alpha'] = args.alpha
        
        if args.gen_mode == 'grammar':
            params['strategy'] = args.strategy
            if args.strategy == 'systematic':
                params['difficulty'] = args.difficulty
            elif args.strategy == 'random':
                if not args.difficulty_range:
                    raise ValueError("--difficulty-range required for --strategy random")
                params['difficulty_range'] = args.difficulty_range
                params['adaptive'] = True
                
            if hasattr(args, 'flexible_recipe'):
                params['flexible_recipe'] = args.flexible_recipe
    
    elif args.tec == 'qca':
        params['arch_sizes'] = [tuple(args.arch_size)]
        params['qca_arch'] = args.qca_arch
        params['num_inputs'] = args.num_inputs
        params['num_derivations'] = args.num_derivations
        params['routing_factor'] = args.routing_factor
        
        if hasattr(args, 'force_grid_size') and args.force_grid_size:
            params['force_grid_size'] = tuple(args.force_grid_size)
            
        if hasattr(args, 'grammar_reconvergence'):
            params['grammar_reconvergence'] = args.grammar_reconvergence
        
        if hasattr(args, 'balanced') and args.balanced:
            params['balanced'] = True
        elif hasattr(args, 'unbalanced') and args.unbalanced:
            params['unbalanced'] = True
        else:
            params['balanced'] = True
            
        if hasattr(args, 'use_layered_tree'):
            params['use_layered_tree'] = args.use_layered_tree
        if hasattr(args, 'obstacle_intensity'):
            params['obstacle_intensity'] = args.obstacle_intensity
    
    return params

def run_benchmark_v2(args):
    print("=" * 60)
    print("BENCHMARK V2")
    print("=" * 60)
    print("Benchmark V2 is under development.")

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logger(verbose=args.verbose)
    
    if args.command == 'single':
        run_single_generation(args)
    elif args.command == 'benchmark_v2':
        run_benchmark_v2(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()