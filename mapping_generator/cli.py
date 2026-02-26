import argparse

def create_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the command-line argument parser for the mapping generator.
    """
    parser = argparse.ArgumentParser(
        description="Mapping Generator for CGRA and QCA.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_single = subparsers.add_parser('single', help='Run a single generation task.')
    parser_single.add_argument('--tec', type=str, default='cgra', choices=['cgra', 'qca'], help='Target technology.')
    parser_single.add_argument('--gen-mode', type=str, default='grammar', choices=['grammar', 'random'], help='Generation mode.')
    parser_single.add_argument('--k-graphs', type=int, default=10, help='Number of graphs to generate.')
    
    parser_single.add_argument(
        '--strategy', 
        type=str, 
        default='systematic', 
        choices=['systematic', 'random'],
        help='Difficulty strategy for CGRA grammar. Only used with --tec cgra --gen-mode grammar.'
    )
    
    parser_single.add_argument(
        '--difficulty', 
        type=int, 
        default=1, 
        help='Difficulty level for systematic strategy (1-20). Required when --strategy systematic.'
    )
    
    parser_single.add_argument(
        '--difficulty-range', 
        type=int, 
        nargs=2, 
        metavar=('MIN', 'MAX'),
        help='Difficulty range for random strategy (e.g., 1 10). Required when --strategy random.'
    )
    
    parser_single.add_argument('--arch-size', type=int, nargs=2, default=[4, 4], help='Architecture dimensions (rows cols).')
    parser_single.add_argument('--graph-range', type=int, nargs=2, default=[8, 10], help='Min and max number of nodes for the DFG.')
    parser_single.add_argument('--bits', type=str, default='1000', help='CGRA interconnection bits (mdht).')
    parser_single.add_argument('--k-range', type=int, nargs=2, default=[2, 3], help='K-range for grammar rules.')
    parser_single.add_argument('--max-path-length', type=int, default=15, help='Max path length for routing.')
    
    parser_single.add_argument('--qca-arch', type=str, default='U', choices=['U', 'R', 'T'], help='QCA architecture type.')
    parser_single.add_argument('--num-inputs', type=int, default=3, help='Number of input nodes to seed.')
    parser_single.add_argument('--num-derivations', type=int, default=10, help='Number of derivations.')
    parser_single.add_argument('--routing-factor', type=float, default=2.5, help='Multiplier for routing.')
    parser_single.add_argument('--num-gates', type=int, default=10, help='Target number of logic gates.')
    parser_single.add_argument('--num-outputs', type=int, default=1, help='Number of output nodes to seed.')
    parser_single.add_argument('--no-detailed-stats', action='store_true', help='Disable exact node type counts in JSON.')
    
    group_qca_strat = parser_single.add_mutually_exclusive_group()
    group_qca_strat.add_argument(
        '--balanced',
        action='store_true',
        help='Generate BALANCED QCA graphs (100%% clock valid, RECOMMENDED).'
    )
    group_qca_strat.add_argument(
        '--unbalanced',
        action='store_true',
        help='Generate UNBALANCED QCA graphs (for comparison/testing).'
    )
    group_qca_strat.add_argument(
        '--backwards',
        action='store_true',
        help='Generate COMPLEX QCA graphs using Backwards (Reverse) flow.'
    )
    
    parser_single.add_argument('--no-extend-io', action='store_true', help='Disable I/O extension to border.')
    parser_single.add_argument('--no-images', action='store_true', help='Disable PNG image generation.')
    
    parser_single.add_argument(
        '--visualize', 
        action='store_true', 
        help='Enable generation of physical grid visualization files (.grid, .phys.png).'
    )
    
    parser_single.add_argument('--ii', type=int, default=None, help='Specify a fixed Initiation Interval (II).')
    parser_single.add_argument('--output-dir', type=str, default='results', help='Base directory to save the output files.')
    parser_single.add_argument('--alpha', type=float, default=0.3, help='For random mode: probability of adding extra edges.')
    parser_single.add_argument('--retries-multiplier', type=int, default=150, help='For grammar mode: multiplier for max attempts (k * multiplier).')
    
    parser_single.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed logs from the generation process.'
    )
    
    parser_single.add_argument(
        '--flexible-recipe',
        action='store_true',
        help='If set, accepts graphs even if the difficulty recipe (convergences) is not 100%% fulfilled (CGRA only).'
    )

    parser_benchmark_v2 = subparsers.add_parser('benchmark_v2', help='Run the new benchmark with configurable strategies.')
    parser_benchmark_v2.add_argument('--output-dir', type=str, default='results_benchmark_v2', help='Base directory to save the benchmark output files.')
    parser_benchmark_v2.add_argument('--workers', type=int, default=None, help='Number of parallel workers. Default is all available CPU cores.')
    parser_benchmark_v2.add_argument('--no-images', action='store_true', help='If specified, disables PNG image generation.')
    parser_benchmark_v2.add_argument('-v', '--verbose', action='store_true', help='Show detailed logs from the generation process.')
    parser_benchmark_v2.add_argument('--clean', action='store_true', help='Automatically run the post-process cleaner after the benchmark.')

    return parser
