import logging
import random
from typing import Optional, List, Tuple
from math import ceil
import os

from .generators.cgra_grammar_generator import CgraGrammarGenerator
from .generators.cgra_random_generator import CgraRandomGenerator
from .qca_generation.QcaMultiClusterGenerator import QcaMultiClusterGenerator
from .qca_generation.QcaGrammarGenerator import QcaGrammarGenerator
from .qca_generation.QcaBackwardsGenerator import QcaBackwardsGenerator 

from ..architectures.qca import QCA
from .strategies import SystematicStrategy, RandomStrategy
from ..utils.file_saver import FileSaver, OutputPathManager
from .strategies.recipes import generate_recipes
from ..utils.qca_analysis import QcaValidator
from ..utils.visualizer import GraphVisualizer

logger = logging.getLogger(__name__)

class QcaGeneratorWithSave:
    """
    Coordinates QCA mapping generation, validation, metrics calculation, and file saving.
    """
    def __init__(
        self, k_target: int, arch_sizes: List[Tuple[int, int]], qca_arch: str, 
        num_inputs: int, num_derivations: int, routing_factor: float, 
        retries_multiplier: int, file_saver: FileSaver, qca_strategy: str = 'multicluster', 
        num_gates: int = 10, num_outputs: int = 1, visualize: bool = False, 
        detailed_stats: bool = True
    ): 
        """
        Initializes the QCA generator wrapper.
        
        Args:
            k_target (int): Target number of valid graphs to generate.
            arch_sizes (List[Tuple[int, int]]): Possible grid dimensions for generation.
            qca_arch (str): QCA clock architecture type ('U', 'R', 'T').
            num_inputs (int): Number of inputs (used in grammar/multicluster strategies).
            num_derivations (int): Target depth/derivations (used in grammar/multicluster).
            routing_factor (float): Multiplier factor for routing constraints.
            retries_multiplier (int): Multiplier for allowed failed generation attempts.
            file_saver (FileSaver): Instance responsible for saving results.
            qca_strategy (str): Chosen generation strategy ('multicluster', 'backwards', 'grammar').
            num_gates (int): Minimum required gate count for backwards strategy.
            num_outputs (int): Required number of outputs for backwards strategy.
            visualize (bool): Flag to trigger visual dot and grid generation.
            detailed_stats (bool): Flag to count node types accurately post-pruning.
        """
        self.k_target = k_target
        self.arch_sizes = arch_sizes
        self.qca_arch_type = qca_arch
        self.num_inputs = num_inputs
        self.num_derivations = num_derivations
        self.routing_factor = routing_factor
        self.retries_multiplier = retries_multiplier
        self.file_saver = file_saver
        self.visualize = visualize
        self.num_gates = num_gates
        self.num_outputs = num_outputs
        self.detailed_stats = detailed_stats
        self.strategy = qca_strategy
        
        if self.strategy == 'multicluster':
            self.strategy_label = 'Balanced (MultiCluster)'
            logger.info("QCA Mode: BALANCED - MultiCluster")
        elif self.strategy == 'backwards':
            self.strategy_label = 'Complex (Backwards)'
            logger.info(f"QCA Mode: COMPLEX - Backwards Generator (Min Gates: {self.num_gates}, Outputs: {self.num_outputs})")
        else:
            self.strategy = 'grammar'
            self.strategy_label = 'Unbalanced (Grammar)'
            logger.warning("QCA Mode: UNBALANCED - Grammar")
        
        self.graphs_generated = 0

    def generate(self) -> bool:
        """
        Executes the main generation loop, evaluating validity and extracting metrics.
        
        Returns:
            bool: True if at least one graph was successfully generated and saved, False otherwise.
        """
        logger.info(f"Starting {self.strategy_label} generation")
        logger.info(f"Target: {self.k_target} graphs")
        
        saved_count = 0
        total_attempts = 0
        max_attempts = self.k_target * self.retries_multiplier
        generated_count = 0
        
        while saved_count < self.k_target and total_attempts < max_attempts:
            total_attempts += 1
            
            if total_attempts % 50 == 0:
                logger.info(f"Progress: {saved_count}/{self.k_target} saved, attempt {total_attempts}/{max_attempts}")
            
            try:
                initial_arch_size = random.choice(self.arch_sizes)
                qca_architecture = QCA(dimensions=initial_arch_size, arch_type=self.qca_arch_type)
                
                generator = self._get_generator_instance(qca_architecture)
                placement_graph = generator.generate()
                
                if not placement_graph or placement_graph.number_of_nodes() == 0:
                    continue
                
                final_arch_size = qca_architecture.dim 
                generated_count += 1
                
                try:
                    is_valid, errors = QcaValidator.validate(placement_graph, qca_architecture.get_border_nodes())
                    if not is_valid and generated_count <= 3:
                        logger.debug(f"Validation failed: {errors[:2]}")
                except Exception as e:
                    logger.error(f"Validation exception: {e}")
                
                metrics = {
                    'node_count': placement_graph.number_of_nodes(),
                    'edge_count': placement_graph.number_of_edges()
                }
                
                if self.detailed_stats:
                    counts = {'routing': 0, 'operation': 0, 'input': 0, 'output': 0, 'outros': 0}
                    for n, d in placement_graph.nodes(data=True):
                        ntype = d.get('type', 'unknown')
                        if ntype in counts:
                            counts[ntype] += 1
                        else:
                            counts['outros'] += 1
                    
                    metrics.update(counts)
                    logger.info(f"Stats - Portas: {counts['operation']} | Fios: {counts['routing']} | In: {counts['input']} | Out: {counts['output']}")

                try:
                    saved_count += 1
                    self._save_graph(placement_graph, saved_count, final_arch_size, metrics)
                except Exception as e:
                    logger.error(f"Save exception: {e}", exc_info=True)
                    saved_count -= 1
                    continue

            except Exception as e:
                logger.debug(f"Outer exception at attempt {total_attempts}: {e}")
                continue
        
        self.graphs_generated = saved_count
        logger.info(f"Generation Complete: Saved {saved_count}/{self.k_target} in {total_attempts} attempts.")
        return saved_count > 0
    
    def _get_generator_instance(self, qca_architecture: QCA) -> object:
        """
        Instantiates the required generation class based on the selected strategy.
        
        Args:
            qca_architecture (QCA): The loaded QCA architecture instance.
            
        Returns:
            object: An instance of the chosen generator strategy.
        """
        if self.strategy == 'multicluster':
            return QcaMultiClusterGenerator(qca_architecture=qca_architecture, num_inputs=self.num_inputs, target_depth=self.num_derivations)
        elif self.strategy == 'grammar':
            return QcaGrammarGenerator(qca_architecture=qca_architecture, num_inputs=self.num_inputs, num_derivations=self.num_derivations, routing_factor=self.routing_factor, strict_balance=False)
        elif self.strategy == 'backwards':
            return QcaBackwardsGenerator(qca_architecture=qca_architecture, target_gates=self.num_gates, num_outputs=self.num_outputs)
        else:
            raise ValueError(f"Invalid QCA strategy: {self.strategy}")

    def _save_graph(self, graph: nx.DiGraph, index: int, arch_size: Tuple[int, int], metrics: dict) -> Dict[str, Optional[str]]:
        """
        Prepares metadata and executes graph saving using the configured FileSaver.
        
        Args:
            graph (nx.DiGraph): The fully generated and validated graph.
            index (int): The generation iteration index.
            arch_size (Tuple[int, int]): Final architecture grid dimensions.
            metrics (dict): Collected metrics and statistics of the graph.
            
        Returns:
            Dict[str, Optional[str]]: Paths to the saved files.
        """
        num_nodes = metrics['node_count']
        num_edges = metrics['edge_count']
        
        if self.strategy == 'backwards':
            filename = f"qca_map_{arch_size[0]}x{arch_size[1]}_P{self.num_gates}_N{num_nodes}_E{num_edges}_{index}"
            difficulty_str = f"P{self.num_gates}"
        else:
            difficulty_str = f"i{self.num_inputs}d{self.num_derivations}"
            filename = OutputPathManager.build_filename(
                tec_name='QCA', arch_size=arch_size, num_nodes=num_nodes,
                num_edges=num_edges, difficulty=difficulty_str, index=index
            )
        
        subdirs = OutputPathManager.build_subdirs(
            tec_name='QCA', gen_mode=self.strategy, arch_size=arch_size,
            num_nodes=num_nodes, qca_arch_type=self.qca_arch_type
        )
        
        metadata = {
            'technology': 'QCA',
            'strategy': self.strategy,
            'arch_size': list(arch_size),
            'qca_arch_type': self.qca_arch_type,
            'metrics': metrics
        }
        
        if self.strategy != 'backwards':
            metadata['num_inputs'] = self.num_inputs
            metadata['num_derivations'] = self.num_derivations
            metadata['difficulty'] = difficulty_str

        try:
            paths = self.file_saver.save_graph(graph, filename, metadata, subdirs)
            
            if self.visualize:
                full_dir = os.path.join(self.file_saver.output_dir, subdirs)
                base_path = os.path.join(full_dir, filename)
                
                grid_path = f"{base_path}.grid"
                GraphVisualizer.save_placement_grid(graph, arch_size, grid_path)
                
                if not self.file_saver.no_images:
                    phys_dot_path = f"{base_path}.phys.dot"
                    phys_png_path = f"{base_path}.phys.png"
                    GraphVisualizer.generate_physical_dot(graph, arch_size, phys_dot_path)
                    try:
                        os.system(f"neato -n2 -Tpng {phys_dot_path} -o {phys_png_path}")
                    except Exception:
                        pass
            
            if paths and 'json' in paths:
                logger.info(f"Saved #{index} | Nodes:{num_nodes} | {filename}")
            else:
                logger.warning(f"Failed to save {self.strategy} graph #{index}")
            return paths
        except Exception as e:
            logger.error(f"Error saving graph #{index}: {e}", exc_info=True)
            raise


class GenerationTask:
    """
    Top-level manager class configuring and triggering the requested generation workflow.
    """
    def __init__(self, tec: str, gen_mode: str, k: int, output_dir: str = 'results', no_images: bool = False, **kwargs):
        """
        Initializes the Generation Task configuration.
        
        Args:
            tec (str): Target technology ('cgra' or 'qca').
            gen_mode (str): Mode of generation.
            k (int): Target number of graphs.
            output_dir (str): Base output directory.
            no_images (bool): Flag to skip image generation.
            **kwargs: Extra parameters dictating architecture options and configurations.
        """
        self.tec = tec
        self.gen_mode = gen_mode
        self.k = k
        self.output_dir = output_dir
        self.no_images = no_images
        self.params = kwargs
        self.file_saver = FileSaver(output_dir, no_images)
        self.generator = None
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validates provided technology parameter against supported types."""
        if self.tec not in ['cgra', 'qca']:
            raise ValueError(f"Invalid technology '{self.tec}'")
    
    def run(self) -> bool:
        """
        Instantiates the generator and starts the execution cycle.
        
        Returns:
            bool: Success status of the generated tasks.
        """
        try:
            self.generator = self._create_generator()
            if not self.generator:
                return False
            return self.generator.generate()
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return False
            
    def _create_generator(self) -> object:
        """
        Routes instantiation to the appropriate generator builder based on tech selection.
        
        Returns:
            object: An instance of the targeted generator wrapper.
        """
        if self.tec == 'cgra':
            if self.gen_mode == 'grammar': return self._create_cgra_grammar_generator()
            elif self.gen_mode == 'random': return self._create_cgra_random_generator()
        elif self.tec == 'qca':
            return self._create_qca_generator()
        raise ValueError(f"Unsupported combination: {self.tec} + {self.gen_mode}")

    def _create_cgra_grammar_generator(self) -> object:
        """
        Builds CGRA Grammar Generator with current parameters.
        
        Returns:
            object: Initialized CgraGrammarGenerator.
        """
        strategy = self._create_difficulty_strategy()
        return CgraGrammarGenerator(
            strategy=strategy, k_target=self.k, arch_sizes=self.params.get('arch_sizes', [(4, 4)]),
            cgra_params=self.params.get('cgra_params', {'bits': '1000'}), graph_range=self.params.get('graph_range', (10, 10)),
            k_range=self.params.get('k_range', (2, 3)), no_extend_io=self.params.get('no_extend_io', False),
            max_path_length=self.params.get('max_path_length', 15), fixed_ii=self.params.get('ii', None),
            retries_multiplier=self.params.get('retries_multiplier', 150), file_saver=self.file_saver,
            allow_partial_recipe=self.params.get('flexible_recipe', False)
        )

    def _create_cgra_random_generator(self) -> object:
        """
        Builds CGRA Random Generator with current parameters.
        
        Returns:
            object: Initialized CgraRandomGenerator.
        """
        return CgraRandomGenerator(
            k_target=self.k, arch_sizes=self.params.get('arch_sizes', [(4, 4)]),
            cgra_params=self.params.get('cgra_params', {'bits': '1000'}), graph_range=self.params.get('graph_range', (10, 10)),
            alpha=self.params.get('alpha', 0.3), fixed_ii=self.params.get('ii', None),
            retries_multiplier=self.params.get('retries_multiplier', 150), file_saver=self.file_saver
        )

    def _create_qca_generator(self) -> object:
        """
        Builds the QCA generator wrapper using parsed arguments.
        
        Returns:
            object: Initialized QcaGeneratorWithSave.
        """
        return QcaGeneratorWithSave(
            k_target=self.k,
            arch_sizes=self.params.get('arch_sizes', [(4, 4)]),
            qca_arch=self.params.get('qca_arch', 'U'),
            num_inputs=self.params.get('num_inputs', 3),
            num_derivations=self.params.get('num_derivations', 10),
            routing_factor=self.params.get('routing_factor', 2.5),
            retries_multiplier=self.params.get('retries_multiplier', 150),
            file_saver=self.file_saver,
            qca_strategy=self.params.get('qca_strategy', 'multicluster'),
            num_gates=self.params.get('num_gates', 10),
            num_outputs=self.params.get('num_outputs', 1),
            detailed_stats=self.params.get('detailed_stats', True), 
            visualize=self.params.get('visualize', False)
        )

    def _create_difficulty_strategy(self) -> object:
        """
        Determines and instantiates the logic strategy for CGRA generation constraint targeting.
        
        Returns:
            object: Loaded difficulty strategy block.
        """
        strategy_name = self.params.get('strategy', 'systematic')
        if strategy_name == 'systematic': return SystematicStrategy(difficulty=self.params.get('difficulty', 1))
        elif strategy_name == 'random':
            diff_range = self.params.get('difficulty_range', (1, 10))
            return RandomStrategy(difficulty_range=tuple(diff_range), adaptive=self.params.get('adaptive', True))
        else: raise ValueError(f"Unknown strategy: {strategy_name}")

def get_ii(num_nodes: int, arch_size: tuple, fixed_ii: Optional[int] = None) -> int:
    """
    Calculates the Initiation Interval (II).
    
    Args:
        num_nodes (int): Total number of nodes requiring logical placement.
        arch_size (tuple): Processing Elements Grid (rows, cols).
        fixed_ii (Optional[int]): Optional forced value bypassing automatic calculation.
        
    Returns:
        int: Computed or explicitly assigned Initiation Interval.
    """
    if fixed_ii is not None:
        return fixed_ii
    rows, cols = arch_size
    total_pes = rows * cols
    if total_pes == 0: return 1
    return int(ceil(num_nodes / total_pes))
