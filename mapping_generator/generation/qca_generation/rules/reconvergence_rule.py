import random
import networkx as nx
import logging
from typing import TYPE_CHECKING, Tuple, Optional, List
from .base import BaseGrammarRule

if TYPE_CHECKING:
    from ..QcaGrammarGenerator import QcaGrammarGenerator

logger = logging.getLogger(__name__)


class ReconvergenceRule(BaseGrammarRule):
    """
    Reconvergence rule: Splits a start node into 'k' disjoint paths 
    that converge to a new destination node.
    """
    
    def __init__(self, k_range: Tuple[int, int] = (2, 2), max_path_length: int = 15, max_skew: int = 3):
        """
        Initializes the reconvergence rule.
        
        Args:
            k_range: Tuple (min_k, max_k) defining number of disjoint paths.
            max_path_length: Maximum allowed length for each path.
            max_skew: Maximum allowed difference between path lengths (default: 3)
        """
        super().__init__(k_range=k_range, max_path_length=max_path_length)
        self.k_range = k_range
        self.max_path_length = max_path_length
        self.max_skew = max_skew
    
    def get_rule_type(self) -> str:
        """Returns the rule identifier."""
        return "reconvergence"
    
    def can_apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Checks if the rule can potentially be applied.
        
        Args:
            generator: The generator instance.
            start_node: The starting node coordinate.
            
        Returns:
            bool: True if applicable.
        """
        if start_node not in generator.placement_graph.nodes():
            return False
        
        available_nodes = set(generator.qca_arch_graph.nodes()) - generator.used_nodes
        min_nodes_needed = self.k_range[0] + 1
        
        if len(available_nodes) < min_nodes_needed:
            return False
        
        return True

    def apply(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> bool:
        """
        Applies the reconvergence rule.
        
        Args:
            generator: The generator instance.
            start_node: The starting node coordinate.
            
        Returns:
            bool: True if successfully applied.
        """
        if not self.can_apply(generator, start_node):
            return False
        
        k = random.randint(self.k_range[0], self.k_range[1])
        target_pool = list(set(generator.qca_arch_graph.nodes()) - generator.used_nodes)
        random.shuffle(target_pool)

        for target_node in target_pool:
            paths = self._find_balanced_disjoint_paths(generator, start_node, target_node, k)
            
            if paths:
                if self._would_create_cycle(generator, paths):
                    continue
                
                try:
                    for path in paths:
                        generator.used_nodes.update(path)
                        nx.add_path(generator.placement_graph, path)
                    
                    generator.placement_graph.nodes[target_node]['type'] = 'operation'
                    
                    for path in paths:
                        for node in path[1:-1]:
                            if 'type' not in generator.placement_graph.nodes[node]:
                                generator.placement_graph.nodes[node]['type'] = 'routing'
                    
                    self._increment_counter()
                    logger.debug(f"ReconvergenceRule: Applied {k} balanced paths from {start_node} -> {target_node}")
                    return True
                    
                except Exception as e:
                    logger.error(f"ReconvergenceRule: Error applying: {e}", exc_info=True)
                    return False
        
        return False

    def _would_create_cycle(self, generator: 'QcaGrammarGenerator', paths: List[List[Tuple[int, int]]]) -> bool:
        """
        Checks if adding the proposed paths would create a cycle in the graph.
        
        Args:
            generator: The generator instance.
            paths: List of paths to be added.
            
        Returns:
            bool: True if a cycle would be created.
        """
        try:
            temp_graph = generator.placement_graph.copy()
            for path in paths:
                nx.add_path(temp_graph, path)
            return not nx.is_directed_acyclic_graph(temp_graph)
        except Exception as e:
            logger.error(f"ReconvergenceRule: Error checking cycle: {e}")
            return True

    def _find_balanced_disjoint_paths(self, generator: 'QcaGrammarGenerator', 
                                      source: Tuple[int, int], target: Tuple[int, int], 
                                      k: int) -> Optional[List[List[Tuple[int, int]]]]:
        """
        Finds 'k' paths that are disjoint at intermediate nodes and balanced in length.
        
        Args:
            generator: The generator instance.
            source: Source node.
            target: Target node.
            k: Number of paths needed.
            
        Returns:
            Optional[List[List[Tuple[int, int]]]]: List of paths or None.
        """
        try:
            temp_graph = generator.qca_arch_graph.copy()
            nodes_to_avoid = generator.used_nodes - {source}
            temp_graph.remove_nodes_from(nodes_to_avoid)

            try:
                all_paths = list(nx.all_shortest_paths(temp_graph, source=source, target=target))
            except nx.NetworkXNoPath:
                return None
            
            if not all_paths: return None
            
            base_path_length = len(all_paths[0]) - 1
            if base_path_length > self.max_path_length: return None
            if len(all_paths) < k: return None

            balanced_paths = []
            for path in all_paths:
                path_length = len(path) - 1
                skew = abs(path_length - base_path_length)
                if skew <= self.max_skew:
                    balanced_paths.append(path)
            
            if len(balanced_paths) < k: return None
            
            random.shuffle(balanced_paths)
            selected_paths = []
            claimed_intermediate = set()

            for path in balanced_paths:
                intermediate_nodes = set(path[1:-1])
                
                if claimed_intermediate.isdisjoint(intermediate_nodes):
                    selected_paths.append(path)
                    claimed_intermediate.update(intermediate_nodes)
                    
                    if len(selected_paths) == k:
                        lengths = [len(p) - 1 for p in selected_paths]
                        actual_skew = max(lengths) - min(lengths)
                        
                        if actual_skew <= self.max_skew:
                            return selected_paths
                        else:
                            return None
            
            return None
                        
        except nx.NodeNotFound:
            return None
        except Exception as e:
            logger.error(f"ReconvergenceRule: Unexpected error: {e}", exc_info=True)
            return None
    
    def estimate_cost(self, generator: 'QcaGrammarGenerator', start_node: Tuple[int, int]) -> Optional[int]:
        """Estimates the cost of applying this rule."""
        if not self.can_apply(generator, start_node):
            return None
        
        avg_k = (self.k_range[0] + self.k_range[1]) / 2
        avg_path_length = self.max_path_length / 2
        return int(avg_k * avg_path_length) + 1