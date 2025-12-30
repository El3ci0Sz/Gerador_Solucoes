import random
import networkx as nx
import logging
from typing import Optional, List, Tuple, Set, Dict

try:
    from ...architectures.qca import QCA
except ImportError:
    pass

logger = logging.getLogger(__name__)

class Qca2DDWaveCascadingGenerator:
    """
    QCA 2DDWave Generator using Natural Growth and Progressive Merge strategy.
    
    This generator is specifically designed for 2DDWave architecture, enforcing
    monotonic signal flow (South/East) and avoiding crossovers.
    
    Strategy:
    1. Random inputs on TOP/LEFT borders.
    2. Free growth of trees.
    3. Progressive merging of trees.
    4. Post-merge growth to reach common points.
    5. Output extension to RIGHT/BOTTOM border.
    """
    
    def __init__(self, qca_architecture, num_inputs: int, target_nodes: int,
                 allow_reconvergence: bool = False, max_skew: int = 3, force_static_grid: bool = False):
        """
        Initializes the generator.

        Args:
            qca_architecture: The QCA architecture object.
            num_inputs (int): Number of inputs.
            target_nodes (int): Target number of nodes for the graph.
            allow_reconvergence (bool): If True, allows fanout (node with >1 children).
            max_skew (int): Maximum allowed skew (unused in current logic but kept for interface).
            force_static_grid (bool): If True, the grid will not expand.
        """
        self.qca_architecture = qca_architecture
        self.num_inputs = num_inputs
        self.target_nodes = target_nodes
        self.allow_reconvergence = allow_reconvergence
        self.max_skew = max_skew
        self.force_static_grid = force_static_grid
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes: Set[Tuple[int, int]] = set()
        self.qca_arch_graph = None
        self.qca_border_nodes = None
        
        self.blocked_leaves: Set[Tuple[int, int]] = set()
        
    def generate(self) -> Optional[nx.DiGraph]:
        """
        Executes the natural growth generation process.

        Returns:
            Optional[nx.DiGraph]: The generated graph or None if failed.
        """
        try:
            self.qca_arch_graph = self.qca_architecture.get_graph()
            self.qca_border_nodes = self.qca_architecture.get_border_nodes()
            self.placement_graph.clear()
            self.used_nodes.clear()
            self.blocked_leaves.clear()
            
            if not self._place_inputs_randomly_on_valid_borders():
                return None
            
            self._grow_freely_until_merge()
            self._progressive_merge()
            self._ensure_outputs_on_border()
            
            if self._validate_graph():
                self._print_stats()
                return self.placement_graph
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in 2DD generator: {e}", exc_info=True)
            return None
    
    def _place_inputs_randomly_on_valid_borders(self) -> bool:
        """
        Places inputs randomly on the TOP and LEFT borders, which are valid
        starting points for 2DDWave flow.

        Returns:
            bool: True if inputs were placed successfully.
        """
        rows, cols = self.qca_architecture.dim
        
        top_border = [(0, c) for c in range(cols) if (0, c) in self.qca_border_nodes]
        left_border = [(r, 0) for r in range(1, rows) if (r, 0) in self.qca_border_nodes]
        
        all_candidates = top_border + left_border
        
        if len(all_candidates) < self.num_inputs:
            return False
        
        random.shuffle(all_candidates)
        input_positions = all_candidates[:self.num_inputs]
        
        for i, pos in enumerate(input_positions):
            self.placement_graph.add_node(pos, type='input', name=f'in_{i}', tree_id=i)
            self.used_nodes.add(pos)
        
        return True
    
    def _grow_freely_until_merge(self):
        """
        Grows trees freely from inputs until they merge naturally or reach the target size.
        """
        current_nodes = self.placement_graph.number_of_nodes()
        iterations = 0
        max_iterations = 500
        
        while current_nodes < self.target_nodes and iterations < max_iterations:
            iterations += 1
            
            all_leaves = [n for n, deg in self.placement_graph.out_degree() 
                         if deg == 0 and n not in self.blocked_leaves]
            
            if not all_leaves:
                break
            
            random.shuffle(all_leaves)
            grew_this_iteration = False
            
            for leaf in all_leaves:
                if current_nodes >= self.target_nodes:
                    break
                
                next_node = self._find_next_free_2ddwave(leaf)
                
                if not next_node:
                    self.blocked_leaves.add(leaf)
                    continue
                
                if next_node in self.used_nodes:
                    self.placement_graph.add_edge(leaf, next_node)
                    grew_this_iteration = True
                else:
                    self.placement_graph.add_edge(leaf, next_node)
                    self.placement_graph.nodes[next_node]['type'] = 'operation'
                    self.used_nodes.add(next_node)
                    current_nodes += 1
                    grew_this_iteration = True
            
            if not grew_this_iteration:
                break
    
    def _find_next_free_2ddwave(self, current: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Finds the next valid position for growth respecting 2DDWave physics (Right/Down).

        Args:
            current (Tuple[int, int]): Current node coordinate.

        Returns:
            Optional[Tuple[int, int]]: Next coordinate or None.
        """
        r, c = current
        rows, cols = self.qca_architecture.dim
        
        directions = []
        if c + 1 < cols:
            directions.append((r, c + 1))
        if r + 1 < rows:
            directions.append((r + 1, c))
        
        random.shuffle(directions)
        
        for next_pos in directions:
            if next_pos not in self.used_nodes:
                return next_pos
            
            if next_pos in self.placement_graph.nodes():
                node_type = self.placement_graph.nodes[next_pos].get('type', 'unknown')
                if node_type == 'input':
                    continue
                return next_pos
        
        return None
    
    def _progressive_merge(self):
        """
        Continues to grow trees until they merge into a single component.
        """
        components = list(nx.weakly_connected_components(self.placement_graph))
        
        if len(components) == 1:
            return
        
        merge_round = 0
        max_rounds = 100
        
        while len(components) > 1 and merge_round < max_rounds:
            merge_round += 1
            
            all_leaves = [n for n, deg in self.placement_graph.out_degree() 
                         if deg == 0 and n not in self.blocked_leaves]
            
            if not all_leaves:
                break
            
            random.shuffle(all_leaves)
            grew_this_round = False
            merges_this_round = 0
            
            for leaf in all_leaves:
                next_node = self._find_next_free_2ddwave(leaf)
                
                if not next_node:
                    self.blocked_leaves.add(leaf)
                    continue
                
                if next_node in self.used_nodes:
                    self.placement_graph.add_edge(leaf, next_node)
                    merges_this_round += 1
                    grew_this_round = True
                else:
                    self.placement_graph.add_edge(leaf, next_node)
                    self.placement_graph.nodes[next_node]['type'] = 'routing'
                    self.used_nodes.add(next_node)
                    grew_this_round = True
            
            if not grew_this_round and merges_this_round == 0:
                break
            
            if merge_round % 20 == 0:
                components = list(nx.weakly_connected_components(self.placement_graph))
    
    def _ensure_outputs_on_border(self):
        """
        Ensures that at least one output is placed on the RIGHT or BOTTOM border.
        """
        leaves = [n for n, deg in self.placement_graph.out_degree() if deg == 0]
        rows, cols = self.qca_architecture.dim
        
        output_candidates = []
        
        for leaf in leaves:
            if leaf in self.qca_border_nodes:
                r, c = leaf
                if c == cols - 1 or r == rows - 1:
                    if self.placement_graph.nodes[leaf].get('type') != 'input':
                        output_candidates.append(leaf)
        
        if output_candidates:
            num_outputs = random.randint(1, min(3, len(output_candidates)))
            random.shuffle(output_candidates)
            
            for i in range(num_outputs):
                leaf = output_candidates[i]
                self.placement_graph.nodes[leaf]['type'] = 'output'
            return
        
        output_count = 0
        random.shuffle(leaves)
        
        for leaf in leaves:
            if random.random() < 0.4:
                success = self._grow_naturally_to_border(leaf)
                if success:
                    output_count += 1
                    if output_count >= 3:
                        break
        
        if output_count == 0 and leaves:
            self._grow_naturally_to_border(leaves[0])
    
    def _grow_naturally_to_border(self, start: Tuple[int, int]) -> bool:
        """
        Grows a specific node towards the border using natural random movement.

        Args:
            start (Tuple[int, int]): Starting node.

        Returns:
            bool: True if border was reached.
        """
        current = start
        max_steps = 50
        steps = 0
        rows, cols = self.qca_architecture.dim
        
        while steps < max_steps:
            r, c = current
            
            if c == cols - 1 or r == rows - 1:
                if current in self.qca_border_nodes:
                    self.placement_graph.nodes[current]['type'] = 'output'
                    return True
            
            next_node = self._find_next_towards_border(current)
            
            if not next_node:
                return False
            
            if next_node not in self.used_nodes:
                self.placement_graph.add_edge(current, next_node)
                self.placement_graph.nodes[next_node]['type'] = 'routing'
                self.used_nodes.add(next_node)
            else:
                self.placement_graph.add_edge(current, next_node)
            
            current = next_node
            steps += 1
        
        return False
    
    def _find_next_towards_border(self, current: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Finds the next node prioritizing directions towards the border.

        Args:
            current (Tuple[int, int]): Current node.

        Returns:
            Optional[Tuple[int, int]]: Next node.
        """
        r, c = current
        rows, cols = self.qca_architecture.dim
        
        candidates = []
        dist_right = (cols - 1) - c
        dist_bottom = (rows - 1) - r
        
        if dist_right > dist_bottom:
            if c + 1 < cols:
                candidates.append(((r, c + 1), 2))
            if r + 1 < rows:
                candidates.append(((r + 1, c), 1))
        else:
            if r + 1 < rows:
                candidates.append(((r + 1, c), 2))
            if c + 1 < cols:
                candidates.append(((r, c + 1), 1))
        
        random.shuffle(candidates)
        
        for pos, _ in candidates:
            if pos not in self.used_nodes:
                return pos
        
        return None
    
    def _validate_graph(self) -> bool:
        """
        Validates the graph against 2DDWave constraints (DAG, connectivity, no fanout).

        Returns:
            bool: True if valid.
        """
        if not nx.is_directed_acyclic_graph(self.placement_graph):
            return False
        
        if not nx.is_weakly_connected(self.placement_graph):
            return False
        
        for node in self.placement_graph.nodes():
            out_degree = self.placement_graph.out_degree(node)
            if out_degree > 1:
                return False
        
        for node in self.placement_graph.nodes():
            if self.placement_graph.nodes[node].get('type') == 'input':
                in_degree = self.placement_graph.in_degree(node)
                if in_degree > 0:
                    return False
        
        for node in self.placement_graph.nodes():
            if self.placement_graph.nodes[node].get('type') == 'input':
                r, c = node
                if not (r == 0 or c == 0):
                    return False
        
        for u, v in self.placement_graph.edges():
            delta_r = v[0] - u[0]
            delta_c = v[1] - u[1]
            
            if not ((delta_r == 0 and delta_c == 1) or (delta_r == 1 and delta_c == 0)):
                return False
        
        return True
    
    def _print_stats(self):
        """Logs final graph statistics."""
        nodes = self.placement_graph.number_of_nodes()
        edges = self.placement_graph.number_of_edges()
        logger.debug(f"2DDWave Generator: {nodes} nodes, {edges} edges")