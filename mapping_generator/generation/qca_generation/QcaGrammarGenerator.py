import random
import networkx as nx
import logging
from typing import Optional, List, Tuple, Set

# Imports handling
try:
    from ...architectures.qca import QCA
    from .rules.tree_rule import TreeRule
    from .rules.reconvergence_rule import ReconvergenceRule
except ImportError:
    pass

logger = logging.getLogger(__name__)

class QcaGrammarGenerator:
    """
    Procedural QCA mapping generator using Stochastic Grammars.
    
    This generator supports both USE (Universal Scalable Efficient) and 
    2DDWave architectures by adapting its routing and merging strategies.
    It uses a grammar-based approach to grow logic structures and a smart
    merging algorithm to connect them while maintaining timing integrity.
    """
    
    def __init__(self, qca_architecture, num_inputs: int, num_derivations: int, 
                 routing_factor: float = 2.5, strict_balance: bool = False, 
                 force_static_grid: bool = False, grammar_reconvergence: bool = True):
        """
        Initializes the generator configuration.

        Args:
            qca_architecture: The QCA architecture object (contains dimensions and graph).
            num_inputs (int): Number of primary inputs to seed.
            num_derivations (int): Number of grammar rule applications (complexity control).
            routing_factor (float): Factor to estimate routing space (not currently used directly).
            strict_balance (bool): If True, would enforce strict ASAP leveling (deprecated/unused).
            force_static_grid (bool): If True, prevents grid expansion during generation.
            grammar_reconvergence (bool): If True, enables the ReconvergenceRule. If False, generates trees only.
        """
        self.qca_architecture = qca_architecture
        self.num_inputs = num_inputs
        self.num_derivations = num_derivations
        self.routing_factor = routing_factor
        self.strict_balance = strict_balance
        self.force_static_grid = force_static_grid
        self.grammar_reconvergence = grammar_reconvergence
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes: Set[Tuple[int, int]] = set()
        
        self.rules = [TreeRule()] 
        
        if self.grammar_reconvergence:
            self.rules.append(ReconvergenceRule(k_range=(2, 2)))
            logger.debug("Reconvergence Rule ENABLED (Grammar)")
        else:
            logger.debug("Reconvergence Rule DISABLED (Tree-only Grammar)")
        
        self.qca_arch_graph = None
        self.qca_border_nodes = None

        self.COST_FREE = 1
        self.COST_BLOCKED = 999999   
        
        raw_scheme = getattr(qca_architecture, 'clock_scheme', 
                         getattr(qca_architecture, 'arch_type', 'UNKNOWN'))
        scheme = str(raw_scheme).upper()

        if '2DD' in scheme or scheme == 'T':
            self.is_2dd = True
            self.COST_CROSSOVER = 999999 
            logger.debug(f"Mode 2DDWave ({scheme}): Enforcing Monotonic Flow.")
            self._enforce_2dd_physics()
        else:
            self.is_2dd = False
            self.COST_CROSSOVER = 15     
            logger.debug(f"Mode USE ({scheme}): Total flexibility.")

        if force_static_grid:
            self.MAX_GRID_EXPANSIONS = 0
        else:
            self.MAX_GRID_EXPANSIONS = 5 if self.is_2dd else 3
        
        self.MAX_PATH_LENGTH = 80
        self.MAX_MERGE_CANDIDATES = 60 
        self.grid_expansions_count = 0

    def _enforce_2dd_physics(self):
        """
        Removes invalid edges from the architecture graph to enforce 2DDWave physics.
        
        In 2DDWave, signals can only flow South or East. Any edge going North or West
        is physically impossible and is removed from the routing graph.
        """
        if not self.qca_arch_graph:
            self.qca_arch_graph = self.qca_architecture.get_graph()
        edges_to_remove = []
        for u, v in self.qca_arch_graph.edges():
            if v[0] < u[0] or v[1] < u[1]:
                edges_to_remove.append((u, v))
        self.qca_arch_graph.remove_edges_from(edges_to_remove)

    def generate(self) -> Optional[nx.DiGraph]:
        """
        Executes the full generation pipeline.

        Steps:
        1. Initialize architecture and border nodes.
        2. Seed input nodes.
        3. Apply growth rules iteratively.
        4. Merge disjoint components balancedly.
        5. Route outputs to the border.
        6. Validate the final graph.

        Returns:
            Optional[nx.DiGraph]: The generated placement graph, or None if generation failed.
        """
        try:
            self.qca_arch_graph = self.qca_architecture.get_graph()
            self.qca_border_nodes = self.qca_architecture.get_border_nodes()
            if self.is_2dd: self._enforce_2dd_physics()
                
            self.placement_graph.clear()
            self.used_nodes.clear()

            if not self._seed_input_nodes(): return None
            
            for _ in range(self.num_derivations):
                self._apply_growth_rule()
            
            if self.placement_graph.number_of_nodes() <= self.num_inputs: return None

            if not self._merge_trees_balanced(): return None

            self._mark_graph_outputs()
            
            if not self._ensure_all_outputs_on_border():
                logger.debug("Generation Failed: Could not route all outputs to border.")
                return None

            self._sanitize_node_types()
            
            if self.placement_graph.number_of_nodes() > 0:
                if self._validate_final_graph():
                    self._log_stats()
                    return self.placement_graph
            
            return None
        except Exception as e:
            logger.error(f"Error in generator: {e}", exc_info=True)
            return None

    def _merge_trees_balanced(self) -> bool:
        """
        Connects disconnected graph components while maintaining timing balance.

        It repeatedly identifies weakly connected components and attempts to merge them
        by finding paths that minimize skew (difference in arrival times).

        Returns:
            bool: True if all components were successfully merged into a single graph, False otherwise.
        """
        expansions_left = self.MAX_GRID_EXPANSIONS
        
        while len(list(nx.weakly_connected_components(self.placement_graph))) > 1:
            components = list(nx.weakly_connected_components(self.placement_graph))
            node_levels = self._calculate_node_levels()
            candidates = self._find_smart_merge_candidates(components, node_levels)
            
            merged = False
            for src, dst, _, _ in candidates:
                if nx.has_path(self.placement_graph, dst, src): continue

                path = self._find_optimized_path(src, dst)
                if path:
                    path_len = len(path) - 1
                    level_src = node_levels.get(src, 0)
                    level_dst = node_levels.get(dst, 0)
                    new_arrival = level_src + path_len
                    
                    if self.placement_graph.in_degree(dst) > 0:
                        skew = abs(new_arrival - level_dst)
                        if skew > 3: continue 
                    
                    tmp = self.placement_graph.copy()
                    nx.add_path(tmp, path)
                    if not nx.is_directed_acyclic_graph(tmp): continue
                        
                    self._realize_path(path)
                    merged = True
                    
                    if self.placement_graph.nodes[dst].get('type') == 'input':
                         self.placement_graph.nodes[dst]['type'] = 'routing'
                         self.placement_graph.nodes[dst].pop('name', None)
                    break
            
            if not merged:
                if expansions_left > 0:
                    self._expand_grid()
                    expansions_left -= 1
                else:
                    return False
        return True

    def _find_smart_merge_candidates(self, components, node_levels):
        """
        Identifies candidate pairs of nodes to merge different components.

        Prioritizes connecting a Leaf node from one component to a Root (Input) node of another,
        as this introduces zero skew. If forced to connect internally, it checks if the
        physical distance matches the logical level difference.

        Args:
            components (List[Set]): List of sets containing nodes for each component.
            node_levels (dict): Dictionary mapping nodes to their logical depth/level.

        Returns:
            List[Tuple]: A sorted list of tuples (source, target, priority, distance).
        """
        candidates = []
        comp_indices = list(range(len(components)))
        random.shuffle(comp_indices)
        
        for k in range(len(components) - 1):
            comp1 = list(components[comp_indices[k]])
            comp2 = list(components[comp_indices[k+1]])
            
            leaves1 = [n for n in comp1 if self.placement_graph.out_degree(n) == 0]
            roots2  = [n for n in comp2 if self.placement_graph.in_degree(n) == 0]
            leaves2 = [n for n in comp2 if self.placement_graph.out_degree(n) == 0]
            roots1  = [n for n in comp1 if self.placement_graph.in_degree(n) == 0]
            
            pairs = []
            for l in random.sample(leaves1, min(len(leaves1), 8)):
                for r in random.sample(roots2, min(len(roots2), 8)):
                    pairs.append((l, r, 0))
            for l in random.sample(leaves2, min(len(leaves2), 8)):
                for r in random.sample(roots1, min(len(roots1), 8)):
                    pairs.append((l, r, 0))

            if self.grammar_reconvergence: 
                nodes1 = random.sample(comp1, min(len(comp1), 5))
                nodes2 = random.sample(comp2, min(len(comp2), 5))
                for n1 in nodes1:
                    for n2 in nodes2:
                        pairs.append((n1, n2, 1))
                        pairs.append((n2, n1, 1))

            for u, v, prio in pairs:
                if self.is_2dd and (v[0] < u[0] or v[1] < u[1]): continue
                
                if not self.grammar_reconvergence and self.placement_graph.in_degree(v) > 0:
                    continue

                dist = abs(u[0] - v[0]) + abs(u[1] - v[1])
                
                if self.placement_graph.in_degree(v) == 0:
                    candidates.append((u, v, prio, dist))
                else:
                    lvl_u = node_levels.get(u, 0)
                    lvl_v = node_levels.get(v, 0)
                    if lvl_v > lvl_u:
                        needed = lvl_v - lvl_u
                        if abs(dist - needed) <= 4:
                            candidates.append((u, v, prio + 1, dist))

        candidates.sort(key=lambda x: (x[2], x[3]))
        return candidates[:self.MAX_MERGE_CANDIDATES]

    def _calculate_node_levels(self):
        """
        Calculates the logical level (ASAP scheduling) for every node in the graph.

        Returns:
            dict: A dictionary mapping node coordinates to their integer level.
        """
        levels = {}
        try:
            for n in nx.topological_sort(self.placement_graph):
                preds = list(self.placement_graph.predecessors(n))
                if not preds: levels[n] = 0
                else: levels[n] = max(levels[p] for p in preds) + 1
        except: pass
        return levels

    def _find_optimized_path(self, source, target, obstacles: Set = None) -> Optional[List[Tuple]]:
        """
        Finds the lowest-cost path between two nodes using A* algorithm.

        Args:
            source (Tuple[int, int]): Starting coordinate.
            target (Tuple[int, int]): Destination coordinate.
            obstacles (Set, optional): Set of nodes to treat as blocked (e.g., ancestors).

        Returns:
            Optional[List[Tuple]]: A list of coordinates representing the path, or None if no path found.
        """
        def weight_func(u, v, d):
            cost = self.COST_FREE
            if obstacles and v in obstacles:
                return self.COST_BLOCKED

            if v in self.used_nodes:
                if v == target:
                    return cost

                node_type = self.placement_graph.nodes[v].get('type', 'unknown')
                if node_type in ['routing', 'crossover', 'convergence']:
                    cost = self.COST_CROSSOVER
                elif node_type in ['input', 'output', 'operation']:
                    return self.COST_BLOCKED
            return cost

        def heuristic(u, v):
            return abs(u[0] - v[0]) + abs(u[1] - v[1])

        try:
            path = nx.astar_path(
                self.qca_arch_graph, source, target, 
                heuristic=heuristic, weight=weight_func
            )
            total_cost = sum(weight_func(path[i], path[i+1], {}) for i in range(len(path)-1))
            if total_cost >= self.COST_BLOCKED: return None
            if len(path)-1 > self.MAX_PATH_LENGTH: return None
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _realize_path(self, path: List[Tuple]):
        """
        Commits a path to the placement graph, updating node types and usage.

        Args:
            path (List[Tuple]): The path to realize.
        """
        nx.add_path(self.placement_graph, path)
        self.used_nodes.update(path)
        for node in path[1:-1]:
            data = self.placement_graph.nodes[node]
            if self.placement_graph.in_degree(node) > 1:
                data['type'] = 'crossover' if self.placement_graph.out_degree(node) > 1 else 'convergence'
            elif data.get('type') not in ['input', 'output']:
                data['type'] = 'routing'
                data['name'] = f"rout_{node}"

    def _seed_input_nodes(self) -> bool:
        """
        Places initial input nodes on valid borders.

        Returns:
            bool: True if inputs were successfully placed, False otherwise.
        """
        if not self.qca_border_nodes: return False
        avail = list(self.qca_border_nodes)
        if self.is_2dd: avail = [n for n in avail if n[0]==0 or n[1]==0]
        if len(avail) < self.num_inputs: return False
        for node in random.sample(avail, self.num_inputs):
            self.placement_graph.add_node(node, type='input', name=f"in_{node}")
            self.used_nodes.add(node)
        return True

    def _apply_growth_rule(self) -> bool:
        """
        Applies a random grammar rule to expand the graph from a leaf node.

        Returns:
            bool: True if a rule was successfully applied.
        """
        if not self.rules: return False
        leafs = [n for n, d in self.placement_graph.out_degree() if d == 0]
        if not leafs: return False
        rule = random.choice(self.rules)
        try: return rule.apply(self, random.choice(leafs))
        except: return False

    def _ensure_all_outputs_on_border(self) -> bool:
        """
        Routes all leaf nodes (outputs) to the nearest valid border.

        Returns:
            bool: True if all outputs were successfully routed to the border.
        """
        outs = [n for n, d in self.placement_graph.out_degree() if d == 0]
        rows, cols = self.qca_architecture.dim
        all_success = True
        
        for out in outs:
            if not (out[0]==0 or out[0]==rows-1 or out[1]==0 or out[1]==cols-1):
                if not self._extend_to_nearest_border(out):
                    all_success = False
        
        return all_success

    def _extend_to_nearest_border(self, node):
        """
        Finds a path from a node to the nearest border and marks the destination as Output.
        Blocks ancestors to prevent cycles.

        Args:
            node (Tuple[int, int]): The node to extend.

        Returns:
            bool: True if extension was successful.
        """
        r, c = node
        rows, cols = self.qca_architecture.dim
        try: ancestors = nx.ancestors(self.placement_graph, node)
        except: ancestors = set()
            
        cands = [((0, c), abs(r)), ((rows-1, c), abs(rows-1-r)), ((r, 0), abs(c)), ((r, cols-1), abs(cols-1-c))]
        cands.sort(key=lambda x: x[1])
        
        for target, _ in cands:
            tr, tc = target
            best_t = None
            min_d = 9999
            
            if tr == 0 or tr == rows-1: 
                for j in range(cols):
                    if (tr, j) not in self.used_nodes:
                        if self.is_2dd and (tr < r or j < c): continue
                        d = abs(tr-r) + abs(j-c)
                        if d < min_d: min_d, best_t = d, (tr, j)
            else: 
                for i in range(rows):
                    if (i, tc) not in self.used_nodes:
                        if self.is_2dd and (i < r or tc < c): continue
                        d = abs(i-r) + abs(tc-c)
                        if d < min_d: min_d, best_t = d, (i, tc)
            
            if best_t:
                path = self._find_optimized_path(node, best_t, obstacles=ancestors)
                
                if path:
                    tmp = self.placement_graph.copy()
                    nx.add_path(tmp, path)
                    if not nx.is_directed_acyclic_graph(tmp):
                        continue

                    self._realize_path(path)
                    if self.placement_graph.nodes[node].get('type') == 'output':
                        self.placement_graph.nodes[node]['type'] = 'routing'
                        self.placement_graph.nodes[node].pop('name', None)
                    self.placement_graph.nodes[best_t]['type'] = 'output'
                    self.placement_graph.nodes[best_t]['name'] = f"OUT_{best_t}"
                    return True
        return False

    def _sanitize_node_types(self):
        """
        Corrects node types based on connectivity (e.g., marks routing with opcodes as operations).
        """
        for node in self.placement_graph.nodes():
            data = self.placement_graph.nodes[node]
            if data.get('type') in ['input', 'output']: continue
            ind = self.placement_graph.in_degree(node)
            if data.get('opcode') == 'op' and data.get('type') == 'routing':
                data['type'] = 'convergence' if ind > 1 else 'operation'
            if ind > 1 and data.get('type') not in ['crossover', 'convergence']:
                data['type'] = 'operation'

    def _mark_graph_outputs(self):
        """
        Identifies leaf nodes and marks them as outputs if they aren't already.
        """
        for node in self.placement_graph.nodes():
            if self.placement_graph.out_degree(node) == 0:
                if self.placement_graph.nodes[node].get('type') != 'input':
                    self.placement_graph.nodes[node]['type'] = 'output'
                    self.placement_graph.nodes[node]['name'] = f"OUT_{node}"

    def _expand_grid(self):
        """
        Expands the QCA grid dimensions if maximum expansions haven't been reached.
        """
        if self.force_static_grid or self.grid_expansions_count >= self.MAX_GRID_EXPANSIONS: return
        rows, cols = self.qca_architecture.dim
        self.qca_architecture.dim = (int(rows*1.2)+2, int(cols*1.2)+2)
        self.qca_arch_graph = self.qca_architecture.get_graph()
        if self.is_2dd: self._enforce_2dd_physics()
        self.qca_border_nodes = self.qca_architecture.get_border_nodes()
        self.grid_expansions_count += 1
        logger.debug(f"Expanded grid to {self.qca_architecture.dim}")

    def _validate_final_graph(self) -> bool:
        """
        Performs final validation on the generated graph.
        Checks for cycles, connectivity, reconvergence balance, and output placement.

        Returns:
            bool: True if the graph is valid.
        """
        if not nx.is_directed_acyclic_graph(self.placement_graph): return False
        if not nx.is_weakly_connected(self.placement_graph): return False
        
        violations = self._check_reconvergence_balance(3)
        if violations:
            logger.warning(f"{len(violations)} unbalanced nodes found.")
            return False
            
        rows, cols = self.qca_architecture.dim
        for n in self.placement_graph.nodes():
            if self.placement_graph.nodes[n].get('type') == 'output':
                r, c = n
                if not (r==0 or r==rows-1 or c==0 or c==cols-1): return False
        return True

    def _check_reconvergence_balance(self, max_skew):
        """
        Checks for path length imbalances (skew) at reconvergence points.

        Args:
            max_skew (int): Maximum allowed difference in path lengths.

        Returns:
            List[Tuple]: List of violations found.
        """
        inputs = [n for n in self.placement_graph.nodes() if self.placement_graph.nodes[n].get('type') == 'input']
        violations = []
        for node in self.placement_graph.nodes():
            if self.placement_graph.in_degree(node) > 1:
                for inp in inputs:
                    try:
                        paths = list(nx.all_simple_paths(self.placement_graph, inp, node))
                        if len(paths) > 1:
                            lens = [len(p)-1 for p in paths]
                            if max(lens) - min(lens) > max_skew:
                                violations.append((node, max(lens)-min(lens)))
                                break
                    except: pass
        return violations

    def _log_stats(self):
        """Logs statistics about the generated graph."""
        logger.debug(f"Generated. Nodes: {self.placement_graph.number_of_nodes()}")

    def find_shortest_path_to_new_node(self, source):
        """
        Finds a path to a free node for expansion (used by TreeRule).

        Args:
            source (Tuple[int, int]): The starting node coordinate.

        Returns:
            Optional[List[Tuple]]: The path to the new node, or None if no valid path found.
        """
        r, c = source
        rows, cols = self.qca_architecture.dim
        if self.is_2dd:
            r_min, r_max = r, min(rows, r+8)
            c_min, c_max = c, min(cols, c+8)
        else:
            r_min, r_max = max(0, r-6), min(rows, r+7)
            c_min, c_max = max(0, c-6), min(cols, c+7)
        targets = []
        for i in range(r_min, r_max):
            for j in range(c_min, c_max):
                if (i, j) != source and (i, j) not in self.used_nodes:
                    targets.append((i, j))
        if not targets: return None
        random.shuffle(targets)
        for t in targets[:3]:
            path = self._find_optimized_path(source, t)
            if path: return path
        return None
