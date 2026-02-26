import random
import networkx as nx
import logging
from typing import Optional, List, Tuple, Set

from ...architectures.qca import QCA

logger = logging.getLogger(__name__)

class QcaBackwardsGenerator:
    
    def __init__(self, qca_architecture: QCA, target_gates: int, num_outputs: int = 1):
        """
        Initializes the Backwards QCA Generator.
        
        Args:
            qca_architecture (QCA): The initialized QCA architecture instance.
            target_gates (int): The minimum acceptable number of logic gates after pruning.
            num_outputs (int): The exact number of outputs required on the border.
        """
        self.qca_architecture = qca_architecture
        self.target_gates = target_gates
        self.num_outputs = num_outputs
        
        self.internal_generation_target = int(target_gates * 2.5) 
        
        self.placement_graph = nx.DiGraph()
        self.used_nodes: Set[Tuple[int, int]] = set()
        self.border_nodes = self.qca_architecture.get_border_nodes()
        self.arch_graph = None 
        
        self.node_depth = {}

    def generate(self) -> Optional[nx.DiGraph]:
        """
        Executes the backwards generation process including growth, rescue routing, 
        pruning, and strict validation.
        
        Returns:
            Optional[nx.DiGraph]: The generated valid placement graph, or None if 
            the attempt fails to meet the strict constraints.
        """
        self.placement_graph.clear()
        self.used_nodes.clear()
        self.node_depth.clear()
        self.arch_graph = self.qca_architecture.get_graph()
        
        frontier = []
        candidates = list(self.border_nodes)
        random.shuffle(candidates)
        
        outputs_placed = 0
        for node in candidates:
            if outputs_placed >= self.num_outputs: break
            preds = list(self.arch_graph.predecessors(node))
            if not preds: continue
            
            self.placement_graph.add_node(node, type='output', name=f"OUT_{node}")
            self.used_nodes.add(node)
            self.node_depth[node] = 0 
            frontier.append(node)
            outputs_placed += 1
            
        if outputs_placed != self.num_outputs:
            return None
            
        current_gates = 0
        max_iter = self.internal_generation_target * 40
        iter_count = 0
        
        while iter_count < max_iter and frontier:
            iter_count += 1
            
            non_border = [n for n in frontier if n not in self.border_nodes]
            current_node = random.choice(non_border) if non_border and random.random() < 0.7 else random.choice(frontier)
            frontier.remove(current_node)

            is_border = current_node in self.border_nodes
            wants_more_gates = current_gates < self.internal_generation_target
            
            if is_border and self.placement_graph.nodes[current_node].get('type') != 'output':
                if not wants_more_gates or random.random() < 0.2:
                    self._finalize_as_input(current_node)
                    continue

            if random.random() < 0.6:
                if self._try_share_predecessor(current_node):
                    continue 

            success = False
            if wants_more_gates and random.random() < 0.8:
                preds = self._add_gate_logic(current_node)
                if preds:
                    frontier.extend(preds)
                    current_gates += 1
                    success = True
            
            if not success:
                pred = self._add_wire_logic(current_node)
                if pred:
                    frontier.append(pred)
                else:
                    if is_border and self.placement_graph.nodes[current_node].get('type') != 'output':
                        self._finalize_as_input(current_node)
                    else:
                        self.placement_graph.nodes[current_node]['stuck'] = True

        leaves = [n for n in self.placement_graph.nodes() 
                 if self.placement_graph.in_degree(n) == 0 
                 and self.placement_graph.nodes[n].get('type') != 'input']
        
        for node in leaves:
            self._rescue_route_to_border(node)

        self._prune_dead_branches()
        
        if not nx.is_directed_acyclic_graph(self.placement_graph):
            return None

        if not nx.is_weakly_connected(self.placement_graph):
            return None

        final_outputs = [n for n, d in self.placement_graph.nodes(data=True) if d.get('type') == 'output']
        final_inputs = [n for n, d in self.placement_graph.nodes(data=True) if d.get('type') == 'input']
        final_gates = len([n for n, d in self.placement_graph.nodes(data=True) if d.get('type') == 'operation'])
        
        if len(final_outputs) != self.num_outputs:
            return None
            
        if len(final_inputs) == 0:
            return None
            
        if final_gates < self.target_gates:
            return None

        return self.placement_graph


    def _try_share_predecessor(self, node: Tuple[int, int]) -> bool:
        """
        Attempts to merge the current path with an existing active node to create a fan-out.
        Strictly enforces clock path depth equality for synchronization.
        
        Args:
            node (Tuple[int, int]): The current node coordinates.
            
        Returns:
            bool: True if the merge was successful, False otherwise.
        """
        if node not in self.arch_graph: return False
        preds = list(self.arch_graph.predecessors(node))
        random.shuffle(preds)
        
        for p in preds:
            if p in self.used_nodes and self.placement_graph.has_node(p):
                expected_depth = self.node_depth.get(node, 0) + 1
                if self.node_depth.get(p) != expected_depth:
                    continue 
                    
                if nx.has_path(self.placement_graph, node, p):
                    continue 
                    
                self.placement_graph.add_edge(p, node)
                return True
        return False

    def _add_gate_logic(self, node: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Transforms the given node into a logic gate by branching backwards to two valid predecessors.
        
        Args:
            node (Tuple[int, int]): The node to be converted into a gate.
            
        Returns:
            Optional[List[Tuple[int, int]]]: A list of the two selected predecessors if successful, None otherwise.
        """
        preds = self._get_valid_predecessors(node)
        if len(preds) < 2: return None
        selected = random.sample(preds, 2)
        
        self.placement_graph.nodes[node]['type'] = 'operation'
        self.placement_graph.nodes[node]['name'] = f"op_{node}"
        
        current_depth = self.node_depth.get(node, 0)
        for p in selected:
            self.placement_graph.add_node(p, type='routing', name=f"rout_{p}")
            self.placement_graph.add_edge(p, node)
            self.used_nodes.add(p)
            self.node_depth[p] = current_depth + 1 
            
        return selected

    def _add_wire_logic(self, node: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Extends a single routing wire backwards to a valid predecessor.
        
        Args:
            node (Tuple[int, int]): The current node coordinates.
            
        Returns:
            Optional[Tuple[int, int]]: The selected predecessor node if successful, None otherwise.
        """
        preds = self._get_valid_predecessors(node)
        if not preds: return None
        selected = random.choice(preds)
        
        self.placement_graph.add_node(selected, type='routing', name=f"rout_{selected}")
        self.placement_graph.add_edge(selected, node)
        self.used_nodes.add(selected)
        self.node_depth[selected] = self.node_depth.get(node, 0) + 1 
        
        return selected

    def _finalize_as_input(self, node: Tuple[int, int]) -> bool:
        """
        Converts a frontier node located on the grid border into a valid circuit input.
        
        Args:
            node (Tuple[int, int]): The node located on the border.
            
        Returns:
            bool: True if the node is on the border and successfully converted, False otherwise.
        """
        if node in self.border_nodes:
            self.placement_graph.nodes[node]['type'] = 'input'
            self.placement_graph.nodes[node]['name'] = f"in_{node}"
            return True
        return False

    def _get_valid_predecessors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retrieves all valid unused predecessors from the physical architecture graph 
        (respecting clock zones).
        
        Args:
            node (Tuple[int, int]): The node coordinates.
            
        Returns:
            List[Tuple[int, int]]: A list of available predecessor coordinates.
        """
        if node not in self.arch_graph: return []
        return [p for p in self.arch_graph.predecessors(node) if p not in self.used_nodes]

    def _rescue_route_to_border(self, start_node: Tuple[int, int]) -> bool:
        """
        Performs a Breadth-First Search (BFS) to route a stuck/dead-end branch 
        to the nearest available border to serve as an input.
        
        Args:
            start_node (Tuple[int, int]): The isolated node that needs to reach the border.
            
        Returns:
            bool: True if a path to the border was successfully mapped, False otherwise.
        """
        queue = [(start_node, [start_node])]
        visited = {start_node}
        limite_busca = max(self.qca_architecture.dim) * 2 
        
        while queue:
            curr, path = queue.pop(0)
            if len(path) > limite_busca: continue 
            
            if curr in self.border_nodes and curr not in self.used_nodes:
                for i in range(len(path) - 1):
                    u, v = path[i+1], path[i]
                    ntype = 'input' if u == path[-1] else 'routing'
                    self.placement_graph.add_node(u, type=ntype, name=f"{'in' if ntype=='input' else 'rout'}_{u}")
                    self.placement_graph.add_edge(u, v)
                    self.used_nodes.add(u)
                    self.node_depth[u] = self.node_depth.get(v, 0) + 1 
                return True
                
            preds = list(self.arch_graph.predecessors(curr))
            random.shuffle(preds)
            for p in preds:
                if p not in self.used_nodes and p not in visited:
                    if not nx.has_path(self.placement_graph, start_node, p):
                        visited.add(p)
                        queue.append((p, path + [p]))
        return False

    def _prune_dead_branches(self) -> None:
        """
        Removes any node that is not reachable from a valid circuit input. 
        Adjusts broken gates (downgrading them to wires) caused by the pruning process.
        """
        valid_inputs = {n for n, d in self.placement_graph.nodes(data=True) if d.get('type') == 'input'}
        alive_nodes = set()
        queue = list(valid_inputs)
        alive_nodes.update(valid_inputs)
        
        while queue:
            curr = queue.pop(0)
            for succ in self.placement_graph.successors(curr):
                if succ not in alive_nodes:
                    alive_nodes.add(succ)
                    queue.append(succ)
                    
        nodes_to_remove = [n for n in self.placement_graph.nodes() if n not in alive_nodes]
        self.placement_graph.remove_nodes_from(nodes_to_remove)
        
        for n, data in list(self.placement_graph.nodes(data=True)):
            if data.get('type') == 'operation':
                preds = list(self.placement_graph.predecessors(n))
                if len(preds) < 2:
                    self.placement_graph.nodes[n]['type'] = 'routing'
