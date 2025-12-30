import networkx as nx
import logging
import random
import heapq
from typing import Optional, List, Tuple, Dict, Set  # Import adicionado

logger = logging.getLogger(__name__)

class QcaLayeredBalancedGenerator:
    """
    Shielded Hybrid QCA Generator.

    Features:
    1. Layered structure with configurable random obstacles.
    2. Pathfinding (A*) with snaking (detours) for clock synchronization.
    3. Strict Validation: Discards graphs with cycles or timing imbalances.
    """
    
    def __init__(self, qca_architecture, num_inputs: int, target_depth: int = None, obstacle_intensity: float = 0.15):
        """
        Initializes the generator configuration.

        Args:
            qca_architecture: Object containing architecture dimensions (rows, cols).
            num_inputs (int): Number of primary input nodes.
            target_depth (int, optional): Target depth (kept for interface compatibility).
            obstacle_intensity (float): Fraction of the grid to be filled with obstacles (0.0 to 1.0).
        """
        self.num_inputs = num_inputs
        self.obstacle_intensity = obstacle_intensity
        
        if hasattr(qca_architecture, 'dimensions'):
            self.max_rows, self.max_cols = qca_architecture.dimensions
        else:
            self.max_rows, self.max_cols = (40, 40)

        self.graph = nx.DiGraph()
        self.occupied = set()
        self.obstacles = set()
        self.node_counter = 0

        self.Y_SPACING = 6 
        self.X_GATE_SPACING = 8 

    def generate(self, max_retries=50) -> nx.DiGraph:
        """
        Attempts to generate a valid, balanced QCA graph multiple times.

        Args:
            max_retries (int): Maximum number of generation attempts.

        Returns:
            nx.DiGraph: The generated QCA graph if successful, or an empty graph on failure.
        """
        for attempt in range(1, max_retries + 1):
            self.graph.clear()
            self.occupied.clear()
            self.obstacles.clear()
            self.node_counter = 0
            
            logger.debug(f"Attempt {attempt}/{max_retries} - Generating with {self.obstacle_intensity*100:.1f}% obstacles...")

            success = self._try_generate_once()
            
            if success:
                is_acyclic = nx.is_directed_acyclic_graph(self.graph)
                is_balanced, balance_msg = self._check_balance()
                
                if is_acyclic and is_balanced:
                    logger.debug(f"Success on attempt {attempt}! Graph is Valid and Balanced.")
                    return self.graph
                else:
                    logger.debug(f"Attempt {attempt} failed validation. Cycles: {not is_acyclic}, Balance: {balance_msg}")
            else:
                logger.debug(f"Attempt {attempt} failed routing.")

        logger.error("Critical Failure: Could not generate a valid graph after all attempts.")
        return nx.DiGraph()

    def _try_generate_once(self) -> bool:
        """
        Executes a single generation pass, attempting to place inputs and route the logic tree.

        Returns:
            bool: True if generation was successful, False otherwise.
        """
        try:
            self._inject_obstacles(self.max_cols, self.max_rows)

            active_tips = self._place_inputs()
            if not active_tips: return False
            
            while len(active_tips) > 1:
                active_tips.sort(key=lambda x: x['row'])
                tip_a = active_tips.pop(0)
                tip_b = active_tips.pop(0)
                
                new_tip = self._merge_nodes(tip_a, tip_b)
                if not new_tip: return False
                active_tips.append(new_tip)

            if active_tips:
                self._create_output(active_tips[0])
                return True
            return False
            
        except Exception as e:
            logger.debug(f"Internal generation error: {e}")
            return False

    def _check_balance(self) -> Tuple[bool, str]:
        """
        Verifies if all paths from inputs to the output have the exact same length (clock synchronization).

        Returns:
            tuple: (bool, str) indicating success/failure and a status message.
        """
        inputs = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'input']
        outputs = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'output']
        
        if not outputs: return False, "No Output"
        output_node = outputs[0]
        
        path_lengths = set()
        
        for inp in inputs:
            try:
                paths = list(nx.all_simple_paths(self.graph, inp, output_node))
                if not paths: return False, f"Disconnected input {inp}"
                for p in paths:
                    path_lengths.add(len(p) - 1)
            except Exception:
                return False, "Error calculating paths"

        if len(path_lengths) == 1:
            return True, f"Balanced (Length: {list(path_lengths)[0]})"
        else:
            return False, f"Unbalanced! Lengths: {path_lengths}"

    def _inject_obstacles(self, w: int, h: int):
        """
        Injects random obstacles into the grid based on the configured intensity.

        Args:
            w (int): Width of the grid.
            h (int): Height of the grid.
        """
        count = int(w * h * self.obstacle_intensity)
        attempts = 0
        added = 0
        while added < count and attempts < count * 5:
            r = random.randint(0, h - 1)
            c = random.randint(2, w - 1) 
            node = (r, c)
            if node not in self.occupied:
                self.occupied.add(node)
                self.obstacles.add(node)
                added += 1
            attempts += 1

    def _place_inputs(self) -> List[Dict]:
        """
        Places input nodes vertically centered on the left edge of the grid.

        Returns:
            list: A list of dictionaries containing input node metadata.
        """
        tips = []
        total_height_needed = (self.num_inputs - 1) * self.Y_SPACING
        start_y = (self.max_rows - total_height_needed) // 2
        if start_y < 0: start_y = 0

        for i in range(self.num_inputs):
            row = start_y + (i * self.Y_SPACING)
            col = 0
            if row >= self.max_rows: continue

            node_id = (row, col)
            if node_id in self.obstacles:
                self.obstacles.remove(node_id)
                self.occupied.remove(node_id)
                
            self._add_node(node_id, 'input', f"IN_{i+1}")
            tips.append({'id': node_id, 'row': row, 'col': col, 'clock': 0})
        return tips

    def _merge_nodes(self, tip_a: dict, tip_b: dict) -> Optional[dict]:
        """
        Merges two nodes into a logic gate, routing connections and adding delay buffers if necessary.

        Args:
            tip_a (dict): Metadata of the first source node.
            tip_b (dict): Metadata of the second source node.

        Returns:
            dict: Metadata of the newly created operation node, or None if merging failed.
        """
        target_row = (tip_a['row'] + tip_b['row']) // 2
        base_x = max(tip_a['col'], tip_b['col']) + self.X_GATE_SPACING
        
        attempts = 0
        while ((target_row, base_x) in self.obstacles or base_x >= self.max_cols) and attempts < 20:
            base_x += 1
            attempts += 1
            
        target_col = base_x
        target_id = (target_row, target_col)
        
        path_a_pure = self._astar(tip_a['id'], target_id)
        path_b_pure = self._astar(tip_b['id'], target_id)
        
        if not path_a_pure or not path_b_pure: return None

        len_a = len(path_a_pure) - 1
        len_b = len(path_b_pure) - 1
        
        arrival_a = tip_a['clock'] + len_a
        arrival_b = tip_b['clock'] + len_b
        sync_clock = max(arrival_a, arrival_b)
        
        delay_a = sync_clock - arrival_a
        delay_b = sync_clock - arrival_b
        
        if not self._apply_path_with_delay(path_a_pure, delay_a): return None
        if not self._apply_path_with_delay(path_b_pure, delay_b): return None
        
        op_name = f"OP_{self.node_counter}"
        self.node_counter += 1
        self._add_node(target_id, 'operation', op_name)
        
        return {'id': target_id, 'row': target_row, 'col': target_col, 'clock': sync_clock + 1}

    def _astar(self, start: tuple, goal: tuple) -> Optional[List]:
        """
        Implements the A* pathfinding algorithm to find the shortest path between two points.

        Args:
            start (tuple): Starting coordinate (row, col).
            goal (tuple): Target coordinate (row, col).

        Returns:
            list: A list of coordinates representing the path, or None if no path is found.
        """
        def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        cost_so_far = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_node = (current[0]+dx, current[1]+dy)
                if not (0 <= next_node[0] < self.max_rows): continue
                if not (0 <= next_node[1] < self.max_cols): continue
                if next_node in self.obstacles and next_node != goal: continue
                if next_node in self.occupied and next_node != goal: continue
                    
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + h(next_node, goal)
                    heapq.heappush(open_set, (priority, next_node))
                    came_from[next_node] = current
        return None

    def _apply_path_with_delay(self, path: List, delay: int) -> bool:
        """
        Routes a connection along a path, inserting detours (snaking) to match a specific delay.

        Args:
            path (list): The base path coordinates.
            delay (int): The number of extra clock cycles (length) to add.

        Returns:
            bool: True if the path was successfully applied with delay, False otherwise.
        """
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            step_done = False
            
            if delay >= 2:
                dr, dc = v[0]-u[0], v[1]-u[1]
                is_vertical = (dr == 0)
                check_dirs = [-1, 1] if is_vertical else [1, -1]
                
                for d in check_dirs:
                    if self._try_create_detour_on_path(u, v, d, is_vertical):
                        delay -= 2
                        step_done = True
                        break 
            
            if not step_done:
                self._connect(u, v)
        return True

    def _try_create_detour_on_path(self, u: tuple, v: tuple, direction: int, is_vertical: bool) -> bool:
        """
        Attempts to create a physical detour (loop) to consume clock cycles.

        Args:
            u (tuple): Current node coordinate.
            v (tuple): Next node coordinate in the path.
            direction (int): Direction multiplier for the detour.
            is_vertical (bool): Flag indicating if the movement is vertical.

        Returns:
            bool: True if the detour was created successfully, False otherwise.
        """
        r, c = u
        if is_vertical:
            m1 = (r + direction, c)
            m2 = (r + direction, c + (1 if v[1]>c else -1))
        else:
            m1 = (r, c + direction)
            m2 = (r + (1 if v[0]>r else -1), c + direction)
            
        if not (0 <= m1[0] < self.max_rows and 0 <= m1[1] < self.max_cols): return False
        if not (0 <= m2[0] < self.max_rows and 0 <= m2[1] < self.max_cols): return False
        
        if (m1 in self.occupied or m1 in self.obstacles or m2 in self.occupied or m2 in self.obstacles):
            return False
            
        self._add_node(m1, 'buffer', f"det_{self.node_counter}_1")
        self._add_node(m2, 'buffer', f"det_{self.node_counter}_2")
        self.node_counter += 1
        self._connect(u, m1)
        self._connect(m1, m2)
        self._connect(m2, v)
        return True

    def _create_output(self, final_tip: dict):
        """
        Routes the final logic node to the output port on the rightmost edge.

        Args:
            final_tip (dict): Metadata of the final logic node.
        """
        start = final_tip['id']
        target_col = self.max_cols - 1
        target_pos = (start[0], target_col) 
        path = self._astar(start, target_pos)
        if not path:
             target_pos = (start[0] + 2, target_col) 
             path = self._astar(start, target_pos)
        
        if path:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if i == len(path) - 2: 
                     self._add_node(v, 'output', "OUTPUT")
                else:
                     if v not in self.graph:
                        self._add_node(v, 'buffer', f"out_{v[0]}_{v[1]}")
                self._connect(u, v)

    def _add_node(self, node_id: tuple, ntype: str, name: str, opcode: str = 'op'):
        """
        Adds a node to the graph and marks its position as occupied.

        Args:
            node_id (tuple): Coordinate (row, col) of the node.
            ntype (str): Type of the node (e.g., 'input', 'buffer', 'operation').
            name (str): Label for the node.
            opcode (str): Operation code (default: 'op').
        """
        if node_id not in self.graph:
            self.graph.add_node(node_id, type=ntype, name=name, opcode=opcode)
            self.occupied.add(node_id)

    def _connect(self, u: tuple, v: tuple):
        """
        Connects two nodes in the graph, creating intermediate wire buffers if necessary.

        Args:
            u (tuple): Source node coordinate.
            v (tuple): Destination node coordinate.
        """
        if u != v:
            if v not in self.graph: 
                self._add_node(v, 'buffer', f"w_{v[0]}_{v[1]}")
            self.graph.add_edge(u, v)