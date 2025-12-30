# mapping_generator/architectures/qca.py

import networkx as nx
import logging
from typing import Optional, Tuple, Set

logger = logging.getLogger(__name__)

class QCA:
    """
    Generates connectivity graphs for QCA architectures.
    
    CORREÇÃO: expand_grid() agora funciona corretamente e sempre invalida o cache do grafo.
    """
    
    def __init__(self, dimensions: Optional[tuple] = None, arch_type: str = 'U', dim: Optional[tuple] = None):
        """
        Initializes the QCA architecture generator.

        Args:
            dimensions (tuple): The (rows, cols) dimensions.
            arch_type (str): The clock zone scheme ('U', 'T', 'R').
            dim (tuple, optional): Alias for dimensions for backward compatibility.
        """
        if dimensions is None and dim is not None:
            dimensions = dim
        if dimensions is None:
            raise ValueError("Dimensions must be provided via 'dimensions' or 'dim'.")
            
        self.dim = dimensions
        self.arch_type = arch_type.upper()
        self.USE_CLOCK_TILE = [[1, 2, 3, 4], [4, 3, 2, 1], [3, 4, 1, 2], [2, 1, 4, 3]]
        self.RES_CLOCK_TILE = [[4, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2]]
        self._graph = None
        self._clock_scheme = arch_type  # Alias para compatibilidade

    def get_graph(self) -> nx.DiGraph:
        """Returns the generated QCA connectivity graph, creating it if necessary."""
        if self._graph is None:
            self._graph = self._generate_connectivity_graph()
        return self._graph

    def expand_grid(self, new_size: int):
        """
        Expands the grid to the specified size.
        
        CORREÇÃO: Agora aceita o tamanho do lado (new_size x new_size)
        ao invés de número total de células, e sempre invalida o cache.
        
        Args:
            new_size (int): The new dimension (will create new_size x new_size grid).
        """
        old_rows, old_cols = self.dim
        old_area = old_rows * old_cols
        
        # CORREÇÃO 1: Interpretar new_size como dimensão do lado, não área total
        # Se new_size < max(old_rows, old_cols), não faz sentido "expandir"
        max_current = max(old_rows, old_cols)
        
        if new_size <= max_current:
            logger.debug(f"QCA.expand_grid: new_size {new_size} <= current max dim {max_current}, no expansion needed")
            return
        
        # CORREÇÃO 2: Criar grid quadrado com o novo tamanho
        new_rows, new_cols = new_size, new_size
        new_area = new_rows * new_cols
        
        logger.info(f"Expanding QCA grid from {old_rows}x{old_cols} (area={old_area}) to {new_rows}x{new_cols} (area={new_area})")
        
        # CORREÇÃO 3: Atualizar dimensões
        self.dim = (new_rows, new_cols)
        
        # CORREÇÃO 4: SEMPRE invalidar o cache do grafo
        self._graph = None
        
        logger.debug(f"QCA grid expanded successfully. New dimensions: {self.dim}")

    def get_border_nodes(self) -> Set[Tuple[int, int]]:
        """
        Identifies nodes located on the edges of the grid.

        Returns:
            Set[Tuple[int, int]]: A set of coordinates (row, col) for border nodes.
        """
        rows, cols = self.dim
        borders = set()
        if rows < 2 or cols < 2:
            return set([(r, c) for r in range(rows) for c in range(cols)])
            
        for r in range(rows):
            borders.add((r, 0))
            borders.add((r, cols - 1))
        for c in range(cols):
            borders.add((0, c))
            borders.add((rows - 1, c))
        return borders

    def _generate_connectivity_graph(self) -> nx.DiGraph:
        """Generates the connectivity graph based on architecture type."""
        graph = nx.DiGraph()
        all_nodes = [(r, c) for r in range(self.dim[0]) for c in range(self.dim[1])]
        
        for node in all_nodes:
            valid_neighbors = set()
            if self.arch_type == "U":
                valid_neighbors = self._get_neighbors_by_clock_flow(node, self._get_use_clock_zone)
            elif self.arch_type == "T":
                valid_neighbors = self._get_neighbors_2ddwave(node)
            elif self.arch_type == "R": 
                valid_neighbors = self._get_neighbors_by_clock_flow(node, self._get_res_clock_zone)
            else:
                raise ValueError(f"Unknown architecture type: {self.arch_type}")
                
            for neighbor in valid_neighbors:
                graph.add_edge(node, neighbor)
        
        logger.debug(f"Generated QCA connectivity graph: {len(all_nodes)} nodes, {graph.number_of_edges()} edges")
        return graph

    def _is_valid_node(self, node: tuple) -> bool:
        r, c = node
        return 0 <= r < self.dim[0] and 0 <= c < self.dim[1]
    
    def _get_use_clock_zone(self, node: tuple) -> int:
        r, c = node
        return self.USE_CLOCK_TILE[r % 4][c % 4]

    def _get_res_clock_zone(self, node: tuple) -> int:
        r, c = node
        return self.RES_CLOCK_TILE[r % 4][c % 4]
    
    def _get_neighbors_by_clock_flow(self, node: tuple, clock_zone_func) -> set:
        r, c = node
        source_zone = clock_zone_func(node)
        target_zone = (source_zone % 4) + 1
        potential_neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = set()
        
        for neighbor in potential_neighbors:
            if self._is_valid_node(neighbor) and clock_zone_func(neighbor) == target_zone:
                valid_neighbors.add(neighbor)
        return valid_neighbors

    def _get_neighbors_2ddwave(self, node: tuple) -> set:
        """
        2DDWave connectivity: only forward propagation (right and down).
        """
        r, c = node
        potential_neighbors = [(r, c + 1), (r + 1, c)]
        return {n for n in potential_neighbors if self._is_valid_node(n)}
    
    # Propriedade para compatibilidade
    @property
    def clock_scheme(self):
        """Returns the architecture type for compatibility."""
        return self.arch_type