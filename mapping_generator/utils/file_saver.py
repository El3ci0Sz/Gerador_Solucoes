import os
import json
import re
import logging
import networkx as nx
from typing import Dict, Optional, Tuple, Any
from .visualizer import GraphVisualizer

logger = logging.getLogger(__name__)

class OutputPathManager:
    """
    Manages file naming conventions and directory structures.
    Centralizes naming rules to ensure consistency across the project.
    """
    
    BIT_TO_NAME = {
        '1000': 'mesh',
        '1001': 'mesh-toroidal',
        '1111': 'all',
        '1010': 'mesh-onehop'
    }
    
    @staticmethod
    def build_subdirs(
        tec_name: str,
        gen_mode: str,
        difficulty: Optional[int | str] = None,
        interconnect: Optional[str] = None,
        arch_size: Optional[Tuple[int, int]] = None,
        num_nodes: Optional[int] = None,
        qca_arch_type: Optional[str] = None
    ) -> str:
        """
        Constructs the subdirectory path based on generation parameters.
        
        Args:
            tec_name (str): The technology name (e.g., 'QCA' or 'CGRA').
            gen_mode (str): The generation strategy used.
            difficulty (Optional[int | str]): The difficulty level of the generation.
            interconnect (Optional[str]): The interconnection architecture type.
            arch_size (Optional[Tuple[int, int]]): The grid dimensions (rows, cols).
            num_nodes (Optional[int]): The total number of nodes in the graph.
            qca_arch_type (Optional[str]): The QCA clock zone architecture type.
            
        Returns:
            str: The relative path for the output directory.
        """
        subdirs = []
        
        if tec_name == "QCA":
            base_folder = f"mappings_qca_{gen_mode}"
            subdirs.append(base_folder)
            
            if qca_arch_type:
                subdirs.append(qca_arch_type)
            if arch_size:
                subdirs.append(f"{arch_size[0]}x{arch_size[1]}")
            if num_nodes:
                subdirs.append(f"{num_nodes}_nodes")
        
        elif tec_name == "CGRA":
            if gen_mode == 'grammar':
                if difficulty == 'random':
                    base_folder = "grammar_random_difficulty"
                elif difficulty == 'smart_random':
                    base_folder = "grammar_smart_random"
                elif isinstance(difficulty, int):
                    base_folder = "grammar_systematic_difficulty"
                else:
                    base_folder = f"mappings_cgra_{gen_mode}"
            else:
                base_folder = f"mappings_cgra_{gen_mode}"
            
            subdirs.append(base_folder)
            
            if interconnect:
                subdirs.append(interconnect)
            if arch_size:
                subdirs.append(f"{arch_size[0]}x{arch_size[1]}")
            if num_nodes:
                subdirs.append(f"{num_nodes}_nodes")
        
        return os.path.join(*subdirs)
    
    @staticmethod
    def build_filename(
        tec_name: str,
        arch_size: Tuple[int, int],
        num_nodes: int,
        num_edges: int,
        difficulty: int | str,
        index: int,
        is_fallback: bool = False
    ) -> str:
        """
        Generates a standardized filename for the output files.
        
        Args:
            tec_name (str): The technology name.
            arch_size (Tuple[int, int]): The grid dimensions (rows, cols).
            num_nodes (int): The total number of nodes in the graph.
            num_edges (int): The total number of edges in the graph.
            difficulty (int | str): The difficulty level or constraint identifier.
            index (int): The unique index of the generated graph.
            is_fallback (bool): Indicates if the file was generated using a fallback strategy.
            
        Returns:
            str: The formatted filename without extension.
        """
        arch_str = f"{arch_size[0]}x{arch_size[1]}"
        
        if is_fallback and isinstance(difficulty, int):
            diff_str = f"{difficulty}fb"
        else:
            diff_str = str(difficulty)
        
        return (
            f"{tec_name.lower()}_map_"
            f"diff{diff_str}_{arch_str}_"
            f"N{num_nodes}_E{num_edges}_{index}"
        )
    
    @staticmethod
    def get_interconnect_name(bits: str) -> str:
        """
        Returns the interconnection name based on the provided bitmask.
        
        Args:
            bits (str): The bitmask representing the interconnection type.
            
        Returns:
            str: The mapped interconnection name or 'custom'.
        """
        return OutputPathManager.BIT_TO_NAME.get(bits, "custom")
    
    @staticmethod
    def build_metadata(
        tec_name: str,
        num_nodes: int,
        num_edges: int,
        arch_size: Tuple[int, int],
        gen_mode: str,
        difficulty: Optional[int | str] = None,
        recipe: Optional[Dict] = None,
        alpha: Optional[float] = None,
        ii: Optional[int] = None,
        bits: Optional[str] = None,
        interconnect_name: Optional[str] = None,
        qca_arch_type: Optional[str] = None,
        metrics: Optional[Dict] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """
        Constructs a comprehensive metadata dictionary for the JSON output.
        
        Args:
            tec_name (str): The technology name.
            num_nodes (int): The total number of nodes.
            num_edges (int): The total number of edges.
            arch_size (Tuple[int, int]): The grid dimensions.
            gen_mode (str): The generation strategy.
            difficulty (Optional[int | str]): The difficulty level.
            recipe (Optional[Dict]): The generation recipe constraints.
            alpha (Optional[float]): The alpha probability for random edges.
            ii (Optional[int]): The initiation interval.
            bits (Optional[str]): The CGRA interconnection bitmask.
            interconnect_name (Optional[str]): The mapped interconnection name.
            qca_arch_type (Optional[str]): The QCA architecture type.
            metrics (Optional[Dict]): Collected graph metrics and statistics.
            **extra_fields: Additional fields to append to the metadata.
            
        Returns:
            Dict[str, Any]: The structured metadata dictionary.
        """
        metadata = {
            'tec_name': tec_name,
            'tec': tec_name.lower(),
            'technology': tec_name,
            'gen_mode': gen_mode,
            'mode': gen_mode,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'arch_size': list(arch_size),
            'architecture_dimensions': f"{arch_size[0]}x{arch_size[1]}"
        }
        
        graph_properties = {
            "node_count": num_nodes,
            "edge_count": num_edges
        }
        if tec_name == "CGRA" and ii is not None:
            graph_properties["II"] = ii
            metadata['ii'] = ii
        metadata['graph_properties'] = graph_properties
        
        generation_properties = {}
        if difficulty is not None:
            generation_properties["difficulty"] = difficulty
            metadata['difficulty'] = difficulty
            if isinstance(difficulty, int) and recipe:
                generation_properties["recipe"] = recipe
                metadata['recipe'] = recipe
        
        if alpha is not None:
            generation_properties["alpha"] = alpha
            metadata['alpha'] = alpha
        
        if generation_properties:
            metadata['generation_properties'] = generation_properties
        
        arch_props = {
            "type": tec_name,
            "dimensions": list(arch_size)
        }
        
        if tec_name == "CGRA":
            if bits:
                arch_props["interconnections"] = {
                    "mesh": bool(int(bits[0])),
                    "diagonal": bool(int(bits[1])),
                    "one_hop": bool(int(bits[2])),
                    "toroidal": bool(int(bits[3]))
                }
                metadata['bits'] = bits
            if interconnect_name:
                arch_props["interconnection_name"] = interconnect_name
                metadata['interconnect_name'] = interconnect_name
        
        if tec_name == "QCA" and qca_arch_type:
            arch_props["qca_arch_type"] = qca_arch_type
            metadata['qca_arch_type'] = qca_arch_type
        
        metadata['architecture_properties'] = arch_props
        
        if metrics:
            metadata['metrics'] = metrics
        
        metadata.update(extra_fields)
        return metadata


class FileSaver:
    """
    Handles saving generated graphs in various formats including DOT, JSON, and PNG.
    """
    
    def __init__(self, output_dir: str, no_images: bool = False):
        """
        Initializes the FileSaver instance.
        
        Args:
            output_dir (str): Base directory for saving files.
            no_images (bool): Flag to disable PNG image generation.
        """
        self.output_dir = output_dir
        self.no_images = no_images
        os.makedirs(output_dir, exist_ok=True)
    
    def save_graph(
        self,
        graph: nx.DiGraph,
        filename_base: str,
        metadata: Dict[str, Any],
        subdirs: str
    ) -> Dict[str, Optional[str]]:
        """
        Saves the generated graph to disk in the required formats.
        
        Args:
            graph (nx.DiGraph): The graph to save.
            filename_base (str): The base filename without extension.
            metadata (Dict[str, Any]): The metadata dictionary for the JSON file.
            subdirs (str): The relative subdirectory path to save the files into.
            
        Returns:
            Dict[str, Optional[str]]: A dictionary containing paths to the saved files.
        """
        full_dir = os.path.join(self.output_dir, subdirs)
        os.makedirs(full_dir, exist_ok=True)
        
        path_base = os.path.join(full_dir, filename_base)
        dot_path = f"{path_base}.dot"
        json_path = f"{path_base}.json"
        png_path = f"{path_base}.png" if not self.no_images else None
        
        paths = {}
        
        try:
            if self.no_images:
                GraphVisualizer.generate_dot_file_only(graph, dot_path)
            else:
                GraphVisualizer.generate_custom_dot_and_image(graph, dot_path, path_base)
            paths['dot'] = dot_path
            
            self._save_json(graph, json_path, metadata)
            paths['json'] = json_path
            paths['png'] = png_path
            
            logger.debug(f"Graph saved: {filename_base}")
            return paths
            
        except Exception as e:
            logger.error(f"Error saving graph {filename_base}: {e}", exc_info=True)
            raise
    
    def _save_json(self, graph: nx.DiGraph, json_path: str, metadata: Dict[str, Any]):
        """
        Saves the graph structure and metadata to a structured JSON file.
        
        Args:
            graph (nx.DiGraph): The graph instance.
            json_path (str): The complete file path where the JSON will be saved.
            metadata (Dict[str, Any]): The metadata to embed in the JSON.
        """
        placement = {}
        for node, data in graph.nodes(data=True):
            node_name = data.get('name', str(node))
            if isinstance(node, tuple) and len(node) >= 2:
                placement[node_name] = list(node)
            else:
                placement[node_name] = [0, 0]
        
        edges = []
        for src, dst in graph.edges():
            src_name = graph.nodes[src].get('name', str(src))
            dst_name = graph.nodes[dst].get('name', str(dst))
            edges.append([src_name, dst_name])
        
        json_data = {
            'graph_name': os.path.basename(json_path).replace('.json', ''),
            'metadata': metadata,
            'placement': placement,
            'edges': edges
        }
        
        if metadata.get('graph_properties'):
            json_data['graph_properties'] = metadata['graph_properties']
            
        if metadata.get('architecture_properties'):
            json_data['architecture_properties'] = metadata['architecture_properties']
            
        if 'generation_properties' in metadata:
            json_data['generation_properties'] = metadata['generation_properties']
        
        pretty_json = json.dumps(json_data, indent=2)
        compacted_json = self._compact_coordinates(pretty_json)
        
        with open(json_path, 'w') as f:
            f.write(compacted_json)
    
    @staticmethod
    def _compact_coordinates(json_string: str) -> str:
        """
        Compacts array representations in the JSON string for better readability.
        
        Args:
            json_string (str): The raw JSON string.
            
        Returns:
            str: The formatted JSON string with compacted coordinates.
        """
        return re.sub(
            r'\[\s*(-?\d+),\s*(-?\d+)(,\s*-?\d+)?\s*\]',
            lambda m: '[' + ', '.join(re.split(r',\s*', m.group(0)[1:-1])) + ']',
            json_string
        )
