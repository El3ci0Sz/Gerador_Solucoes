import os
import logging
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """
    Utility class for exporting graphs to DOT (logical/physical) and ASCII Grid formats.
    """

    @staticmethod
    def save_placement_grid(graph: nx.DiGraph, dimensions: tuple, filename: str):
        """
        Generates an ASCII grid representation of the placement.
        Prioritizes 'opcode' to correctly visualize operations hidden as routing nodes.
        """
        rows, cols = dimensions
        grid = [[" . " for _ in range(cols)] for _ in range(rows)]
        
        for node, data in graph.nodes(data=True):
            if isinstance(node, tuple) and len(node) == 2:
                r, c = node
                if 0 <= r < rows and 0 <= c < cols:
                    ntype = data.get('type', 'unknown')
                    opcode = data.get('opcode', '')
                    
                    symb = " ? "
                    
                    if ntype == 'input':
                        symb = " I "
                    elif ntype == 'output':
                        symb = "OUT"
                    elif ntype == 'operation' or (opcode == 'op' and ntype != 'routing'): 
                        symb = "[O]"
                    elif ntype == 'convergence':
                        symb = "<C>"
                    elif ntype == 'crossover':
                        symb = " X "
                    elif ntype == 'routing':
                        if opcode == 'op':
                            symb = "[b]" 
                        else:
                            symb = " + " 
                    
                    grid[r][c] = symb

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Dimensions: {rows}x{cols}\n")
            f.write(f"Legend: I=Input, OUT=Output, [O]=Logic, [b]=Buffer, <C>=Conv, +=Wire, X=Cross\n\n")
            
            f.write("    " + "".join(f"{c:^3}" for c in range(cols)) + "\n")
            f.write("    " + "---" * cols + "\n")
            
            for r in range(rows):
                line = "".join(grid[r])
                f.write(f"{r:2} |{line}|\n")
            f.write("    " + "---" * cols + "\n")

    @staticmethod
    def generate_physical_dot(graph: nx.DiGraph, dimensions: tuple, filename: str):
        """
        Generates a DOT file where nodes are pinned to their physical (row, col) coordinates.
        Uses the 'neato' layout engine.
        """
        rows, cols = dimensions
        scale = 75.0 
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("digraph PhysicalLayout {\n")
            f.write('    layout=neato;\n')
            f.write('    node [shape=rect, style="filled,rounded", fixedsize=true, width=0.85, height=0.85, fontsize=9, fontname="Arial"];\n')
            f.write('    edge [penwidth=1.5, arrowsize=0.8, color="#555555"];\n')
            f.write('    splines=false;\n') 
            
            for node, data in graph.nodes(data=True):
                if not (isinstance(node, tuple) and len(node) == 2): continue
                
                r, c = node
                node_type = data.get('type', 'unknown')
                opcode = data.get('opcode', '')
                
                fill = "white"
                font = "black"
                label_txt = f"{node_type[:3].upper()}\n({r},{c})"
                shape = "rect"

                if node_type == 'input': 
                    fill, label_txt = "#2ECC71", f"IN\n({r},{c})"
                elif node_type == 'output': 
                    fill, label_txt = "#E74C3C", f"OUT\n({r},{c})"
                elif node_type == 'operation' or (opcode == 'op' and node_type != 'routing'): 
                    fill, font, label_txt = "#3498DB", "white", f"OP\n({r},{c})"
                elif node_type == 'routing':
                    if opcode == 'op': 
                         fill, label_txt = "#F1C40F", f"BUF\n({r},{c})"
                    else:
                         fill, label_txt = "#ECF0F1", f"+\n({r},{c})"
                         shape = "circle"
                elif node_type == 'crossover':
                    fill, font, label_txt = "#9B59B6", "white", f"X\n({r},{c})"
                         
                pos_str = f"{c*scale},{-r*scale}!"
                
                f.write(f'    "{node}" [pos="{pos_str}", label="{label_txt}", fillcolor="{fill}", fontcolor="{font}", shape="{shape}"];\n')
            
            f.write("\n")
            for u, v in graph.edges():
                f.write(f'    "{u}" -> "{v}";\n')
                    
            f.write("}\n")

    @staticmethod
    def _write_dot_file(graph: nx.DiGraph, dot_filename: str):
        """Internal helper to write a graph to a .dot file with ranked layout."""
        levels = defaultdict(list)
        try:
            for node in nx.topological_sort(graph):
                level = 0
                for pred in graph.predecessors(node):
                    level = max(level, graph.nodes[pred].get('level', -1) + 1)
                graph.nodes[node]['level'] = level
                levels[level].append(node)
        except nx.NetworkXUnfeasible:
            logger.warning("Graph contains cycles; cannot calculate levels for alignment.")
            levels.clear()

        with open(dot_filename, "w", encoding="utf-8") as f:
            f.write("strict digraph {\n")
            f.write("    rankdir=TB;\n")
            for node, data in graph.nodes(data=True):
                node_name = data.get('name', str(node))
                opcode = data.get('opcode', 'op')
                f.write(f'    "{node_name}" [opcode={opcode}];\n')
            f.write("\n")
            for src, dst in sorted(graph.edges()):
                source_name = graph.nodes[src].get('name', str(src))
                dest_name = graph.nodes[dst].get('name', str(dst))
                f.write(f'    "{source_name}" -> "{dest_name}";\n')
            if levels:
                for level in sorted(levels.keys()):
                    nodes_in_level = " ".join([f'"{graph.nodes[n].get("name", str(n))}"' for n in levels[level]])
                    if len(levels[level]) > 1:
                        f.write(f"    {{ rank = same; {nodes_in_level} }}\n")
            f.write("}\n")

    @staticmethod
    def generate_custom_dot_and_image(graph: nx.DiGraph, dot_filename: str, output_image_filename: str):
        """Generates a logical .dot file and renders a PNG image."""
        if not graph or not graph.nodes:
            logger.warning("Graph is empty, no image to generate.")
            return
        
        try:
            GraphVisualizer._write_dot_file(graph, dot_filename)
        except Exception as e:
            logger.error(f"Error writing custom .dot file: {e}")
            return
        
        try:
            base_name = os.path.splitext(output_image_filename)[0]
            command = f"dot -Tpng {dot_filename} -o {base_name}.png"
            os.system(command)
        except Exception as e:
            logger.debug(f"Error generating image with Graphviz: {e}")

    @staticmethod
    def generate_dot_file_only(graph: nx.DiGraph, dot_filename: str):
        """Generates a .dot file without rendering an image."""
        if not graph or not graph.nodes:
            logger.warning("Graph is empty, no .dot file to generate.")
            return
        
        try:
            GraphVisualizer._write_dot_file(graph, dot_filename)
        except Exception as e:
            logger.error(f"Error writing .dot file: {e}")