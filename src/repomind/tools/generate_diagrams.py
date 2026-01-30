"""Architecture and Data Flow Diagram Generator.

This tool generates visual diagrams based on semantic search results:
1. Architecture diagrams (component relationships, layers)
2. Data flow diagrams (how data moves through the system)
3. Call flow diagrams (execution paths)
4. Dependency diagrams (module/package dependencies)

Output formats:
- Mermaid (for GitHub/GitLab/Notion rendering)
- PlantUML (for IDE plugins)
- DOT (for Graphviz)
- ASCII (for terminal/plain text)

Example Usage:
    # Generate architecture diagram from semantic search
    result = generate_architecture_diagram(
        query="user authentication flow",
        repo_name="my-app",
        output_format="mermaid"
    )

    # Generate data flow diagram
    result = generate_dataflow_diagram(
        query="payment processing",
        repo_name="my-app",
        output_format="mermaid"
    )

Author: RepoMind Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..models.chunk import ChunkType
from ..services.embedding import EmbeddingService
from ..services.storage import StorageService
from ..services.call_graph import CallGraphService
from ..services.symbol_table import SymbolTableService

console = Console(width=200, force_terminal=False)


class DiagramFormat(str, Enum):
    """Supported diagram output formats."""
    MERMAID = "mermaid"
    PLANTUML = "plantuml"
    DOT = "dot"
    ASCII = "ascii"


class DiagramType(str, Enum):
    """Types of diagrams that can be generated."""
    ARCHITECTURE = "architecture"
    DATAFLOW = "dataflow"
    CALLFLOW = "callflow"
    DEPENDENCY = "dependency"
    CLASS = "class"
    SEQUENCE = "sequence"


@dataclass
class DiagramNode:
    """A node in the diagram."""
    id: str
    label: str
    node_type: str  # component, class, function, module, etc.
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DiagramEdge:
    """An edge/connection in the diagram."""
    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "default"  # calls, imports, extends, uses, etc.
    metadata: dict = field(default_factory=dict)


@dataclass
class DiagramResult:
    """Result of diagram generation."""
    diagram_type: DiagramType
    format: DiagramFormat
    content: str
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    title: str
    description: str
    statistics: dict


class DiagramGenerator:
    """Generates diagrams from code analysis results."""

    def __init__(self):
        self.storage = StorageService()
        self.call_graph = CallGraphService()
        self.symbol_table = SymbolTableService()
        self.embedding_service = EmbeddingService()

    def generate_from_search(
        self,
        query: str,
        diagram_type: DiagramType,
        output_format: DiagramFormat = DiagramFormat.MERMAID,
        repo_name: Optional[str] = None,
        max_nodes: int = 30,
        depth: int = 2,
        include_external: bool = False,
    ) -> DiagramResult:
        """Generate a diagram based on semantic search results.

        Args:
            query: Semantic search query to find relevant code
            diagram_type: Type of diagram to generate
            output_format: Output format (mermaid, plantuml, dot, ascii)
            repo_name: Filter to specific repository
            max_nodes: Maximum number of nodes in diagram
            depth: How many levels of relationships to include
            include_external: Include external dependencies

        Returns:
            DiagramResult with the generated diagram
        """
        # Step 1: Semantic search to find relevant code
        query_embedding = self.embedding_service.embed_query(query)

        search_results = self.storage.search(
            query_embedding=query_embedding,
            n_results=max_nodes,
            repo_filter=repo_name,
        )

        if not search_results:
            return DiagramResult(
                diagram_type=diagram_type,
                format=output_format,
                content="No results found for query",
                nodes=[],
                edges=[],
                title=f"Empty Diagram: {query}",
                description="No matching code found",
                statistics={"nodes": 0, "edges": 0}
            )

        # Step 2: Build graph based on diagram type
        if diagram_type == DiagramType.ARCHITECTURE:
            nodes, edges = self._build_architecture_graph(search_results, repo_name, depth)
        elif diagram_type == DiagramType.DATAFLOW:
            nodes, edges = self._build_dataflow_graph(search_results, repo_name, depth)
        elif diagram_type == DiagramType.CALLFLOW:
            nodes, edges = self._build_callflow_graph(search_results, repo_name, depth)
        elif diagram_type == DiagramType.DEPENDENCY:
            nodes, edges = self._build_dependency_graph(search_results, repo_name, depth)
        elif diagram_type == DiagramType.CLASS:
            nodes, edges = self._build_class_graph(search_results, repo_name, depth)
        elif diagram_type == DiagramType.SEQUENCE:
            nodes, edges = self._build_sequence_graph(search_results, repo_name, depth)
        else:
            nodes, edges = self._build_architecture_graph(search_results, repo_name, depth)

        # Step 3: Filter external dependencies if needed
        if not include_external:
            nodes, edges = self._filter_external(nodes, edges, repo_name)

        # Step 4: Limit nodes
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            node_ids = {n.id for n in nodes}
            edges = [e for e in edges if e.source in node_ids and e.target in node_ids]

        # Step 5: Generate diagram content
        content = self._render_diagram(nodes, edges, diagram_type, output_format, query)

        return DiagramResult(
            diagram_type=diagram_type,
            format=output_format,
            content=content,
            nodes=nodes,
            edges=edges,
            title=f"{diagram_type.value.title()} Diagram: {query}",
            description=f"Generated from semantic search for '{query}'",
            statistics={
                "nodes": len(nodes),
                "edges": len(edges),
                "search_results": len(search_results),
                "depth": depth,
            }
        )

    def _build_architecture_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build architecture graph showing components and layers."""
        nodes = []
        edges = []
        seen_nodes = set()

        # Group by module/package
        modules = defaultdict(list)
        for chunk, score in search_results:
            module = self._get_module_name(chunk.file_path)
            modules[module].append(chunk)

        # Create module nodes (components)
        for module, chunks in modules.items():
            if module not in seen_nodes:
                layer = self._detect_layer(module, chunks)
                nodes.append(DiagramNode(
                    id=self._sanitize_id(module),
                    label=module,
                    node_type="component",
                    file_path=chunks[0].file_path if chunks else None,
                    metadata={
                        "layer": layer,
                        "chunk_count": len(chunks),
                        "types": list(set(c.type.value for c in chunks))
                    }
                ))
                seen_nodes.add(module)

        # Find relationships between modules
        for module, chunks in modules.items():
            for chunk in chunks:
                qualified_name = chunk.get_qualified_name()

                # Get callees for this chunk
                callees = self.call_graph.find_callees(qualified_name, repo_name, max_depth=1)

                for callee in callees:
                    target_module = self._get_module_name(callee.caller_file)
                    if target_module != module and target_module in seen_nodes:
                        edge_id = f"{module}->{target_module}"
                        if not any(e.source == self._sanitize_id(module) and
                                  e.target == self._sanitize_id(target_module) for e in edges):
                            edges.append(DiagramEdge(
                                source=self._sanitize_id(module),
                                target=self._sanitize_id(target_module),
                                label="uses",
                                edge_type="dependency"
                            ))

        return nodes, edges

    def _build_dataflow_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build data flow graph showing how data moves through the system."""
        nodes = []
        edges = []
        seen_nodes = set()

        # Identify data sources, processors, and sinks
        for chunk, score in search_results:
            qualified_name = chunk.get_qualified_name()

            if qualified_name in seen_nodes:
                continue
            seen_nodes.add(qualified_name)

            # Determine node type based on patterns
            node_type = self._classify_dataflow_node(chunk)

            nodes.append(DiagramNode(
                id=self._sanitize_id(qualified_name),
                label=chunk.name,
                node_type=node_type,
                file_path=chunk.file_path,
                metadata={
                    "chunk_type": chunk.type.value,
                    "signature": chunk.signature,
                    "dataflow_type": node_type
                }
            ))

        # Build data flow edges based on call graph
        for chunk, score in search_results:
            qualified_name = chunk.get_qualified_name()

            # Find what this function calls
            callees = self.call_graph.find_callees(qualified_name, repo_name, max_depth=depth)

            for callee in callees:
                if callee.callee in seen_nodes:
                    # Determine data flow direction
                    label = self._infer_dataflow_label(qualified_name, callee.callee)

                    edges.append(DiagramEdge(
                        source=self._sanitize_id(qualified_name),
                        target=self._sanitize_id(callee.callee),
                        label=label,
                        edge_type="dataflow"
                    ))

        return nodes, edges

    def _build_callflow_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build call flow graph showing execution paths."""
        nodes = []
        edges = []
        seen_nodes = set()

        # Add search result nodes
        for chunk, score in search_results:
            qualified_name = chunk.get_qualified_name()

            if qualified_name in seen_nodes:
                continue
            seen_nodes.add(qualified_name)

            # Determine if entry point
            is_entry = self._is_entry_point(chunk)

            nodes.append(DiagramNode(
                id=self._sanitize_id(qualified_name),
                label=chunk.name,
                node_type="entry" if is_entry else chunk.type.value,
                file_path=chunk.file_path,
                metadata={
                    "is_entry_point": is_entry,
                    "signature": chunk.signature,
                    "line": chunk.start_line
                }
            ))

        # Build call edges with depth traversal
        for chunk, score in search_results:
            qualified_name = chunk.get_qualified_name()
            self._add_call_edges(
                qualified_name,
                nodes,
                edges,
                seen_nodes,
                repo_name,
                current_depth=0,
                max_depth=depth
            )

        return nodes, edges

    def _build_dependency_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build dependency graph showing imports and module relationships."""
        nodes = []
        edges = []
        seen_nodes = set()

        # Group by file/module
        files = defaultdict(list)
        for chunk, score in search_results:
            files[chunk.file_path].append(chunk)

        # Create file nodes
        for file_path, chunks in files.items():
            module_name = self._get_module_name(file_path)

            if module_name not in seen_nodes:
                seen_nodes.add(module_name)
                nodes.append(DiagramNode(
                    id=self._sanitize_id(module_name),
                    label=module_name.split(".")[-1],
                    node_type="module",
                    file_path=file_path,
                    metadata={
                        "full_path": file_path,
                        "symbol_count": len(chunks)
                    }
                ))

        # Find import relationships via call graph
        for file_path, chunks in files.items():
            source_module = self._get_module_name(file_path)

            for chunk in chunks:
                callees = self.call_graph.find_callees(
                    chunk.get_qualified_name(),
                    repo_name,
                    max_depth=1
                )

                for callee in callees:
                    target_module = self._get_module_name(callee.caller_file)

                    if target_module != source_module and target_module in seen_nodes:
                        # Avoid duplicate edges
                        if not any(e.source == self._sanitize_id(source_module) and
                                  e.target == self._sanitize_id(target_module) for e in edges):
                            edges.append(DiagramEdge(
                                source=self._sanitize_id(source_module),
                                target=self._sanitize_id(target_module),
                                label="imports",
                                edge_type="import"
                            ))

        return nodes, edges

    def _build_class_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build class diagram showing inheritance and composition."""
        nodes = []
        edges = []
        seen_nodes = set()

        # Filter to classes and interfaces
        class_chunks = [
            (chunk, score) for chunk, score in search_results
            if chunk.type in [ChunkType.CLASS, ChunkType.INTERFACE]
        ]

        for chunk, score in class_chunks:
            class_name = chunk.name

            if class_name in seen_nodes:
                continue
            seen_nodes.add(class_name)

            # Get methods for this class
            methods = []
            for other_chunk, _ in search_results:
                if other_chunk.parent_name == class_name:
                    methods.append(other_chunk.name)

            nodes.append(DiagramNode(
                id=self._sanitize_id(class_name),
                label=class_name,
                node_type="interface" if chunk.type == ChunkType.INTERFACE else "class",
                file_path=chunk.file_path,
                metadata={
                    "methods": methods[:10],  # Limit methods shown
                    "signature": chunk.signature,
                    "is_interface": chunk.type == ChunkType.INTERFACE
                }
            ))

            # Find parent classes/interfaces
            if chunk.inheritance:
                for parent in chunk.inheritance:
                    if parent.parent_name:
                        edges.append(DiagramEdge(
                            source=self._sanitize_id(class_name),
                            target=self._sanitize_id(parent.parent_name),
                            edge_type="extends" if parent.relation_type == "extends" else "implements"
                        ))

                        # Add parent node if not seen
                        if parent.parent_name not in seen_nodes:
                            seen_nodes.add(parent.parent_name)
                            nodes.append(DiagramNode(
                                id=self._sanitize_id(parent.parent_name),
                                label=parent.parent_name,
                                node_type="interface" if parent.relation_type == "implements" else "class",
                                metadata={"external": True}
                            ))

        return nodes, edges

    def _build_sequence_graph(
        self,
        search_results: list,
        repo_name: Optional[str],
        depth: int
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Build sequence diagram data showing call order."""
        nodes = []
        edges = []
        seen_nodes = set()
        sequence_num = 0

        # Find entry points first
        entry_points = [
            (chunk, score) for chunk, score in search_results
            if self._is_entry_point(chunk)
        ]

        if not entry_points:
            # Use highest scored result as starting point
            entry_points = search_results[:1]

        # Build sequence from entry points
        for chunk, score in entry_points:
            qualified_name = chunk.get_qualified_name()
            participant = chunk.parent_name or self._get_module_name(chunk.file_path)

            if participant not in seen_nodes:
                seen_nodes.add(participant)
                nodes.append(DiagramNode(
                    id=self._sanitize_id(participant),
                    label=participant,
                    node_type="participant",
                    file_path=chunk.file_path
                ))

            # Trace call sequence
            callees = self.call_graph.find_callees(qualified_name, repo_name, max_depth=depth)

            for callee in callees:
                sequence_num += 1
                target_participant = callee.callee.split(".")[0] if "." in callee.callee else callee.callee

                if target_participant not in seen_nodes:
                    seen_nodes.add(target_participant)
                    nodes.append(DiagramNode(
                        id=self._sanitize_id(target_participant),
                        label=target_participant,
                        node_type="participant"
                    ))

                method_name = callee.callee.split(".")[-1] if "." in callee.callee else callee.callee
                edges.append(DiagramEdge(
                    source=self._sanitize_id(participant),
                    target=self._sanitize_id(target_participant),
                    label=method_name,
                    edge_type="call",
                    metadata={"sequence": sequence_num}
                ))

        return nodes, edges

    def _add_call_edges(
        self,
        symbol: str,
        nodes: list,
        edges: list,
        seen_nodes: set,
        repo_name: Optional[str],
        current_depth: int,
        max_depth: int
    ):
        """Recursively add call edges."""
        if current_depth >= max_depth:
            return

        callees = self.call_graph.find_callees(symbol, repo_name, max_depth=1)

        for callee in callees:
            # Add edge
            edges.append(DiagramEdge(
                source=self._sanitize_id(symbol),
                target=self._sanitize_id(callee.callee),
                label="calls",
                edge_type="call",
                metadata={"line": callee.caller_line}
            ))

            # Add node if not seen
            if callee.callee not in seen_nodes:
                seen_nodes.add(callee.callee)
                nodes.append(DiagramNode(
                    id=self._sanitize_id(callee.callee),
                    label=callee.callee.split(".")[-1],
                    node_type="function",
                    file_path=callee.caller_file,
                    metadata={"depth": current_depth + 1}
                ))

                # Recurse
                self._add_call_edges(
                    callee.callee,
                    nodes,
                    edges,
                    seen_nodes,
                    repo_name,
                    current_depth + 1,
                    max_depth
                )

    def _render_diagram(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        diagram_type: DiagramType,
        output_format: DiagramFormat,
        title: str
    ) -> str:
        """Render the diagram in the specified format."""
        if output_format == DiagramFormat.MERMAID:
            return self._render_mermaid(nodes, edges, diagram_type, title)
        elif output_format == DiagramFormat.PLANTUML:
            return self._render_plantuml(nodes, edges, diagram_type, title)
        elif output_format == DiagramFormat.DOT:
            return self._render_dot(nodes, edges, diagram_type, title)
        elif output_format == DiagramFormat.ASCII:
            return self._render_ascii(nodes, edges, diagram_type, title)
        else:
            return self._render_mermaid(nodes, edges, diagram_type, title)

    def _render_mermaid(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        diagram_type: DiagramType,
        title: str
    ) -> str:
        """Render diagram in Mermaid format."""
        lines = []

        if diagram_type == DiagramType.SEQUENCE:
            lines.append("sequenceDiagram")
            lines.append(f"    %% {title}")

            # Add participants
            for node in nodes:
                lines.append(f"    participant {node.id} as {node.label}")

            # Add messages (sorted by sequence)
            sorted_edges = sorted(edges, key=lambda e: e.metadata.get("sequence", 0))
            for edge in sorted_edges:
                lines.append(f"    {edge.source}->>+{edge.target}: {edge.label or 'call'}")

        elif diagram_type == DiagramType.CLASS:
            lines.append("classDiagram")
            lines.append(f"    %% {title}")

            # Add classes with methods
            for node in nodes:
                if node.node_type == "interface":
                    lines.append(f"    class {node.id} {{")
                    lines.append(f"        <<interface>>")
                else:
                    lines.append(f"    class {node.id} {{")

                for method in node.metadata.get("methods", [])[:5]:
                    lines.append(f"        +{method}()")
                lines.append("    }")

            # Add relationships
            for edge in edges:
                if edge.edge_type == "implements":
                    lines.append(f"    {edge.target} <|.. {edge.source}")
                elif edge.edge_type == "extends":
                    lines.append(f"    {edge.target} <|-- {edge.source}")
                else:
                    lines.append(f"    {edge.source} --> {edge.target}")

        elif diagram_type == DiagramType.DATAFLOW:
            lines.append("flowchart LR")
            lines.append(f"    %% {title}")

            # Add nodes with shapes based on type
            for node in nodes:
                if node.node_type == "source":
                    lines.append(f"    {node.id}[({node.label})]")  # Database shape
                elif node.node_type == "sink":
                    lines.append(f"    {node.id}[/{node.label}/]")  # Parallelogram
                elif node.node_type == "processor":
                    lines.append(f"    {node.id}[{node.label}]")  # Rectangle
                else:
                    lines.append(f"    {node.id}({node.label})")  # Rounded

            # Add edges with labels
            for edge in edges:
                if edge.label:
                    lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")
                else:
                    lines.append(f"    {edge.source} --> {edge.target}")

        else:  # Architecture, Callflow, Dependency
            lines.append("flowchart TD")
            lines.append(f"    %% {title}")

            # Group by layer for architecture diagrams
            if diagram_type == DiagramType.ARCHITECTURE:
                layers = defaultdict(list)
                for node in nodes:
                    layer = node.metadata.get("layer", "unknown")
                    layers[layer].append(node)

                # Add subgraphs for layers
                layer_order = ["api", "controller", "service", "repository", "model", "util", "unknown"]
                for layer in layer_order:
                    if layer in layers:
                        lines.append(f"    subgraph {layer.title()} Layer")
                        for node in layers[layer]:
                            lines.append(f"        {node.id}[{node.label}]")
                        lines.append("    end")
            else:
                # Add nodes
                for node in nodes:
                    if node.node_type == "entry":
                        lines.append(f"    {node.id}(({node.label}))")  # Circle for entry points
                    elif node.node_type == "class":
                        lines.append(f"    {node.id}[{node.label}]")
                    else:
                        lines.append(f"    {node.id}({node.label})")

            # Add edges
            for edge in edges:
                arrow = "-->" if edge.edge_type == "call" else "-.->'"
                if edge.label:
                    lines.append(f"    {edge.source} {arrow}|{edge.label}| {edge.target}")
                else:
                    lines.append(f"    {edge.source} {arrow} {edge.target}")

        return "\n".join(lines)

    def _render_plantuml(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        diagram_type: DiagramType,
        title: str
    ) -> str:
        """Render diagram in PlantUML format."""
        lines = ["@startuml", f"title {title}", ""]

        if diagram_type == DiagramType.SEQUENCE:
            for node in nodes:
                lines.append(f'participant "{node.label}" as {node.id}')
            lines.append("")

            sorted_edges = sorted(edges, key=lambda e: e.metadata.get("sequence", 0))
            for edge in sorted_edges:
                lines.append(f"{edge.source} -> {edge.target}: {edge.label or 'call'}")

        elif diagram_type == DiagramType.CLASS:
            for node in nodes:
                if node.node_type == "interface":
                    lines.append(f"interface {node.id} {{")
                else:
                    lines.append(f"class {node.id} {{")

                for method in node.metadata.get("methods", [])[:5]:
                    lines.append(f"  +{method}()")
                lines.append("}")
                lines.append("")

            for edge in edges:
                if edge.edge_type == "implements":
                    lines.append(f"{edge.source} ..|> {edge.target}")
                elif edge.edge_type == "extends":
                    lines.append(f"{edge.source} --|> {edge.target}")
                else:
                    lines.append(f"{edge.source} --> {edge.target}")

        else:  # Flowchart-style diagrams
            # Define components
            for node in nodes:
                if node.node_type == "component":
                    lines.append(f"component [{node.label}] as {node.id}")
                elif node.node_type == "module":
                    lines.append(f"package {node.id} {{")
                    lines.append("}")
                else:
                    lines.append(f"rectangle {node.id} as \"{node.label}\"")

            lines.append("")

            # Add relationships
            for edge in edges:
                if edge.label:
                    lines.append(f"{edge.source} --> {edge.target} : {edge.label}")
                else:
                    lines.append(f"{edge.source} --> {edge.target}")

        lines.append("")
        lines.append("@enduml")
        return "\n".join(lines)

    def _render_dot(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        diagram_type: DiagramType,
        title: str
    ) -> str:
        """Render diagram in DOT/Graphviz format."""
        lines = [
            f'digraph "{title}" {{',
            '    rankdir=TB;',
            '    node [shape=box, style=rounded];',
            ''
        ]

        # Group by layer for architecture
        if diagram_type == DiagramType.ARCHITECTURE:
            layers = defaultdict(list)
            for node in nodes:
                layer = node.metadata.get("layer", "unknown")
                layers[layer].append(node)

            for layer, layer_nodes in layers.items():
                lines.append(f'    subgraph cluster_{layer} {{')
                lines.append(f'        label="{layer.title()} Layer";')
                for node in layer_nodes:
                    lines.append(f'        {node.id} [label="{node.label}"];')
                lines.append('    }')
        else:
            # Add nodes
            for node in nodes:
                shape = "ellipse" if node.node_type == "entry" else "box"
                lines.append(f'    {node.id} [label="{node.label}", shape={shape}];')

        lines.append('')

        # Add edges
        for edge in edges:
            style = "dashed" if edge.edge_type == "import" else "solid"
            if edge.label:
                lines.append(f'    {edge.source} -> {edge.target} [label="{edge.label}", style={style}];')
            else:
                lines.append(f'    {edge.source} -> {edge.target} [style={style}];')

        lines.append('}')
        return "\n".join(lines)

    def _render_ascii(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        diagram_type: DiagramType,
        title: str
    ) -> str:
        """Render diagram in ASCII art format."""
        lines = [
            "=" * 60,
            f"  {title}",
            "=" * 60,
            "",
            "NODES:",
            "-" * 40
        ]

        # Group nodes by type
        by_type = defaultdict(list)
        for node in nodes:
            by_type[node.node_type].append(node)

        for node_type, type_nodes in by_type.items():
            lines.append(f"\n[{node_type.upper()}]")
            for node in type_nodes:
                file_info = f" ({node.file_path})" if node.file_path else ""
                lines.append(f"  • {node.label}{file_info}")

        lines.extend([
            "",
            "RELATIONSHIPS:",
            "-" * 40
        ])

        for edge in edges:
            label = f" ({edge.label})" if edge.label else ""
            lines.append(f"  {edge.source} --> {edge.target}{label}")

        lines.extend([
            "",
            "=" * 60
        ])

        return "\n".join(lines)

    # Helper methods

    def _sanitize_id(self, name: str) -> str:
        """Sanitize a name for use as a diagram node ID."""
        # Replace invalid characters
        sanitized = name.replace(".", "_").replace("/", "_").replace("-", "_")
        sanitized = sanitized.replace("<", "").replace(">", "").replace(" ", "_")
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized

    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        if not file_path:
            return "unknown"

        # Remove common prefixes
        path = file_path.replace("\\", "/")
        for prefix in ["src/", "lib/", "app/", "main/"]:
            if prefix in path:
                path = path.split(prefix, 1)[1]
                break

        # Convert path to module name
        parts = path.rsplit(".", 1)[0].split("/")
        # Remove __init__ and similar
        parts = [p for p in parts if p not in ("__init__", "index")]

        return ".".join(parts[-3:]) if len(parts) > 3 else ".".join(parts)

    def _detect_layer(self, module: str, chunks: list) -> str:
        """Detect architectural layer from module name and content."""
        module_lower = module.lower()

        # Check module name patterns
        if any(p in module_lower for p in ["api", "route", "endpoint", "handler", "controller"]):
            return "api"
        if any(p in module_lower for p in ["service", "business", "usecase"]):
            return "service"
        if any(p in module_lower for p in ["repo", "dao", "database", "storage", "persist"]):
            return "repository"
        if any(p in module_lower for p in ["model", "entity", "domain", "schema"]):
            return "model"
        if any(p in module_lower for p in ["util", "helper", "common", "shared"]):
            return "util"

        # Check chunk content patterns
        for chunk in chunks[:5]:  # Check first few chunks
            if chunk.metadata:
                if chunk.metadata.get("route"):
                    return "api"
                if chunk.metadata.get("is_test"):
                    return "test"

        return "service"  # Default to service layer

    def _classify_dataflow_node(self, chunk) -> str:
        """Classify a chunk as source, processor, or sink for data flow."""
        name_lower = chunk.name.lower()

        # Sources (data providers)
        if any(p in name_lower for p in ["read", "fetch", "get", "load", "query", "find"]):
            return "source"

        # Sinks (data consumers)
        if any(p in name_lower for p in ["write", "save", "store", "send", "push", "export"]):
            return "sink"

        # Default to processor
        return "processor"

    def _infer_dataflow_label(self, source: str, target: str) -> str:
        """Infer a data flow label from source and target names."""
        target_lower = target.lower()

        if "validate" in target_lower:
            return "validate"
        if "transform" in target_lower or "convert" in target_lower:
            return "transform"
        if "save" in target_lower or "store" in target_lower:
            return "persist"
        if "send" in target_lower or "notify" in target_lower:
            return "emit"
        if "fetch" in target_lower or "get" in target_lower:
            return "fetch"

        return "process"

    def _is_entry_point(self, chunk) -> bool:
        """Check if a chunk is an entry point."""
        # Check metadata for route
        if chunk.metadata and chunk.metadata.get("route"):
            return True

        # Check name patterns
        name_lower = chunk.name.lower()
        if name_lower in ("main", "__main__"):
            return True
        if name_lower.startswith("test_"):
            return True

        return False

    def _filter_external(
        self,
        nodes: list[DiagramNode],
        edges: list[DiagramEdge],
        repo_name: Optional[str]
    ) -> tuple[list[DiagramNode], list[DiagramEdge]]:
        """Filter out external dependencies."""
        # Keep only nodes with file paths (internal)
        internal_nodes = [n for n in nodes if n.file_path or not n.metadata.get("external")]
        internal_ids = {n.id for n in internal_nodes}

        # Keep only edges between internal nodes
        internal_edges = [
            e for e in edges
            if e.source in internal_ids and e.target in internal_ids
        ]

        return internal_nodes, internal_edges


# ============================================================================
# Public API Functions
# ============================================================================

def generate_architecture_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_nodes: int = 30,
    depth: int = 2,
    include_external: bool = False,
) -> dict:
    """Generate an architecture diagram based on semantic search.

    Creates a component/layer diagram showing how different parts of
    the codebase relate to each other based on the search query.

    Args:
        query: Natural language query to find relevant code
            Examples:
            - "user authentication"
            - "payment processing flow"
            - "database operations"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_nodes: Maximum number of components to show (default: 30)
        depth: How many levels of relationships to traverse (default: 2)
        include_external: Include external dependencies (default: False)

    Returns:
        Dictionary containing:
        - diagram: The diagram content in the specified format
        - format: The output format used
        - nodes: List of components/nodes in the diagram
        - edges: List of relationships between components
        - statistics: Summary statistics

    Example:
        >>> result = generate_architecture_diagram(
        ...     query="user authentication",
        ...     repo_name="my-app",
        ...     output_format="mermaid"
        ... )
        >>> print(result["diagram"])
        flowchart TD
            subgraph API Layer
                auth_controller[AuthController]
            end
            subgraph Service Layer
                auth_service[AuthService]
            end
            auth_controller --> auth_service
    """
    console.print(f"[bold blue]Generating architecture diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.ARCHITECTURE,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_nodes,
        depth=depth,
        include_external=include_external,
    )

    # Display diagram
    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "nodes": [{"id": n.id, "label": n.label, "type": n.node_type, "file": n.file_path}
                  for n in result.nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label, "type": e.edge_type}
                  for e in result.edges],
        "statistics": result.statistics,
    }


def generate_dataflow_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_nodes: int = 25,
    depth: int = 3,
) -> dict:
    """Generate a data flow diagram based on semantic search.

    Creates a diagram showing how data moves through the system:
    - Sources (where data comes from)
    - Processors (how data is transformed)
    - Sinks (where data goes)

    Args:
        query: Natural language query to find relevant data flow
            Examples:
            - "user registration data flow"
            - "order processing pipeline"
            - "file upload handling"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_nodes: Maximum number of nodes to show (default: 25)
        depth: How many levels of data flow to trace (default: 3)

    Returns:
        Dictionary containing:
        - diagram: The diagram content in the specified format
        - format: The output format used
        - nodes: List of data flow nodes (sources, processors, sinks)
        - edges: List of data flow connections
        - statistics: Summary statistics

    Example:
        >>> result = generate_dataflow_diagram(
        ...     query="user registration",
        ...     repo_name="my-app"
        ... )
        >>> print(result["diagram"])
        flowchart LR
            register_user[(User Input)]
            validate_input[Validate]
            save_user[/Save to DB/]
            register_user -->|validate| validate_input
            validate_input -->|persist| save_user
    """
    console.print(f"[bold blue]Generating data flow diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.DATAFLOW,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_nodes,
        depth=depth,
    )

    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "nodes": [{"id": n.id, "label": n.label, "type": n.node_type,
                   "dataflow_type": n.metadata.get("dataflow_type")}
                  for n in result.nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label}
                  for e in result.edges],
        "statistics": result.statistics,
    }


def generate_callflow_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_nodes: int = 30,
    depth: int = 3,
) -> dict:
    """Generate a call flow diagram based on semantic search.

    Creates a diagram showing function/method call relationships
    starting from entry points found by the search query.

    Args:
        query: Natural language query to find entry points
            Examples:
            - "API endpoint for user login"
            - "main application entry"
            - "webhook handler"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_nodes: Maximum number of functions to show (default: 30)
        depth: How deep to trace call chains (default: 3)

    Returns:
        Dictionary containing:
        - diagram: The diagram content
        - format: The output format used
        - nodes: List of functions/methods
        - edges: List of call relationships
        - statistics: Summary statistics

    Example:
        >>> result = generate_callflow_diagram(
        ...     query="login endpoint",
        ...     repo_name="my-app"
        ... )
    """
    console.print(f"[bold blue]Generating call flow diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.CALLFLOW,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_nodes,
        depth=depth,
    )

    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "nodes": [{"id": n.id, "label": n.label, "type": n.node_type,
                   "is_entry": n.metadata.get("is_entry_point", False),
                   "file": n.file_path}
                  for n in result.nodes],
        "edges": [{"source": e.source, "target": e.target, "label": e.label,
                   "line": e.metadata.get("line")}
                  for e in result.edges],
        "statistics": result.statistics,
    }


def generate_class_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_nodes: int = 20,
) -> dict:
    """Generate a class diagram based on semantic search.

    Creates a UML-style class diagram showing classes, interfaces,
    and their relationships (inheritance, implementation).

    Args:
        query: Natural language query to find relevant classes
            Examples:
            - "user domain models"
            - "repository interfaces"
            - "service classes"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_nodes: Maximum number of classes to show (default: 20)

    Returns:
        Dictionary containing:
        - diagram: The class diagram content
        - format: The output format used
        - nodes: List of classes/interfaces with methods
        - edges: List of relationships (extends, implements)
        - statistics: Summary statistics

    Example:
        >>> result = generate_class_diagram(
        ...     query="user models",
        ...     repo_name="my-app"
        ... )
    """
    console.print(f"[bold blue]Generating class diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.CLASS,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_nodes,
        depth=1,
    )

    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "nodes": [{"id": n.id, "label": n.label, "type": n.node_type,
                   "methods": n.metadata.get("methods", []),
                   "is_interface": n.metadata.get("is_interface", False)}
                  for n in result.nodes],
        "edges": [{"source": e.source, "target": e.target, "type": e.edge_type}
                  for e in result.edges],
        "statistics": result.statistics,
    }


def generate_sequence_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_participants: int = 10,
    depth: int = 5,
) -> dict:
    """Generate a sequence diagram based on semantic search.

    Creates a UML sequence diagram showing the order of method
    calls between participants (classes/modules).

    Args:
        query: Natural language query to find the starting point
            Examples:
            - "user login flow"
            - "order checkout sequence"
            - "file upload process"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_participants: Maximum number of participants (default: 10)
        depth: How many call levels to trace (default: 5)

    Returns:
        Dictionary containing:
        - diagram: The sequence diagram content
        - format: The output format used
        - participants: List of participants (classes/modules)
        - messages: List of method calls in order
        - statistics: Summary statistics

    Example:
        >>> result = generate_sequence_diagram(
        ...     query="user login",
        ...     repo_name="my-app"
        ... )
        >>> print(result["diagram"])
        sequenceDiagram
            participant AuthController
            participant AuthService
            participant UserRepository
            AuthController->>+AuthService: authenticate
            AuthService->>+UserRepository: findByEmail
    """
    console.print(f"[bold blue]Generating sequence diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.SEQUENCE,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_participants,
        depth=depth,
    )

    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "participants": [{"id": n.id, "label": n.label} for n in result.nodes],
        "messages": [{"from": e.source, "to": e.target, "method": e.label,
                      "sequence": e.metadata.get("sequence")}
                     for e in result.edges],
        "statistics": result.statistics,
    }


def generate_dependency_diagram(
    query: str,
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    max_modules: int = 25,
) -> dict:
    """Generate a module dependency diagram based on semantic search.

    Creates a diagram showing import/dependency relationships
    between modules and packages.

    Args:
        query: Natural language query to find relevant modules
            Examples:
            - "authentication modules"
            - "database layer"
            - "API routes"
        repo_name: Filter to specific repository
        output_format: Diagram format - "mermaid", "plantuml", "dot", or "ascii"
        max_modules: Maximum number of modules to show (default: 25)

    Returns:
        Dictionary containing:
        - diagram: The dependency diagram content
        - format: The output format used
        - modules: List of modules
        - dependencies: List of import relationships
        - statistics: Summary statistics
    """
    console.print(f"[bold blue]Generating dependency diagram for: {query}[/bold blue]")

    try:
        format_enum = DiagramFormat(output_format.lower())
    except ValueError:
        return {"error": "Invalid format. Valid options: mermaid, plantuml, dot, ascii"}

    generator = DiagramGenerator()
    result = generator.generate_from_search(
        query=query,
        diagram_type=DiagramType.DEPENDENCY,
        output_format=format_enum,
        repo_name=repo_name,
        max_nodes=max_modules,
        depth=2,
    )

    console.print(Panel(
        Syntax(result.content, "text" if format_enum == DiagramFormat.ASCII else "markdown"),
        title=result.title
    ))

    return {
        "diagram": result.content,
        "format": result.format.value,
        "type": result.diagram_type.value,
        "modules": [{"id": n.id, "label": n.label, "file": n.file_path,
                     "symbol_count": n.metadata.get("symbol_count", 0)}
                    for n in result.nodes],
        "dependencies": [{"from": e.source, "to": e.target, "type": e.edge_type}
                         for e in result.edges],
        "statistics": result.statistics,
    }


# Convenience function for all diagram types
def generate_diagram(
    query: str,
    diagram_type: str = "architecture",
    repo_name: Optional[str] = None,
    output_format: str = "mermaid",
    **kwargs
) -> dict:
    """Generate any type of diagram based on semantic search.

    This is a unified function that can generate any diagram type.

    Args:
        query: Natural language query for semantic search
        diagram_type: Type of diagram to generate:
            - "architecture": Component/layer diagram
            - "dataflow": Data flow diagram
            - "callflow": Function call flow
            - "class": UML class diagram
            - "sequence": UML sequence diagram
            - "dependency": Module dependency diagram
        repo_name: Filter to specific repository
        output_format: Output format - "mermaid", "plantuml", "dot", "ascii"
        **kwargs: Additional arguments passed to specific diagram generator

    Returns:
        Dictionary with diagram content and metadata

    Example:
        >>> result = generate_diagram(
        ...     query="user authentication",
        ...     diagram_type="sequence",
        ...     repo_name="my-app",
        ...     output_format="mermaid"
        ... )
    """
    type_map = {
        "architecture": generate_architecture_diagram,
        "dataflow": generate_dataflow_diagram,
        "callflow": generate_callflow_diagram,
        "class": generate_class_diagram,
        "sequence": generate_sequence_diagram,
        "dependency": generate_dependency_diagram,
    }

    if diagram_type not in type_map:
        return {"error": f"Invalid diagram_type. Valid options: {list(type_map.keys())}"}

    return type_map[diagram_type](
        query=query,
        repo_name=repo_name,
        output_format=output_format,
        **kwargs
    )
