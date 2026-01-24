"""TypeScript/JavaScript parser using tree-sitter."""

from pathlib import Path
from typing import Optional

from ..models.chunk import CallInfo, ChunkType, CodeChunk, ParseResult
from .base import BaseParser


class TypeScriptParser(BaseParser):
    """Parser for TypeScript/JavaScript source files using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx", ".mjs"]

    def _get_parser(self):
        """Lazy initialization of tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_typescript as tstypescript
                from tree_sitter import Language, Parser

                self._language = Language(tstypescript.language_typescript())
                self._parser = Parser(self._language)
            except ImportError as e:
                raise ImportError(
                    "tree-sitter-typescript is required. Install with: pip install tree-sitter-typescript"
                ) from e
        return self._parser

    def parse_file(self, file_path: Path, repo_name: str) -> ParseResult:
        """Parse a TypeScript/JavaScript file and extract code chunks and call relationships."""
        parser = self._get_parser()

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            return ParseResult()

        tree = parser.parse(bytes(source_code, "utf-8"))
        chunks = []
        calls = []
        lines = source_code.split("\n")
        relative_path = str(file_path)

        # Determine actual language from extension
        actual_language = "javascript" if file_path.suffix in [".js", ".jsx", ".mjs"] else "typescript"

        self._extract_from_node(tree.root_node, lines, relative_path, repo_name, chunks, calls, actual_language)

        return ParseResult(chunks=chunks, calls=calls)

    def _extract_from_node(
        self,
        node,
        lines: list[str],
        file_path: str,
        repo_name: str,
        chunks: list[CodeChunk],
        calls: list[CallInfo],
        actual_language: str,
        parent_name: Optional[str] = None,
        parent_type: Optional[ChunkType] = None,
    ):
        """Recursively extract chunks from a node."""
        chunk_type = self._get_chunk_type(node.type)

        if chunk_type:
            name = self._get_node_name(node)
            if name:
                chunk = self._create_chunk(
                    node, lines, file_path, repo_name, name, chunk_type,
                    parent_name, parent_type, actual_language
                )
                chunks.append(chunk)

                # Extract calls from function/method bodies
                if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
                    qualified_name = f"{parent_name}.{name}" if parent_name else name
                    func_calls = self._extract_calls(node, qualified_name, file_path)
                    calls.extend(func_calls)

                # For classes, extract members
                if chunk_type == ChunkType.CLASS:
                    body = self._get_class_body(node)
                    if body:
                        for child in body.children:
                            self._extract_from_node(
                                child, lines, file_path, repo_name, chunks, calls,
                                actual_language, name, chunk_type
                            )
                    return

        # Recurse into children
        for child in node.children:
            self._extract_from_node(
                child, lines, file_path, repo_name, chunks, calls,
                actual_language, parent_name, parent_type
            )

    def _create_chunk(
        self,
        node,
        lines: list[str],
        file_path: str,
        repo_name: str,
        name: str,
        chunk_type: ChunkType,
        parent_name: Optional[str],
        parent_type: Optional[ChunkType],
        actual_language: str,
    ) -> CodeChunk:
        """Create a CodeChunk from a node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        signature = None
        docstring = self._extract_jsdoc(node, lines)

        if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
            signature = self._extract_signature(node, lines)

        return CodeChunk(
            id=self._generate_chunk_id(repo_name, file_path, name, chunk_type.value, start_line),
            repo_name=repo_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            name=name,
            content=content,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
            parent_type=parent_type,
            language=actual_language,
        )

    def _get_chunk_type(self, node_type: str) -> Optional[ChunkType]:
        """Map tree-sitter node type to ChunkType."""
        mapping = {
            "function_declaration": ChunkType.FUNCTION,
            "function": ChunkType.FUNCTION,
            "arrow_function": ChunkType.FUNCTION,
            "method_definition": ChunkType.METHOD,
            "class_declaration": ChunkType.CLASS,
            "interface_declaration": ChunkType.INTERFACE,
            "type_alias_declaration": ChunkType.CLASS,  # Treat type aliases like classes
            "import_statement": ChunkType.IMPORT,
            "export_statement": ChunkType.IMPORT,
        }
        return mapping.get(node_type)

    def _get_node_name(self, node) -> Optional[str]:
        """Extract the name from a node."""
        # Handle different node structures
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
            # For variable declarations with arrow functions
            if child.type == "variable_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        return subchild.text.decode("utf-8")
        return None

    def _get_class_body(self, node):
        """Get the body of a class."""
        for child in node.children:
            if child.type == "class_body":
                return child
        return None

    def _extract_signature(self, node, lines: list[str]) -> Optional[str]:
        """Extract function/method signature."""
        start_line = node.start_point[0]
        # Get lines until we find the opening brace
        for i in range(start_line, min(start_line + 10, len(lines))):
            line = lines[i]
            if "{" in line:
                sig_lines = lines[start_line : i + 1]
                sig = "\n".join(sig_lines)
                brace_idx = sig.find("{")
                return sig[:brace_idx].strip()
        return None

    def _extract_jsdoc(self, node, lines: list[str]) -> Optional[str]:
        """Extract JSDoc comment preceding a node."""
        start_line = node.start_point[0]
        if start_line == 0:
            return None

        doc_lines = []
        in_jsdoc = False

        for i in range(start_line - 1, max(start_line - 30, -1), -1):
            line = lines[i].strip()

            if line.endswith("*/"):
                in_jsdoc = True
                doc_lines.insert(0, line)
            elif in_jsdoc:
                doc_lines.insert(0, line)
                if line.startswith("/**"):
                    break
            elif line and not line.startswith("//"):
                break

        if doc_lines:
            cleaned = []
            for line in doc_lines:
                line = line.strip()
                if line.startswith("/**"):
                    line = line[3:]
                if line.endswith("*/"):
                    line = line[:-2]
                if line.startswith("*"):
                    line = line[1:]
                line = line.strip()
                if line:
                    cleaned.append(line)
            return "\n".join(cleaned) if cleaned else None

        return None

    def _extract_calls(
        self,
        func_node,
        caller_qualified_name: str,
        file_path: str,
    ) -> list[CallInfo]:
        """Extract function/method calls from a function body."""
        calls = []

        def visit_node(node):
            """Recursively visit nodes looking for call expressions."""
            if node.type == "call_expression":
                call_info = self._parse_call_expression(
                    node, caller_qualified_name, file_path
                )
                if call_info:
                    calls.append(call_info)
            elif node.type == "new_expression":
                call_info = self._parse_new_expression(
                    node, caller_qualified_name, file_path
                )
                if call_info:
                    calls.append(call_info)

            for child in node.children:
                visit_node(child)

        # Find the function body (statement_block)
        for child in func_node.children:
            if child.type == "statement_block":
                visit_node(child)
                break

        return calls

    def _parse_call_expression(
        self,
        node,
        caller_qualified_name: str,
        file_path: str,
    ) -> Optional[CallInfo]:
        """Parse a call_expression node to extract callee information."""
        if not node.children:
            return None

        callee_node = node.children[0]
        callee_name = self._get_callee_name(callee_node)

        if not callee_name:
            return None

        call_type = "direct"
        if callee_node.type == "member_expression":
            call_type = "method"

        return CallInfo(
            caller_qualified_name=caller_qualified_name,
            callee_name=callee_name,
            caller_file=file_path,
            caller_line=node.start_point[0] + 1,
            call_type=call_type,
        )

    def _parse_new_expression(
        self,
        node,
        caller_qualified_name: str,
        file_path: str,
    ) -> Optional[CallInfo]:
        """Parse a new_expression (new Foo()) to extract constructor call."""
        # Find the identifier being instantiated
        for child in node.children:
            if child.type == "identifier":
                class_name = child.text.decode("utf-8")
                return CallInfo(
                    caller_qualified_name=caller_qualified_name,
                    callee_name=f"{class_name}.constructor",
                    caller_file=file_path,
                    caller_line=node.start_point[0] + 1,
                    call_type="constructor",
                )
        return None

    def _get_callee_name(self, node) -> Optional[str]:
        """Extract the name of what's being called.

        Handles:
        - Simple calls: foo()
        - Member calls: obj.method()
        - Chained calls: obj.method1().method2()
        """
        if node.type == "identifier":
            return node.text.decode("utf-8")

        if node.type == "member_expression":
            # Get the full member expression (e.g., "this.method" or "obj.foo.bar")
            parts = []
            current = node

            while current.type == "member_expression":
                # Find the property name
                for child in reversed(current.children):
                    if child.type == "property_identifier":
                        parts.insert(0, child.text.decode("utf-8"))
                        break
                # Move to the object
                current = current.children[0] if current.children else None
                if current is None:
                    break

            # Add the base identifier if present
            if current and current.type == "identifier":
                parts.insert(0, current.text.decode("utf-8"))
            elif current and current.type == "this":
                parts.insert(0, "this")

            if parts:
                return ".".join(parts)

        if node.type == "call_expression":
            # Chained call
            if node.children:
                return self._get_callee_name(node.children[0])

        return None
