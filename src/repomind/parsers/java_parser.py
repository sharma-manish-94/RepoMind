"""Java parser using tree-sitter."""

from pathlib import Path
from typing import Optional

from ..models.chunk import CallInfo, ChunkType, CodeChunk, ParseResult
from .base import BaseParser


class JavaParser(BaseParser):
    """Parser for Java source files using tree-sitter."""

    def __init__(self):
        self._parser = None
        self._language = None

    @property
    def language(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    def _get_parser(self):
        """Lazy initialization of tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_java as tsjava
                from tree_sitter import Language, Parser

                self._language = Language(tsjava.language())
                self._parser = Parser(self._language)
            except ImportError as e:
                raise ImportError(
                    "tree-sitter-java is required. Install with: pip install tree-sitter-java"
                ) from e
        return self._parser

    def parse_file(self, file_path: Path, repo_name: str) -> ParseResult:
        """Parse a Java file and extract code chunks and call relationships."""
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

        # Walk the tree to find classes and interfaces
        self._extract_from_node(tree.root_node, lines, relative_path, repo_name, chunks, calls)

        return ParseResult(chunks=chunks, calls=calls)

    def _extract_from_node(
        self,
        node,
        lines: list[str],
        file_path: str,
        repo_name: str,
        chunks: list[CodeChunk],
        calls: list[CallInfo],
        parent_name: Optional[str] = None,
        parent_type: Optional[ChunkType] = None,
    ):
        """Recursively extract chunks from a node."""
        chunk_type = self._get_chunk_type(node.type)

        if chunk_type:
            name = self._get_node_name(node)
            if name:
                chunk = self._create_chunk(
                    node, lines, file_path, repo_name, name, chunk_type, parent_name, parent_type
                )
                chunks.append(chunk)

                # Extract calls from method/constructor bodies
                if chunk_type in (ChunkType.METHOD, ChunkType.CONSTRUCTOR):
                    qualified_name = f"{parent_name}.{name}" if parent_name else name
                    method_calls = self._extract_calls(node, qualified_name, file_path)
                    calls.extend(method_calls)

                # For classes/interfaces, extract members
                if chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
                    body = self._get_class_body(node)
                    if body:
                        for child in body.children:
                            self._extract_from_node(
                                child, lines, file_path, repo_name, chunks, calls, name, chunk_type
                            )
                    return  # Don't recurse further for class children

        # Recurse into children for package-level declarations
        for child in node.children:
            self._extract_from_node(child, lines, file_path, repo_name, chunks, calls, parent_name, parent_type)

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
    ) -> CodeChunk:
        """Create a CodeChunk from a node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        signature = None
        docstring = None

        if chunk_type in (ChunkType.METHOD, ChunkType.CONSTRUCTOR):
            signature = self._extract_signature(node, lines)
            docstring = self._extract_javadoc(node, lines)
        elif chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
            docstring = self._extract_javadoc(node, lines)

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
            language=self.language,
        )

    def _get_chunk_type(self, node_type: str) -> Optional[ChunkType]:
        """Map tree-sitter node type to ChunkType."""
        mapping = {
            "class_declaration": ChunkType.CLASS,
            "interface_declaration": ChunkType.INTERFACE,
            "method_declaration": ChunkType.METHOD,
            "constructor_declaration": ChunkType.CONSTRUCTOR,
            "field_declaration": ChunkType.PROPERTY,
            "import_declaration": ChunkType.IMPORT,
        }
        return mapping.get(node_type)

    def _get_node_name(self, node) -> Optional[str]:
        """Extract the name from a node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_class_body(self, node):
        """Get the body of a class or interface."""
        for child in node.children:
            if child.type == "class_body" or child.type == "interface_body":
                return child
        return None

    def _extract_signature(self, node, lines: list[str]) -> Optional[str]:
        """Extract method/constructor signature."""
        start_line = node.start_point[0]
        # Find the opening brace
        for i in range(start_line, min(start_line + 10, len(lines))):
            line = lines[i]
            if "{" in line:
                sig_lines = lines[start_line : i + 1]
                sig = "\n".join(sig_lines)
                # Trim to just before the brace
                brace_idx = sig.rfind("{")
                return sig[:brace_idx].strip()
        return None

    def _extract_javadoc(self, node, lines: list[str]) -> Optional[str]:
        """Extract Javadoc comment preceding a node."""
        start_line = node.start_point[0]
        if start_line == 0:
            return None

        # Look backwards for Javadoc
        doc_lines = []
        in_javadoc = False

        for i in range(start_line - 1, max(start_line - 30, -1), -1):
            line = lines[i].strip()

            if line.endswith("*/"):
                in_javadoc = True
                doc_lines.insert(0, line)
            elif in_javadoc:
                doc_lines.insert(0, line)
                if line.startswith("/**"):
                    break
            elif line and not line.startswith("@"):
                # Hit non-empty, non-annotation line - stop
                break

        if doc_lines:
            # Clean up Javadoc formatting
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
        method_node,
        caller_qualified_name: str,
        file_path: str,
    ) -> list[CallInfo]:
        """Extract method/constructor calls from a method body."""
        calls = []

        def visit_node(node):
            """Recursively visit nodes looking for method invocations."""
            if node.type == "method_invocation":
                call_info = self._parse_method_invocation(
                    node, caller_qualified_name, file_path
                )
                if call_info:
                    calls.append(call_info)
            elif node.type == "object_creation_expression":
                call_info = self._parse_constructor_call(
                    node, caller_qualified_name, file_path
                )
                if call_info:
                    calls.append(call_info)

            for child in node.children:
                visit_node(child)

        # Find the method body
        for child in method_node.children:
            if child.type == "block":
                visit_node(child)
                break

        return calls

    def _parse_method_invocation(
        self,
        node,
        caller_qualified_name: str,
        file_path: str,
    ) -> Optional[CallInfo]:
        """Parse a method_invocation node to extract callee information."""
        # method_invocation children: [object], name, argument_list
        callee_name = None
        call_type = "direct"

        for child in node.children:
            if child.type == "identifier":
                # This is the method name
                method_name = child.text.decode("utf-8")
                if callee_name:
                    callee_name = f"{callee_name}.{method_name}"
                else:
                    callee_name = method_name
            elif child.type == "field_access":
                # Object.method pattern
                callee_name = child.text.decode("utf-8")
                call_type = "method"

        if not callee_name:
            return None

        return CallInfo(
            caller_qualified_name=caller_qualified_name,
            callee_name=callee_name,
            caller_file=file_path,
            caller_line=node.start_point[0] + 1,
            call_type=call_type,
        )

    def _parse_constructor_call(
        self,
        node,
        caller_qualified_name: str,
        file_path: str,
    ) -> Optional[CallInfo]:
        """Parse an object_creation_expression (new Foo()) to extract callee."""
        # Find the type being instantiated
        for child in node.children:
            if child.type == "type_identifier":
                class_name = child.text.decode("utf-8")
                return CallInfo(
                    caller_qualified_name=caller_qualified_name,
                    callee_name=f"{class_name}.<init>",
                    caller_file=file_path,
                    caller_line=node.start_point[0] + 1,
                    call_type="constructor",
                )
        return None
