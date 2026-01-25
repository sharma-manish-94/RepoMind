"""
Python Source File Parser.

This module provides a tree-sitter based parser for Python source files.
It extracts semantic code chunks (functions, classes, methods) and
call relationships for indexing and analysis.

What Gets Extracted:
    - **Functions**: Top-level function definitions
    - **Classes**: Class definitions with their docstrings
    - **Methods**: Functions defined inside classes
    - **Imports**: Import statements (for dependency tracking)
    - **Call Graph**: Which functions call which other functions

How It Works:
    1. Parse source file using tree-sitter Python grammar
    2. Walk the AST to find function/class definitions
    3. Extract metadata (name, signature, docstring)
    4. Find all function calls within each function body
    5. Return ParseResult with chunks and call relationships

Example:
    from repomind.parsers.python_parser import PythonParser

    parser = PythonParser()
    result = parser.parse_file(Path("src/service.py"), "MyRepo")

    for chunk in result.chunks:
        print(f"{chunk.chunk_type}: {chunk.name}")

    for call in result.calls:
        print(f"{call.caller_qualified_name} -> {call.callee_name}")

Tree-Sitter Grammar Reference:
    - function_definition: def foo(): ...
    - class_definition: class Foo: ...
    - call: foo() or obj.method()
    - attribute: obj.attr

Author: RepoMind Team
"""

from pathlib import Path
from typing import Optional

from ..logging import get_logger
from ..models.chunk import CallInfo, ChunkType, CodeChunk, InheritanceInfo, ParseResult
from .base import BaseParser


# Module logger
logger = get_logger(__name__)


class PythonParser(BaseParser):
    """
    Parser for Python source files using tree-sitter.

    This parser extracts semantic code units from Python files:
    - Functions and their signatures/docstrings
    - Classes with their methods
    - Function call relationships for the call graph

    The parser is lazy-initialized - the tree-sitter grammar is only
    loaded when the first file is parsed.

    Attributes:
        language: Always "python"
        file_extensions: [".py"]

    Example:
        parser = PythonParser()

        # Parse a single file
        result = parser.parse_file(
            file_path=Path("src/handlers/auth.py"),
            repo_name="Actions"
        )

        # Access extracted chunks
        for chunk in result.chunks:
            if chunk.chunk_type == ChunkType.FUNCTION:
                print(f"Found function: {chunk.name}")
                print(f"  Signature: {chunk.signature}")
                print(f"  Docstring: {chunk.docstring}")

        # Access call relationships
        for call in result.calls:
            print(f"{call.caller_qualified_name} calls {call.callee_name}")
    """

    def __init__(self):
        """Initialize the Python parser with lazy tree-sitter loading."""
        self._parser = None
        self._language = None

    @property
    def language(self) -> str:
        """Return the language identifier for Python."""
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        """Return file extensions handled by this parser."""
        return [".py"]

    def _get_parser(self):
        """
        Lazy initialization of the tree-sitter parser.

        The parser and language grammar are loaded on first use,
        avoiding startup overhead when the parser isn't needed.

        Returns:
            Configured tree-sitter Parser instance.

        Raises:
            ImportError: If tree-sitter-python is not installed.
        """
        if self._parser is None:
            try:
                import tree_sitter_python as tspython
                from tree_sitter import Language, Parser

                self._language = Language(tspython.language())
                self._parser = Parser(self._language)

                logger.debug("Tree-sitter Python parser initialized")

            except ImportError as e:
                raise ImportError(
                    "tree-sitter-python is required for Python parsing. "
                    "Install with: pip install tree-sitter-python"
                ) from e

        return self._parser

    def parse_file(self, file_path: Path, repo_name: str) -> ParseResult:
        """
        Parse a Python file and extract code chunks and call relationships.

        This is the main entry point for parsing. It:
        1. Reads the source file
        2. Parses it into an AST using tree-sitter
        3. Extracts functions, classes, and methods as CodeChunks
        4. Extracts function calls for the call graph

        Args:
            file_path: Path to the Python file to parse.
            repo_name: Name of the repository (for chunk identification).

        Returns:
            ParseResult containing extracted chunks and call relationships.
            Returns empty ParseResult if file cannot be read or parsed.

        Example:
            parser = PythonParser()
            result = parser.parse_file(
                Path("/code/repo/src/service.py"),
                "MyRepo"
            )

            print(f"Found {len(result.chunks)} code chunks")
            print(f"Found {len(result.calls)} call relationships")
        """
        parser = self._get_parser()

        # Read source file
        try:
            source_code = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file as UTF-8: {file_path}")
            return ParseResult()
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return ParseResult()

        # Parse into AST
        abstract_syntax_tree = parser.parse(bytes(source_code, "utf-8"))

        # Prepare for extraction
        extracted_chunks = []
        extracted_calls = []
        extracted_inheritance = []
        source_lines = source_code.split("\n")
        relative_path = str(file_path)

        # Walk top-level nodes and extract chunks
        for node in abstract_syntax_tree.root_node.children:
            chunk = self._extract_chunk_from_node(
                node, source_lines, relative_path, repo_name
            )

            if chunk:
                extracted_chunks.append(chunk)

                # Extract calls from function bodies
                if chunk.chunk_type == ChunkType.FUNCTION:
                    function_calls = self._extract_calls_from_function(
                        node, chunk.name, relative_path
                    )
                    extracted_calls.extend(function_calls)

            # For classes, also extract methods and inheritance
            if node.type == "class_definition":
                class_name = self._get_node_name(node)
                method_chunks, method_calls = self._extract_class_members(
                    node, source_lines, relative_path, repo_name, class_name
                )
                extracted_chunks.extend(method_chunks)
                extracted_calls.extend(method_calls)

                # Extract inheritance information
                inheritance_info = self._extract_inheritance(
                    node, class_name, relative_path
                )
                extracted_inheritance.extend(inheritance_info)

        logger.debug(
            f"Parsed {file_path.name}",
            extra={
                "chunks": len(extracted_chunks),
                "calls": len(extracted_calls),
                "inheritance": len(extracted_inheritance),
            }
        )

        return ParseResult(
            chunks=extracted_chunks,
            calls=extracted_calls,
            inheritance=extracted_inheritance
        )

    def _extract_chunk_from_node(
        self,
        node,
        source_lines: list[str],
        file_path: str,
        repo_name: str,
        parent_name: Optional[str] = None,
        parent_type: Optional[ChunkType] = None,
    ) -> Optional[CodeChunk]:
        """
        Extract a CodeChunk from a tree-sitter AST node.

        Examines the node type and extracts relevant information
        like name, content, signature, and docstring.

        Args:
            node: Tree-sitter AST node to extract from.
            source_lines: All lines of the source file.
            file_path: Path to the source file.
            repo_name: Name of the repository.
            parent_name: Name of parent construct (for methods).
            parent_type: Type of parent construct (for methods).

        Returns:
            CodeChunk if the node is a supported construct, None otherwise.
        """
        chunk_type = self._map_node_type_to_chunk_type(node.type)
        if chunk_type is None:
            return None

        name = self._get_node_name(node)
        if not name:
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(source_lines[start_line - 1 : end_line])

        signature = None
        docstring = None

        if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
            signature = self._extract_function_signature(node, source_lines)
            docstring = self._extract_docstring(node, source_lines)

        if chunk_type == ChunkType.CLASS:
            docstring = self._extract_docstring(node, source_lines)

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

    def _extract_class_members(
        self,
        class_node,
        source_lines: list[str],
        file_path: str,
        repo_name: str,
        class_name: str,
    ) -> tuple[list[CodeChunk], list[CallInfo]]:
        """
        Extract methods and properties from a class definition.

        Walks the class body to find method definitions and extracts
        both the method chunks and their call relationships.

        Args:
            class_node: Tree-sitter node for the class definition.
            source_lines: All lines of the source file.
            file_path: Path to the source file.
            repo_name: Name of the repository.
            class_name: Name of the class (for qualified method names).

        Returns:
            Tuple of (method chunks, call relationships).
        """
        extracted_chunks = []
        extracted_calls = []

        # Find the class body block
        class_body = None
        for child in class_node.children:
            if child.type == "block":
                class_body = child
                break

        if not class_body:
            return extracted_chunks, extracted_calls

        # Extract each method in the class
        for node in class_body.children:
            if node.type == "function_definition":
                chunk = self._extract_chunk_from_node(
                    node, source_lines, file_path, repo_name, class_name, ChunkType.CLASS
                )
                if chunk:
                    # Mark as METHOD (not FUNCTION) since it's inside a class
                    chunk.chunk_type = ChunkType.METHOD
                    extracted_chunks.append(chunk)

                    # Extract calls from the method body
                    qualified_method_name = f"{class_name}.{chunk.name}"
                    method_calls = self._extract_calls_from_function(
                        node, qualified_method_name, file_path
                    )
                    extracted_calls.extend(method_calls)

        return extracted_chunks, extracted_calls

    def _extract_calls_from_function(
        self,
        function_node,
        caller_qualified_name: str,
        file_path: str,
    ) -> list[CallInfo]:
        """
        Extract all function/method calls from a function body.

        Recursively traverses the AST to find all call expressions
        within the function and creates CallInfo objects for each.

        Args:
            function_node: Tree-sitter node for the function definition.
            caller_qualified_name: Fully qualified name of the calling function
                                   (e.g., "ClassName.method_name" or "function_name").
            file_path: Path to the source file.

        Returns:
            List of CallInfo objects representing each function call.
        """
        discovered_calls = []

        def visit_node_recursively(node):
            """Recursively visit nodes looking for call expressions."""
            if node.type == "call":
                call_info = self._parse_call_expression(
                    node, caller_qualified_name, file_path
                )
                if call_info:
                    discovered_calls.append(call_info)

            # Continue recursing into children
            for child in node.children:
                visit_node_recursively(child)

        # Find the function body and traverse it
        for child in function_node.children:
            if child.type == "block":
                visit_node_recursively(child)
                break

        return discovered_calls

    def _parse_call_expression(
        self,
        call_node,
        caller_qualified_name: str,
        file_path: str,
    ) -> Optional[CallInfo]:
        """
        Parse a call expression node to extract callee information.

        Args:
            call_node: Tree-sitter node representing a function call.
            caller_qualified_name: Name of the function containing this call.
            file_path: Path to the source file.

        Returns:
            CallInfo object if the call can be parsed, None otherwise.
        """
        # The first child of a call node is what's being called
        if not call_node.children:
            return None

        callee_node = call_node.children[0]
        callee_name = self._extract_callee_name(callee_node)

        if not callee_name:
            return None

        # Determine call type
        call_type = "direct"
        if callee_node.type == "attribute":
            call_type = "method"

        return CallInfo(
            caller_qualified_name=caller_qualified_name,
            callee_name=callee_name,
            caller_file=file_path,
            caller_line=call_node.start_point[0] + 1,
            call_type=call_type,
        )

    def _extract_callee_name(self, node) -> Optional[str]:
        """
        Extract the name of what's being called from a call expression.

        Handles various call patterns:
        - Simple calls: foo() -> "foo"
        - Attribute calls: obj.method() -> "obj.method"
        - Chained calls: obj.method1().method2() -> "method2"

        Args:
            node: Tree-sitter node representing the callee.

        Returns:
            String name of the callee, or None if it cannot be determined.
        """
        if node.type == "identifier":
            return node.text.decode("utf-8")

        if node.type == "attribute":
            # Build the full attribute path (e.g., "self.method" or "obj.foo.bar")
            name_parts = []
            current_node = node

            while current_node.type == "attribute":
                # Find the attribute name (rightmost identifier)
                for child in reversed(current_node.children):
                    if child.type == "identifier":
                        name_parts.insert(0, child.text.decode("utf-8"))
                        break
                # Move to the object being accessed
                current_node = current_node.children[0] if current_node.children else None
                if current_node is None:
                    break

            # Add the base identifier if present
            if current_node and current_node.type == "identifier":
                name_parts.insert(0, current_node.text.decode("utf-8"))

            if name_parts:
                return ".".join(name_parts)

        if node.type == "call":
            # Chained call: something().method()
            # Just get the final method name
            if node.children:
                return self._extract_callee_name(node.children[0])

        return None

    def _map_node_type_to_chunk_type(self, node_type: str) -> Optional[ChunkType]:
        """
        Map a tree-sitter node type to our ChunkType enum.

        Args:
            node_type: The tree-sitter node type string.

        Returns:
            Corresponding ChunkType, or None if not a supported type.
        """
        node_type_mapping = {
            "function_definition": ChunkType.FUNCTION,
            "class_definition": ChunkType.CLASS,
            "import_statement": ChunkType.IMPORT,
            "import_from_statement": ChunkType.IMPORT,
        }
        return node_type_mapping.get(node_type)

    def _get_node_name(self, node) -> Optional[str]:
        """
        Extract the name identifier from a tree-sitter node.

        Args:
            node: Tree-sitter AST node.

        Returns:
            The name string, or None if no name found.
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _extract_function_signature(self, node, source_lines: list[str]) -> Optional[str]:
        """
        Extract the function/method signature.

        Gets the def line(s) including parameters and return type annotation.

        Args:
            node: Tree-sitter node for the function definition.
            source_lines: All lines of the source file.

        Returns:
            The function signature string, or None if not found.
        """
        # Find the parameters node
        for child in node.children:
            if child.type == "parameters":
                start_line = node.start_point[0]
                # Get just the def line(s) up to the colon
                signature_lines = []
                for i in range(start_line, min(start_line + 5, len(source_lines))):
                    signature_lines.append(source_lines[i])
                    if ":" in source_lines[i]:
                        break
                return "\n".join(signature_lines).strip()
        return None

    def _extract_docstring(self, node, source_lines: list[str]) -> Optional[str]:
        """
        Extract the docstring from a function or class.

        Looks for a string literal as the first statement in the body.

        Args:
            node: Tree-sitter node for the function/class definition.
            source_lines: All lines of the source file.

        Returns:
            The docstring text (without quotes), or None if not present.
        """
        # Find the block/body
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        if not body or not body.children:
            return None

        # First child of body might be expression_statement containing string
        first_statement = body.children[0]
        if first_statement.type == "expression_statement":
            for child in first_statement.children:
                if child.type == "string":
                    docstring = child.text.decode("utf-8")
                    # Clean up triple quotes
                    if docstring.startswith('"""') or docstring.startswith("'''"):
                        docstring = docstring[3:-3].strip()
                    return docstring

        return None

    def _extract_inheritance(
        self, class_node, class_name: str, file_path: str
    ) -> list[InheritanceInfo]:
        """
        Extract inheritance information from a class definition.

        Looks for the argument_list after the class name to find base classes.
        In Python, all inheritance uses the same syntax: class Child(Parent1, Parent2)

        Args:
            class_node: Tree-sitter node for the class definition.
            class_name: Name of the class being defined.
            file_path: Path to the source file.

        Returns:
            List of InheritanceInfo objects for each base class.
        """
        inheritance_list = []
        line_number = class_node.start_point[0] + 1

        # Find the argument_list which contains base classes
        for child in class_node.children:
            if child.type == "argument_list":
                # Iterate through base classes
                for arg in child.children:
                    parent_name = None

                    if arg.type == "identifier":
                        # Simple base class: class Child(Parent)
                        parent_name = arg.text.decode("utf-8")
                    elif arg.type == "attribute":
                        # Qualified base class: class Child(module.Parent)
                        parent_name = arg.text.decode("utf-8")
                    elif arg.type == "call":
                        # Generic base class: class Child(Generic[T])
                        # Get the function being called
                        if arg.children:
                            callee = arg.children[0]
                            if callee.type == "identifier":
                                parent_name = callee.text.decode("utf-8")
                            elif callee.type == "attribute":
                                parent_name = callee.text.decode("utf-8")

                    if parent_name and parent_name not in ("object",):
                        inheritance_list.append(
                            InheritanceInfo(
                                child_name=class_name,
                                child_qualified=class_name,
                                parent_name=parent_name,
                                relation_type="extends",
                                file_path=file_path,
                                line_number=line_number,
                            )
                        )

        return inheritance_list
