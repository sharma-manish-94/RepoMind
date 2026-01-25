"""Java parser using tree-sitter."""

from pathlib import Path
from typing import Optional

from ..models.chunk import CallInfo, ChunkType, CodeChunk, InheritanceInfo, ParseResult
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
        """Parse a Java file and extract code chunks, call relationships, and inheritance."""
        parser = self._get_parser()

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            return ParseResult()

        tree = parser.parse(bytes(source_code, "utf-8"))
        chunks = []
        calls = []
        inheritance = []
        lines = source_code.split("\n")
        relative_path = str(file_path)

        # Walk the tree to find classes and interfaces
        self._extract_from_node(tree.root_node, lines, relative_path, repo_name, chunks, calls, inheritance)

        return ParseResult(chunks=chunks, calls=calls, inheritance=inheritance)

    def _extract_from_node(
        self,
        node,
        lines: list[str],
        file_path: str,
        repo_name: str,
        chunks: list[CodeChunk],
        calls: list[CallInfo],
        inheritance: list[InheritanceInfo],
        parent_name: Optional[str] = None,
        parent_type: Optional[ChunkType] = None,
        class_annotations: Optional[dict] = None
    ):
        """Recursively extract chunks from a node."""
        chunk_type = self._get_chunk_type(node.type)

        if not class_annotations:
            class_annotations = {}

        if chunk_type:
            name = self._get_node_name(node)
            if name:
                # Handle class-level annotations
                if chunk_type == ChunkType.CLASS:
                    class_annotations = self._extract_annotations(node)

                chunk = self._create_chunk(
                    node, lines, file_path, repo_name, name, chunk_type, parent_name, parent_type, class_annotations
                )
                chunks.append(chunk)

                # Extract calls from method/constructor bodies
                if chunk_type in (ChunkType.METHOD, ChunkType.CONSTRUCTOR):
                    qualified_name = f"{parent_name}.{name}" if parent_name else name
                    method_calls = self._extract_calls(node, qualified_name, file_path)
                    calls.extend(method_calls)

                # For classes/interfaces, extract members and inheritance
                if chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
                    # Extract inheritance information
                    inheritance_info = self._extract_inheritance(node, name, file_path)
                    inheritance.extend(inheritance_info)

                    body = self._get_class_body(node)
                    if body:
                        for child in body.children:
                            self._extract_from_node(
                                child, lines, file_path, repo_name, chunks, calls, inheritance, name, chunk_type, class_annotations
                            )
                    return  # Don't recurse further for class children

        # Recurse into children for package-level declarations
        for child in node.children:
            self._extract_from_node(child, lines, file_path, repo_name, chunks, calls, inheritance, parent_name, parent_type, class_annotations)

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
        class_annotations: Optional[dict] = None
    ) -> CodeChunk:
        """Create a CodeChunk from a node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1 : end_line])

        signature = None
        docstring = None
        metadata = {}

        if chunk_type in (ChunkType.METHOD, ChunkType.CONSTRUCTOR):
            signature = self._extract_signature(node, lines)
            docstring = self._extract_javadoc(node, lines)
            # Spring MVC route handling
            annotations = self._extract_annotations(node)
            route_info = self._get_route_info(annotations, class_annotations)
            if route_info:
                metadata["route"] = route_info
        elif chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
            docstring = self._extract_javadoc(node, lines)
            # Extract Spring DI dependencies
            if chunk_type == ChunkType.CLASS:
                dependencies = self._extract_spring_dependencies(node)
                if dependencies:
                    metadata["dependencies"] = dependencies

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
            metadata=metadata,
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

        # We need to look at the node's sibling to find the comment
        previous_sibling = node.prev_named_sibling
        if previous_sibling and previous_sibling.type == 'block_comment':
             comment_text = previous_sibling.text.decode('utf-8')
             if comment_text.startswith('/**'):
                cleaned_lines = []
                for line in comment_text.split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line.startswith('/**'):
                        cleaned_line = cleaned_line[3:]
                    if cleaned_line.endswith('*/'):
                        cleaned_line = cleaned_line[:-2]
                    if cleaned_line.startswith('*'):
                        cleaned_line = cleaned_line[1:]
                    cleaned_lines.append(cleaned_line.strip())
                return "\n".join(cleaned_lines)
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
        callee_name = None
        call_type = "direct"

        for child in node.children:
            if child.type == "identifier":
                method_name = child.text.decode("utf-8")
                if callee_name:
                    callee_name = f"{callee_name}.{method_name}"
                else:
                    callee_name = method_name
            elif child.type == "field_access":
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

    def _extract_inheritance(
        self, class_node, class_name: str, file_path: str
    ) -> list[InheritanceInfo]:
        """Extract inheritance information from a class or interface declaration."""
        inheritance_list = []
        line_number = class_node.start_point[0] + 1

        for child in class_node.children:
            if child.type == "superclass":
                for type_child in child.children:
                    if type_child.type == "type_identifier":
                        parent_name = type_child.text.decode("utf-8")
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
            elif child.type == "super_interfaces":
                for interface_child in child.children:
                    if interface_child.type == "type_list":
                        for type_child in interface_child.children:
                            if type_child.type == "type_identifier":
                                interface_name = type_child.text.decode("utf-8")
                                inheritance_list.append(
                                    InheritanceInfo(
                                        child_name=class_name,
                                        child_qualified=class_name,
                                        parent_name=interface_name,
                                        relation_type="implements",
                                        file_path=file_path,
                                        line_number=line_number,
                                    )
                                )
                    elif interface_child.type == "type_identifier":
                        interface_name = interface_child.text.decode("utf-8")
                        inheritance_list.append(
                            InheritanceInfo(
                                child_name=class_name,
                                child_qualified=class_name,
                                parent_name=interface_name,
                                relation_type="implements",
                                file_path=file_path,
                                line_number=line_number,
                            )
                        )
            elif child.type == "extends_interfaces":
                for interface_child in child.children:
                    if interface_child.type == "type_list":
                        for type_child in interface_child.children:
                            if type_child.type == "type_identifier":
                                parent_name = type_child.text.decode("utf-8")
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
                    elif interface_child.type == "type_identifier":
                        parent_name = interface_child.text.decode("utf-8")
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

    def _extract_annotations(self, node) -> dict:
        """Extracts annotations from a class or method node."""
        annotations = {}
        modifiers_node = None
        for child in node.children:
            if child.type == 'modifiers':
                modifiers_node = child
                break

        if modifiers_node:
            for child in modifiers_node.children:
                if child.type == 'annotation':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        name = name_node.text.decode('utf-8')
                        arguments_node = child.child_by_field_name('arguments')
                        args = {}
                        if arguments_node:
                            # Simplified parsing of arguments
                            # This can be improved to handle complex cases
                            arg_text = arguments_node.text.decode('utf-8').strip('()')
                            if '=' in arg_text:
                                try:
                                    key, value = arg_text.split('=', 1)
                                    args[key.strip()] = value.strip().strip('"')
                                except ValueError:
                                    args['value'] = arg_text.strip('"')
                            else:
                                args['value'] = arg_text.strip('"')
                        annotations[name] = args
        return annotations

    def _get_route_info(self, annotations: dict, class_annotations: dict) -> Optional[dict]:
        """Gets route information from Spring MVC annotations."""
        http_method = None
        path = ""

        if "GetMapping" in annotations:
            http_method = "GET"
            path = annotations.get("GetMapping", {}).get("value", "")
        elif "PostMapping" in annotations:
            http_method = "POST"
            path = annotations.get("PostMapping", {}).get("value", "")
        elif "PutMapping" in annotations:
            http_method = "PUT"
            path = annotations.get("PutMapping", {}).get("value", "")
        elif "DeleteMapping" in annotations:
            http_method = "DELETE"
            path = annotations.get("DeleteMapping", {}).get("value", "")
        elif "RequestMapping" in annotations:
            # Can specify method, defaults to GET
            http_method = annotations.get("RequestMapping", {}).get("method", "GET")
            path = annotations.get("RequestMapping", {}).get("value", "")

        if http_method:
            base_path = ""
            if "RequestMapping" in class_annotations:
                base_path = class_annotations.get("RequestMapping", {}).get("value", "")

            full_path = f"{base_path}{path}"
            # Normalize path
            full_path = '/' + '/'.join(filter(None, full_path.split('/')))

            return {"method": http_method, "path": full_path}
        return None

    def _extract_spring_dependencies(self, class_node) -> list[dict]:
        """
        Extract Spring dependency injection information from a class.

        Detects:
        - Field injection: @Autowired or @Inject on fields
        - Constructor injection: Parameters of constructors in @Component/@Service classes

        Args:
            class_node: Tree-sitter node for the class declaration.

        Returns:
            List of dependency dicts with 'name' and 'type' keys.
        """
        dependencies = []
        class_annotations = self._extract_annotations(class_node)
        is_spring_component = any(
            ann in class_annotations
            for ann in ("Component", "Service", "Repository", "Controller", "RestController", "Configuration")
        )

        body = self._get_class_body(class_node)
        if not body:
            return dependencies

        for child in body.children:
            # Check for field injection
            if child.type == "field_declaration":
                field_annotations = self._extract_annotations(child)
                if "Autowired" in field_annotations or "Inject" in field_annotations:
                    field_info = self._extract_field_info(child)
                    if field_info:
                        dependencies.append({
                            "name": field_info["name"],
                            "type": field_info["type"],
                            "injection_type": "field"
                        })

            # Check for constructor injection
            elif child.type == "constructor_declaration" and is_spring_component:
                constructor_params = self._extract_constructor_params(child)
                for param in constructor_params:
                    dependencies.append({
                        "name": param["name"],
                        "type": param["type"],
                        "injection_type": "constructor"
                    })

        return dependencies

    def _extract_field_info(self, field_node) -> Optional[dict]:
        """Extract field name and type from a field declaration."""
        field_type = None
        field_name = None

        for child in field_node.children:
            if child.type == "type_identifier":
                field_type = child.text.decode("utf-8")
            elif child.type == "generic_type":
                # Handle generics like List<User>
                field_type = child.text.decode("utf-8")
            elif child.type == "variable_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        field_name = subchild.text.decode("utf-8")
                        break

        if field_type and field_name:
            return {"name": field_name, "type": field_type}
        return None

    def _extract_constructor_params(self, constructor_node) -> list[dict]:
        """Extract parameter names and types from a constructor."""
        params = []

        for child in constructor_node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter":
                        param_type = None
                        param_name = None

                        for subchild in param.children:
                            if subchild.type == "type_identifier":
                                param_type = subchild.text.decode("utf-8")
                            elif subchild.type == "generic_type":
                                param_type = subchild.text.decode("utf-8")
                            elif subchild.type == "identifier":
                                param_name = subchild.text.decode("utf-8")

                        if param_type and param_name:
                            params.append({"name": param_name, "type": param_type})

        return params
