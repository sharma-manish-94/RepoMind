"""TypeScript/JavaScript parser using tree-sitter.

Enhanced to support route extraction from:
- Express.js: app.get('/users', handler), router.post('/users/:id', ...)
- NestJS: @Controller('users'), @Get(':id'), @Post(), etc.
- Fastify: fastify.get('/users', handler)
- Koa Router: router.get('/users', handler)
"""

import re
from pathlib import Path
from typing import Optional

from ..models.chunk import CallInfo, ChunkType, CodeChunk, InheritanceInfo, ParseResult
from .base import BaseParser


# HTTP methods supported by Express/NestJS
HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head", "all"}

# NestJS decorators for routes
NESTJS_ROUTE_DECORATORS = {"Get", "Post", "Put", "Delete", "Patch", "Options", "Head", "All"}
NESTJS_CONTROLLER_DECORATOR = "Controller"


class TypeScriptParser(BaseParser):
    """Parser for TypeScript/JavaScript source files using tree-sitter.

    Enhanced with route extraction for:
    - Express.js routes (app.get, router.post, etc.)
    - NestJS decorators (@Controller, @Get, @Post, etc.)
    - Router mounting (app.use('/api', router))
    """

    def __init__(self):
        self._parser = None
        self._language = None
        self._express_vars: set[str] = set()  # Track Express app/router variables
        self._current_controller_path: Optional[str] = None  # NestJS controller base path

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
        """Parse a TypeScript/JavaScript file and extract code chunks, call relationships, and inheritance."""
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

        # Determine actual language from extension
        actual_language = "javascript" if file_path.suffix in [".js", ".jsx", ".mjs"] else "typescript"

        self._extract_from_node(tree.root_node, lines, relative_path, repo_name, chunks, calls, inheritance, actual_language)

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

                # For classes/interfaces, extract members and inheritance
                if chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE):
                    # Extract inheritance information
                    inheritance_info = self._extract_inheritance(node, name, file_path)
                    inheritance.extend(inheritance_info)

                    body = self._get_class_body(node) if chunk_type == ChunkType.CLASS else self._get_interface_body(node)
                    if body:
                        for child in body.children:
                            self._extract_from_node(
                                child, lines, file_path, repo_name, chunks, calls,
                                inheritance, actual_language, name, chunk_type
                            )
                    return

        # Recurse into children
        for child in node.children:
            self._extract_from_node(
                child, lines, file_path, repo_name, chunks, calls,
                inheritance, actual_language, parent_name, parent_type
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

        # Check for React component, outgoing API calls, and route definitions
        metadata = {}
        if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
            react_info = self._get_react_component_info(node)
            if react_info:
                metadata.update(react_info)

            # Extract outgoing API calls
            api_calls = self._extract_outgoing_api_calls(node)
            if api_calls:
                metadata["outgoing_api_calls"] = api_calls

        # Extract route information for functions, methods, and classes
        if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS):
            route_info = self._extract_route_for_chunk(node, lines, chunk_type)
            if route_info:
                metadata["route"] = route_info

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
            metadata=metadata,
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

    def _get_interface_body(self, node):
        """Get the body of an interface."""
        for child in node.children:
            if child.type == "object_type" or child.type == "interface_body":
                return child
        return None

    def _extract_inheritance(
        self, class_node, class_name: str, file_path: str
    ) -> list[InheritanceInfo]:
        """
        Extract inheritance information from a class or interface declaration.

        Looks for:
        - 'extends' clause for class/interface inheritance
        - 'implements' clause for interface implementation

        Args:
            class_node: Tree-sitter node for the class/interface declaration.
            class_name: Name of the class/interface being defined.
            file_path: Path to the source file.

        Returns:
            List of InheritanceInfo objects for each parent class/interface.
        """
        inheritance_list = []
        line_number = class_node.start_point[0] + 1

        for child in class_node.children:
            # Handle 'extends' clause (class_heritage)
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    if heritage_child.type == "extends_clause":
                        # extends_clause contains type_identifier(s)
                        for type_child in heritage_child.children:
                            if type_child.type == "identifier" or type_child.type == "type_identifier":
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

                    elif heritage_child.type == "implements_clause":
                        # implements_clause contains type_identifier(s)
                        for type_child in heritage_child.children:
                            if type_child.type == "identifier" or type_child.type == "type_identifier":
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

            # Handle interface extends (extends_type_clause)
            elif child.type == "extends_type_clause":
                for type_child in child.children:
                    if type_child.type == "identifier" or type_child.type == "type_identifier":
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

        return inheritance_list

    def _get_react_component_info(self, node) -> Optional[dict]:
        """
        Detect if a function is a React component and extract metadata.

        Heuristics:
        1. Function returns JSX elements (jsx_element, jsx_self_closing_element)
        2. Function is typed as React.FC, React.FunctionComponent, FC, etc.
        3. Function name starts with uppercase (React convention)

        Args:
            node: Tree-sitter node for the function declaration.

        Returns:
            Dict with React component info, or None if not a component.
        """
        # Get the function name - React components start with uppercase
        name = self._get_node_name(node)
        if not name or not name[0].isupper():
            return None

        # Check if it returns JSX
        if self._returns_jsx(node):
            props_interface = self._extract_props_interface(node)
            result = {"is_react_component": True}
            if props_interface:
                result["props_interface"] = props_interface
            return result

        # Check for React.FC type annotation
        type_annotation = self._get_react_fc_type(node)
        if type_annotation:
            result = {"is_react_component": True}
            if type_annotation != "FC":
                result["props_interface"] = type_annotation
            return result

        return None

    def _returns_jsx(self, node) -> bool:
        """Check if a function body contains JSX return statements."""
        jsx_types = {"jsx_element", "jsx_self_closing_element", "jsx_fragment"}

        def has_jsx(n):
            if n.type in jsx_types:
                return True
            for child in n.children:
                if has_jsx(child):
                    return True
            return False

        # Find the function body
        for child in node.children:
            if child.type == "statement_block":
                return has_jsx(child)
            # Arrow function with direct JSX return (no braces)
            if child.type in jsx_types:
                return True
            # Parenthesized expression containing JSX
            if child.type == "parenthesized_expression":
                return has_jsx(child)

        return False

    def _extract_props_interface(self, node) -> Optional[str]:
        """Extract the props interface name from function parameters."""
        for child in node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "required_parameter" or param.type == "optional_parameter":
                        # Look for type annotation
                        for param_child in param.children:
                            if param_child.type == "type_annotation":
                                for type_child in param_child.children:
                                    if type_child.type == "type_identifier":
                                        return type_child.text.decode("utf-8")
                                    # Handle object type inline
                                    if type_child.type == "object_type":
                                        return None  # Inline type, no named interface
        return None

    def _get_react_fc_type(self, node) -> Optional[str]:
        """
        Check if function is typed as React.FC or similar.

        Returns the props type if found, or 'FC' if it's a generic FC.
        """
        # For variable declarations with type annotation
        parent = node.parent
        if parent and parent.type == "variable_declarator":
            for child in parent.children:
                if child.type == "type_annotation":
                    type_text = child.text.decode("utf-8")
                    # Match React.FC<Props>, FC<Props>, React.FunctionComponent<Props>
                    if "FC" in type_text or "FunctionComponent" in type_text:
                        # Extract the generic type parameter
                        if "<" in type_text and ">" in type_text:
                            start = type_text.index("<") + 1
                            end = type_text.rindex(">")
                            return type_text[start:end].strip()
                        return "FC"
        return None

    def _extract_outgoing_api_calls(self, node) -> list[dict]:
        """
        Extract outgoing API calls (fetch, axios, etc.) from a function body.

        Detects:
        - fetch(url, options)
        - axios.get(url), axios.post(url), etc.
        - api.get(url), client.post(url), etc.

        Args:
            node: Tree-sitter node for the function declaration.

        Returns:
            List of dicts with 'url' and 'method' keys.
        """
        api_calls = []

        def visit_node(n):
            if n.type == "call_expression":
                api_call = self._parse_api_call(n)
                if api_call:
                    api_calls.append(api_call)

            for child in n.children:
                visit_node(child)

        # Find the function body
        for child in node.children:
            if child.type == "statement_block":
                visit_node(child)
                break
            # Arrow function with expression body
            if child.type == "call_expression":
                visit_node(child)

        return api_calls

    def _parse_api_call(self, call_node) -> Optional[dict]:
        """
        Parse a call expression to detect API calls.

        Args:
            call_node: Tree-sitter call_expression node.

        Returns:
            Dict with 'url' and 'method' keys, or None if not an API call.
        """
        if not call_node.children:
            return None

        callee_node = call_node.children[0]
        callee_name = self._get_callee_name(callee_node)

        if not callee_name:
            return None

        http_method = None
        url = ""

        # Check for fetch()
        if callee_name == "fetch":
            http_method = "GET"  # Default for fetch
            url = self._extract_url_from_arguments(call_node)
            # Check for method in options
            options_method = self._extract_fetch_method(call_node)
            if options_method:
                http_method = options_method

        # Check for axios.get(), axios.post(), etc.
        elif callee_name.startswith("axios.") or callee_name.endswith((".get", ".post", ".put", ".delete", ".patch")):
            method_part = callee_name.split(".")[-1].lower()
            if method_part in ("get", "post", "put", "delete", "patch"):
                http_method = method_part.upper()
                url = self._extract_url_from_arguments(call_node)

        # Check for $http (Angular) or http client patterns
        elif "$http" in callee_name or "http." in callee_name.lower():
            method_part = callee_name.split(".")[-1].lower()
            if method_part in ("get", "post", "put", "delete", "patch"):
                http_method = method_part.upper()
                url = self._extract_url_from_arguments(call_node)

        if http_method and url:
            return {"method": http_method, "url": url}

        return None

    def _extract_url_from_arguments(self, call_node) -> str:
        """Extract the URL argument from a call expression."""
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "string":
                        # Simple string literal
                        return arg.text.decode("utf-8").strip("'\"")
                    elif arg.type == "template_string":
                        # Template literal - normalize path params
                        return self._normalize_template_url(arg.text.decode("utf-8"))
        return ""

    def _normalize_template_url(self, template: str) -> str:
        """
        Normalize a template literal URL to a standard format.

        Converts `/api/users/${id}` to `/api/users/{id}`
        """
        import re
        # Remove backticks
        template = template.strip("`")
        # Convert ${...} to {...}
        template = re.sub(r'\$\{([^}]+)\}', r'{\1}', template)
        return template

    def _extract_fetch_method(self, call_node) -> Optional[str]:
        """Extract the HTTP method from fetch options."""
        for child in call_node.children:
            if child.type == "arguments":
                # Look for the options object (second argument)
                args = [c for c in child.children if c.type != ","]
                if len(args) >= 2:
                    options_node = args[1]
                    if options_node.type == "object":
                        for prop in options_node.children:
                            if prop.type == "pair":
                                key_node = prop.children[0] if prop.children else None
                                value_node = prop.children[-1] if len(prop.children) > 1 else None
                                if key_node and value_node:
                                    key = key_node.text.decode("utf-8").strip("'\"")
                                    if key == "method":
                                        return value_node.text.decode("utf-8").strip("'\"").upper()
        return None

    # =========================================================================
    # Express.js Route Extraction
    # =========================================================================

    def _detect_express_imports(self, node) -> None:
        """Detect Express imports and track app/router variables.

        Patterns detected:
        - import express from 'express'
        - const express = require('express')
        - const app = express()
        - const router = express.Router()
        """
        if node.type == "import_statement":
            source = self._get_import_source(node)
            if source and "express" in source.lower():
                # Track the default import name
                for child in node.children:
                    if child.type == "import_clause":
                        for sub in child.children:
                            if sub.type == "identifier":
                                self._express_vars.add(sub.text.decode("utf-8"))

        elif node.type == "lexical_declaration" or node.type == "variable_declaration":
            # Check for require('express') or express()
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    var_name = None
                    init_node = None

                    for child in declarator.children:
                        if child.type == "identifier":
                            var_name = child.text.decode("utf-8")
                        elif child.type == "call_expression":
                            init_node = child

                    if var_name and init_node:
                        callee = self._get_callee_name(init_node.children[0]) if init_node.children else None

                        # express() or require('express')
                        if callee == "express" or callee == "require":
                            if callee == "require":
                                # Check if requiring express
                                args = self._get_call_arguments(init_node)
                                if args and "express" in args[0].lower():
                                    self._express_vars.add(var_name)
                            else:
                                self._express_vars.add(var_name)

                        # express.Router()
                        elif callee == "express.Router":
                            self._express_vars.add(var_name)

                        # Check if it's an express/router variable calling Router()
                        elif callee and any(ev in callee for ev in self._express_vars):
                            self._express_vars.add(var_name)

    def _get_import_source(self, import_node) -> Optional[str]:
        """Get the source string from an import statement."""
        for child in import_node.children:
            if child.type == "string":
                return child.text.decode("utf-8").strip("'\"")
        return None

    def _get_call_arguments(self, call_node) -> list[str]:
        """Get string arguments from a call expression."""
        args = []
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "string":
                        args.append(arg.text.decode("utf-8").strip("'\""))
                    elif arg.type == "template_string":
                        args.append(arg.text.decode("utf-8").strip("`"))
        return args

    def _extract_express_routes(self, node, lines: list[str]) -> list[dict]:
        """Extract Express.js route definitions from a node.

        Detects patterns:
        - app.get('/users', handler)
        - router.post('/users/:id', handler)
        - app.use('/api', subRouter)
        - router.route('/users').get(handler).post(handler)

        Args:
            node: Tree-sitter node to analyze
            lines: Source code lines

        Returns:
            List of route dictionaries with method, path, framework
        """
        routes = []

        def visit(n):
            if n.type == "call_expression":
                route = self._parse_express_route_call(n)
                if route:
                    routes.append(route)

            for child in n.children:
                visit(child)

        visit(node)
        return routes

    def _parse_express_route_call(self, call_node) -> Optional[dict]:
        """Parse an Express route call expression.

        Args:
            call_node: Tree-sitter call_expression node

        Returns:
            Route dict or None
        """
        if not call_node.children:
            return None

        callee_node = call_node.children[0]
        callee = self._get_callee_name(callee_node)

        if not callee:
            return None

        # Check if it's a route method call (app.get, router.post, etc.)
        parts = callee.split(".")
        if len(parts) < 2:
            return None

        obj_name = parts[0]
        method_name = parts[-1].lower()

        # Verify it's an Express variable and an HTTP method
        is_express_var = obj_name in self._express_vars or obj_name in ("app", "router", "server")
        is_http_method = method_name in HTTP_METHODS

        if not is_express_var or not is_http_method:
            # Check for .use() middleware mounting
            if is_express_var and method_name == "use":
                return self._parse_express_use(call_node)
            return None

        # Extract the path argument
        path = self._extract_route_path_from_call(call_node)
        if not path:
            return None

        return {
            "method": method_name.upper(),
            "path": path,
            "framework": "express",
            "line": call_node.start_point[0] + 1,
        }

    def _parse_express_use(self, call_node) -> Optional[dict]:
        """Parse an Express .use() call for middleware/router mounting.

        Detects: app.use('/api', apiRouter)

        Returns:
            Route dict with 'use' method or None
        """
        args = self._get_call_arguments(call_node)
        if args:
            # First string argument is the base path
            return {
                "method": "USE",
                "path": args[0],
                "framework": "express",
                "is_mount": True,
                "line": call_node.start_point[0] + 1,
            }
        return None

    def _extract_route_path_from_call(self, call_node) -> Optional[str]:
        """Extract the route path from a call expression's arguments."""
        for child in call_node.children:
            if child.type == "arguments":
                for arg in child.children:
                    if arg.type == "string":
                        return arg.text.decode("utf-8").strip("'\"")
                    elif arg.type == "template_string":
                        return self._normalize_template_url(arg.text.decode("utf-8"))
        return None

    # =========================================================================
    # NestJS Route Extraction
    # =========================================================================

    def _extract_nestjs_routes(self, class_node, class_name: str) -> list[dict]:
        """Extract NestJS route definitions from a class.

        Detects patterns:
        - @Controller('users')
        - @Get(':id')
        - @Post()
        - @Put(':id')

        Args:
            class_node: Tree-sitter node for class declaration
            class_name: Name of the class

        Returns:
            List of route dictionaries
        """
        routes = []
        controller_path = ""

        # Find @Controller decorator on the class
        decorators = self._get_decorators(class_node)
        for dec in decorators:
            if dec["name"] == NESTJS_CONTROLLER_DECORATOR:
                controller_path = dec.get("arg", "")
                break

        # Scan methods for route decorators
        body = self._get_class_body(class_node)
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    method_name = self._get_method_name(child)
                    method_decorators = self._get_decorators(child)

                    for dec in method_decorators:
                        if dec["name"] in NESTJS_ROUTE_DECORATORS:
                            http_method = dec["name"].upper()
                            method_path = dec.get("arg", "")

                            # Combine controller path with method path
                            full_path = self._combine_paths(controller_path, method_path)

                            routes.append({
                                "method": http_method if http_method != "ALL" else "GET",
                                "path": full_path,
                                "framework": "nestjs",
                                "handler": method_name,
                                "line": child.start_point[0] + 1,
                            })

        return routes

    def _get_decorators(self, node) -> list[dict]:
        """Extract decorators from a node.

        Returns list of dicts with 'name' and optional 'arg'.
        """
        decorators = []

        # Look for decorator nodes preceding the target node
        parent = node.parent
        if not parent:
            return decorators

        # In tree-sitter-typescript, decorators are siblings before the decorated item
        found_node = False
        for sibling in parent.children:
            if sibling == node:
                found_node = True
                break
            if sibling.type == "decorator":
                dec_info = self._parse_decorator(sibling)
                if dec_info:
                    decorators.append(dec_info)

        return decorators

    def _parse_decorator(self, decorator_node) -> Optional[dict]:
        """Parse a decorator node to extract name and argument.

        @Controller('users') -> {"name": "Controller", "arg": "users"}
        @Get() -> {"name": "Get", "arg": ""}
        """
        for child in decorator_node.children:
            if child.type == "call_expression":
                callee = None
                arg = ""

                for sub in child.children:
                    if sub.type == "identifier":
                        callee = sub.text.decode("utf-8")
                    elif sub.type == "arguments":
                        # Get first string argument
                        for arg_child in sub.children:
                            if arg_child.type == "string":
                                arg = arg_child.text.decode("utf-8").strip("'\"")
                                break

                if callee:
                    return {"name": callee, "arg": arg}

            elif child.type == "identifier":
                # Decorator without parentheses: @Get
                return {"name": child.text.decode("utf-8"), "arg": ""}

        return None

    def _get_method_name(self, method_node) -> Optional[str]:
        """Get the name of a method from a method_definition node."""
        for child in method_node.children:
            if child.type == "property_identifier":
                return child.text.decode("utf-8")
        return None

    def _combine_paths(self, base: str, path: str) -> str:
        """Combine a base path with a relative path.

        Examples:
            ('users', ':id') -> '/users/:id'
            ('/api', '/users') -> '/api/users'
            ('', 'health') -> '/health'
        """
        # Normalize base
        if base and not base.startswith("/"):
            base = "/" + base
        if base and base.endswith("/"):
            base = base[:-1]

        # Normalize path
        if path and not path.startswith("/"):
            path = "/" + path
        if not path:
            path = ""

        combined = base + path
        return combined if combined else "/"

    def _extract_route_for_chunk(self, node, lines: list[str], chunk_type: ChunkType) -> Optional[dict]:
        """Extract route information for a code chunk.

        This is called during chunk creation to detect if the chunk
        defines an API route.

        Args:
            node: Tree-sitter node for the chunk
            lines: Source code lines
            chunk_type: Type of code chunk

        Returns:
            Route dict or None
        """
        # First, scan the file for Express imports
        root = node
        while root.parent:
            root = root.parent
        self._detect_express_imports(root)

        # Check for NestJS class with route decorators
        if chunk_type == ChunkType.CLASS:
            class_name = self._get_node_name(node)
            if class_name:
                routes = self._extract_nestjs_routes(node, class_name)
                if routes:
                    # Return the controller-level route info
                    decorators = self._get_decorators(node)
                    for dec in decorators:
                        if dec["name"] == NESTJS_CONTROLLER_DECORATOR:
                            return {
                                "method": "CONTROLLER",
                                "path": dec.get("arg", ""),
                                "framework": "nestjs",
                            }

        # Check for NestJS method with route decorators
        if chunk_type == ChunkType.METHOD:
            decorators = self._get_decorators(node)
            for dec in decorators:
                if dec["name"] in NESTJS_ROUTE_DECORATORS:
                    http_method = dec["name"].upper()
                    method_path = dec.get("arg", "")

                    # Try to get controller path from parent class
                    parent_class = self._find_parent_class(node)
                    controller_path = ""
                    if parent_class:
                        parent_decorators = self._get_decorators(parent_class)
                        for pdec in parent_decorators:
                            if pdec["name"] == NESTJS_CONTROLLER_DECORATOR:
                                controller_path = pdec.get("arg", "")
                                break

                    full_path = self._combine_paths(controller_path, method_path)

                    return {
                        "method": http_method if http_method != "ALL" else "GET",
                        "path": full_path,
                        "framework": "nestjs",
                    }

        # Check for Express-style function that IS a route handler
        # This requires analyzing the context where the function is used
        if chunk_type == ChunkType.FUNCTION:
            # Look for Express route patterns in the parent scope
            routes = self._extract_express_routes(root, lines)
            func_name = self._get_node_name(node)

            # Check if this function is referenced as a handler in any route
            for route in routes:
                if route.get("handler") == func_name:
                    return route

        return None

    def _find_parent_class(self, node) -> Optional[any]:
        """Find the parent class of a method node."""
        current = node.parent
        while current:
            if current.type == "class_declaration":
                return current
            current = current.parent
        return None
