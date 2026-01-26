"""
LSP Client Manager - Language Server Protocol Integration.

This module provides integration with Language Server Protocol (LSP) servers
for compiler-grade precision in code analysis. It supports:

1. Multiple language servers (Python, TypeScript, Java, Go, etc.)
2. Symbol resolution with full type information
3. Cross-file reference finding
4. Type hierarchy analysis

LSP provides advantages over tree-sitter:
- Resolves overloaded methods correctly
- Handles generic type parameters
- Understands import aliasing
- Tracks dynamic dispatch targets

Architecture:
    LSPClientManager
        ├── manages multiple language servers
        ├── routes requests by file type
        └── handles server lifecycle

Usage:
    manager = LSPClientManager()
    await manager.initialize("/path/to/project")

    # Find all references to a symbol
    refs = await manager.find_references("auth.py", 45, 10)

    # Go to definition
    defn = await manager.goto_definition("main.py", 30, 15)

    await manager.shutdown()

Author: RepoMind Team
"""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class LSPServerType(str, Enum):
    """Supported LSP server types."""
    PYLSP = "pylsp"           # Python (python-lsp-server)
    PYRIGHT = "pyright"       # Python (Pyright/Pylance)
    TSSERVER = "tsserver"     # TypeScript/JavaScript
    JDTLS = "jdtls"           # Java (Eclipse JDT)
    GOPLS = "gopls"           # Go
    RUST_ANALYZER = "rust-analyzer"  # Rust
    CLANGD = "clangd"         # C/C++


@dataclass
class LSPPosition:
    """A position in a text document (0-indexed)."""
    line: int
    character: int

    def to_dict(self) -> dict:
        return {"line": self.line, "character": self.character}


@dataclass
class LSPRange:
    """A range in a text document."""
    start: LSPPosition
    end: LSPPosition

    def to_dict(self) -> dict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}


@dataclass
class LSPLocation:
    """A location in a document."""
    uri: str
    range: LSPRange

    @classmethod
    def from_dict(cls, data: dict) -> "LSPLocation":
        return cls(
            uri=data["uri"],
            range=LSPRange(
                start=LSPPosition(
                    line=data["range"]["start"]["line"],
                    character=data["range"]["start"]["character"],
                ),
                end=LSPPosition(
                    line=data["range"]["end"]["line"],
                    character=data["range"]["end"]["character"],
                ),
            ),
        )


@dataclass
class LSPSymbol:
    """A symbol found by LSP."""
    name: str
    kind: int  # LSP SymbolKind
    location: LSPLocation
    container_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "LSPSymbol":
        return cls(
            name=data["name"],
            kind=data["kind"],
            location=LSPLocation.from_dict(data["location"]),
            container_name=data.get("containerName"),
        )


@dataclass
class LSPServerConfig:
    """Configuration for an LSP server."""
    server_type: LSPServerType
    command: list[str]
    file_extensions: list[str]
    initialization_options: dict = field(default_factory=dict)
    workspace_config: dict = field(default_factory=dict)


# Default server configurations
DEFAULT_SERVER_CONFIGS = {
    "python": LSPServerConfig(
        server_type=LSPServerType.PYLSP,
        command=["pylsp"],
        file_extensions=[".py"],
        initialization_options={},
        workspace_config={
            "pylsp": {
                "plugins": {
                    "pycodestyle": {"enabled": False},
                    "mccabe": {"enabled": False},
                    "pyflakes": {"enabled": True},
                    "rope_completion": {"enabled": False},
                }
            }
        },
    ),
    "typescript": LSPServerConfig(
        server_type=LSPServerType.TSSERVER,
        command=["typescript-language-server", "--stdio"],
        file_extensions=[".ts", ".tsx", ".js", ".jsx"],
        initialization_options={},
    ),
    "java": LSPServerConfig(
        server_type=LSPServerType.JDTLS,
        command=["jdtls"],
        file_extensions=[".java"],
        initialization_options={},
    ),
    "go": LSPServerConfig(
        server_type=LSPServerType.GOPLS,
        command=["gopls", "serve"],
        file_extensions=[".go"],
        initialization_options={},
    ),
}


class LSPClientManager:
    """
    Manages multiple LSP server connections.

    Provides a unified interface for interacting with different
    language servers, routing requests based on file type.
    """

    def __init__(self, configs: Optional[dict[str, LSPServerConfig]] = None):
        """
        Initialize the LSP client manager.

        Args:
            configs: Optional custom server configurations.
        """
        self.configs = configs or DEFAULT_SERVER_CONFIGS
        self._servers: dict[str, "LSPServerConnection"] = {}
        self._workspace_root: Optional[Path] = None
        self._initialized = False

    async def initialize(self, workspace_root: str) -> bool:
        """
        Initialize LSP servers for a workspace.

        Args:
            workspace_root: Root directory of the project.

        Returns:
            True if at least one server initialized successfully.
        """
        self._workspace_root = Path(workspace_root)
        success_count = 0

        for language, config in self.configs.items():
            try:
                server = LSPServerConnection(config, self._workspace_root)
                if await server.start():
                    self._servers[language] = server
                    success_count += 1
                    logger.info(f"LSP server started for {language}")
            except Exception as e:
                logger.warning(f"Failed to start LSP server for {language}: {e}")

        self._initialized = success_count > 0
        return self._initialized

    async def shutdown(self):
        """Shutdown all LSP servers."""
        for language, server in self._servers.items():
            try:
                await server.shutdown()
                logger.info(f"LSP server stopped for {language}")
            except Exception as e:
                logger.warning(f"Error stopping LSP server for {language}: {e}")
        self._servers.clear()
        self._initialized = False

    def _get_server_for_file(self, file_path: str) -> Optional["LSPServerConnection"]:
        """Get the appropriate LSP server for a file."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        for language, config in self.configs.items():
            if suffix in config.file_extensions and language in self._servers:
                return self._servers[language]
        return None

    def is_available(self, file_path: str) -> bool:
        """Check if LSP is available for a file type."""
        return self._get_server_for_file(file_path) is not None

    async def find_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[LSPLocation]:
        """
        Find all references to a symbol at the given position.

        Args:
            file_path: Path to the file.
            line: Line number (0-indexed).
            character: Character position (0-indexed).
            include_declaration: Include the declaration in results.

        Returns:
            List of locations where the symbol is referenced.
        """
        server = self._get_server_for_file(file_path)
        if not server:
            return []

        return await server.find_references(
            file_path, line, character, include_declaration
        )

    async def goto_definition(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> Optional[LSPLocation]:
        """
        Go to the definition of a symbol at the given position.

        Args:
            file_path: Path to the file.
            line: Line number (0-indexed).
            character: Character position (0-indexed).

        Returns:
            Location of the definition, or None if not found.
        """
        server = self._get_server_for_file(file_path)
        if not server:
            return None

        return await server.goto_definition(file_path, line, character)

    async def find_implementations(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> list[LSPLocation]:
        """
        Find implementations of an interface/abstract method.

        Args:
            file_path: Path to the file.
            line: Line number (0-indexed).
            character: Character position (0-indexed).

        Returns:
            List of implementation locations.
        """
        server = self._get_server_for_file(file_path)
        if not server:
            return []

        return await server.find_implementations(file_path, line, character)

    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[LSPSymbol]:
        """
        Get all symbols in a document.

        Args:
            file_path: Path to the file.

        Returns:
            List of symbols in the document.
        """
        server = self._get_server_for_file(file_path)
        if not server:
            return []

        return await server.get_document_symbols(file_path)

    async def get_type_hierarchy(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> dict[str, Any]:
        """
        Get the type hierarchy for a symbol.

        Args:
            file_path: Path to the file.
            line: Line number (0-indexed).
            character: Character position (0-indexed).

        Returns:
            Type hierarchy information.
        """
        server = self._get_server_for_file(file_path)
        if not server:
            return {}

        return await server.get_type_hierarchy(file_path, line, character)


class LSPServerConnection:
    """
    Connection to a single LSP server process.

    Handles the JSON-RPC communication protocol with the server.
    """

    def __init__(self, config: LSPServerConfig, workspace_root: Path):
        """
        Initialize an LSP server connection.

        Args:
            config: Server configuration.
            workspace_root: Root directory of the workspace.
        """
        self.config = config
        self.workspace_root = workspace_root
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """
        Start the LSP server process.

        Returns:
            True if server started successfully.
        """
        try:
            # Check if command exists
            cmd = self.config.command[0]
            if not self._command_exists(cmd):
                logger.warning(f"LSP command not found: {cmd}")
                return False

            self._process = subprocess.Popen(
                self.config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace_root),
            )

            # Start reading responses
            self._reader_task = asyncio.create_task(self._read_responses())

            # Initialize the server
            return await self._initialize()

        except Exception as e:
            logger.error(f"Failed to start LSP server: {e}")
            return False

    def _command_exists(self, cmd: str) -> bool:
        """Check if a command exists on the system."""
        import shutil
        return shutil.which(cmd) is not None

    async def _initialize(self) -> bool:
        """Initialize the LSP server with the workspace."""
        try:
            result = await self._send_request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": self.workspace_root.as_uri(),
                    "capabilities": {
                        "textDocument": {
                            "references": {"dynamicRegistration": False},
                            "definition": {"dynamicRegistration": False},
                            "implementation": {"dynamicRegistration": False},
                            "documentSymbol": {"dynamicRegistration": False},
                            "typeHierarchy": {"dynamicRegistration": False},
                        },
                        "workspace": {
                            "workspaceFolders": True,
                        },
                    },
                    "initializationOptions": self.config.initialization_options,
                    "workspaceFolders": [
                        {
                            "uri": self.workspace_root.as_uri(),
                            "name": self.workspace_root.name,
                        }
                    ],
                },
            )

            if result:
                # Send initialized notification
                await self._send_notification("initialized", {})
                return True
            return False

        except Exception as e:
            logger.error(f"LSP initialization failed: {e}")
            return False

    async def shutdown(self):
        """Shutdown the LSP server."""
        if self._process:
            try:
                await self._send_request("shutdown", {})
                await self._send_notification("exit", {})
            except Exception:
                pass
            finally:
                self._process.terminate()
                self._process = None

        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None

    async def find_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[LSPLocation]:
        """Find all references to a symbol."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        result = await self._send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration},
            },
        )

        if not result:
            return []

        return [LSPLocation.from_dict(loc) for loc in result]

    async def goto_definition(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> Optional[LSPLocation]:
        """Go to the definition of a symbol."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        result = await self._send_request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
            },
        )

        if not result:
            return None

        # Result can be a single location or an array
        if isinstance(result, list):
            return LSPLocation.from_dict(result[0]) if result else None
        return LSPLocation.from_dict(result)

    async def find_implementations(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> list[LSPLocation]:
        """Find implementations of an interface/abstract method."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        result = await self._send_request(
            "textDocument/implementation",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
            },
        )

        if not result:
            return []

        if isinstance(result, list):
            return [LSPLocation.from_dict(loc) for loc in result]
        return [LSPLocation.from_dict(result)]

    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[LSPSymbol]:
        """Get all symbols in a document."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        result = await self._send_request(
            "textDocument/documentSymbol",
            {
                "textDocument": {"uri": path.as_uri()},
            },
        )

        if not result:
            return []

        symbols = []
        for item in result:
            # Handle both DocumentSymbol and SymbolInformation
            if "location" in item:
                symbols.append(LSPSymbol.from_dict(item))
            elif "range" in item:
                # DocumentSymbol format - convert to SymbolInformation-like
                symbols.append(
                    LSPSymbol(
                        name=item["name"],
                        kind=item["kind"],
                        location=LSPLocation(
                            uri=path.as_uri(),
                            range=LSPRange(
                                start=LSPPosition(
                                    line=item["range"]["start"]["line"],
                                    character=item["range"]["start"]["character"],
                                ),
                                end=LSPPosition(
                                    line=item["range"]["end"]["line"],
                                    character=item["range"]["end"]["character"],
                                ),
                            ),
                        ),
                        container_name=None,
                    )
                )
        return symbols

    async def get_type_hierarchy(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> dict[str, Any]:
        """Get the type hierarchy for a symbol."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        # Try to prepare type hierarchy
        prepare_result = await self._send_request(
            "textDocument/prepareTypeHierarchy",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
            },
        )

        if not prepare_result:
            return {}

        hierarchy = {"item": prepare_result}

        # Get supertypes
        if prepare_result:
            item = prepare_result[0] if isinstance(prepare_result, list) else prepare_result
            supertypes = await self._send_request(
                "typeHierarchy/supertypes",
                {"item": item},
            )
            hierarchy["supertypes"] = supertypes or []

            # Get subtypes
            subtypes = await self._send_request(
                "typeHierarchy/subtypes",
                {"item": item},
            )
            hierarchy["subtypes"] = subtypes or []

        return hierarchy

    async def _send_request(
        self,
        method: str,
        params: dict,
        timeout: float = 30.0,
    ) -> Optional[Any]:
        """Send a request to the LSP server."""
        if not self._process or self._process.poll() is not None:
            return None

        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            self._write_message(message)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"LSP request timed out: {method}")
            return None
        except Exception as e:
            logger.error(f"LSP request failed: {e}")
            return None
        finally:
            self._pending_requests.pop(request_id, None)

    async def _send_notification(self, method: str, params: dict):
        """Send a notification to the LSP server (no response expected)."""
        if not self._process or self._process.poll() is not None:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._write_message(message)

    def _write_message(self, message: dict):
        """Write a JSON-RPC message to the server."""
        if not self._process or not self._process.stdin:
            return

        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self._process.stdin.write(header.encode())
        self._process.stdin.write(content.encode())
        self._process.stdin.flush()

    async def _read_responses(self):
        """Background task to read responses from the server."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self._process.poll() is None:
                # Read header
                header_line = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.readline
                )
                if not header_line:
                    break

                # Parse Content-Length
                header = header_line.decode().strip()
                if not header.startswith("Content-Length:"):
                    continue

                content_length = int(header.split(":")[1].strip())

                # Skip empty line
                await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.readline
                )

                # Read content
                content = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._process.stdout.read(content_length)
                )

                # Parse and handle response
                try:
                    response = json.loads(content.decode())
                    self._handle_response(response)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from LSP server")

        except Exception as e:
            logger.error(f"Error reading LSP responses: {e}")

    def _handle_response(self, response: dict):
        """Handle a response from the LSP server."""
        if "id" in response:
            request_id = response["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests[request_id]
                if "error" in response:
                    future.set_exception(
                        Exception(response["error"].get("message", "Unknown error"))
                    )
                else:
                    future.set_result(response.get("result"))
