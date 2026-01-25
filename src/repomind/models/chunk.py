"""
Code Chunk Model - The Fundamental Unit of Indexed Code.

This module defines the core data models used throughout RepoMind:

1. **CodeChunk**: A semantic unit of code (function, class, method)
   - Stored in ChromaDB with vector embeddings
   - Contains source code, metadata, and location info

2. **CallInfo**: Information about function/method calls
   - Used to build the call graph
   - Enables "who calls this?" analysis

3. **ParseResult**: Output from parsing a source file
   - Contains extracted chunks and call relationships

Data Flow:
    Source File → Parser → ParseResult → (CodeChunks, CallInfos)
                                ↓
                      EmbeddingService → Vectors
                                ↓
                      StorageService → ChromaDB + JSON

Example:
    # A parsed Python function becomes a CodeChunk:
    chunk = CodeChunk(
        id="sha256-hash",
        repo_name="Actions",
        file_path="src/handlers/auth.py",
        start_line=45,
        end_line=67,
        chunk_type=ChunkType.FUNCTION,
        name="validate_token",
        content="def validate_token(token: str) -> bool: ...",
        signature="def validate_token(token: str) -> bool",
        docstring="Validate a JWT token and return validity.",
        language="python",
    )

Author: RepoMind Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class CallInfo:
    """
    Information about a function or method call extracted during parsing.

    This represents a single call relationship: "function A calls function B".
    Used to build the call graph that enables impact analysis.

    Attributes:
        caller_qualified_name: Full name of the calling function
                               (e.g., "ClassName.method_name" or "function_name")
        callee_name: Name of the function being called
                     (may be simple like "print" or qualified like "os.path.join")
        caller_file: Path to the file containing the call
        caller_line: Line number where the call occurs
        call_type: Type of call (direct, method, constructor, etc.)

    Example:
        # For this Python code:
        # class AuthService:
        #     def validate(self, token):
        #         result = jwt.decode(token)  # Line 25

        call = CallInfo(
            caller_qualified_name="AuthService.validate",
            callee_name="jwt.decode",
            caller_file="src/auth.py",
            caller_line=25,
            call_type="method"
        )
    """

    caller_qualified_name: str
    callee_name: str
    caller_file: str
    caller_line: int
    call_type: str = "direct"


@dataclass
class InheritanceInfo:
    """
    Information about an inheritance relationship extracted during parsing.

    This represents a class hierarchy relationship: "Child extends/implements Parent".
    Used to build the inheritance graph for finding implementations.

    Attributes:
        child_name: Simple name of the child class/interface
        child_qualified: Fully qualified name of the child
        parent_name: Name of the parent class/interface being extended/implemented
        relation_type: Type of relationship ("extends" or "implements")
        file_path: Path to the file containing the child class
        line_number: Line number where the class is defined

    Example:
        # For this Python code:
        # class AuthService(BaseService):  # Line 10
        #     pass

        info = InheritanceInfo(
            child_name="AuthService",
            child_qualified="AuthService",
            parent_name="BaseService",
            relation_type="extends",
            file_path="src/auth.py",
            line_number=10
        )
    """

    child_name: str
    child_qualified: str
    parent_name: str
    relation_type: str  # "extends" | "implements"
    file_path: str
    line_number: int


@dataclass
class ParseResult:
    """
    Result of parsing a single source file.

    Contains the semantic code chunks (for embedding), call relationships
    (for the call graph), and inheritance relationships (for finding implementations).

    Attributes:
        chunks: List of CodeChunk objects extracted from the file.
        calls: List of CallInfo objects representing function calls.
        inheritance: List of InheritanceInfo objects representing class hierarchies.

    Example:
        # After parsing a Python file with 2 functions:
        result = ParseResult(
            chunks=[function_a_chunk, function_b_chunk],
            calls=[call_from_a_to_b, call_from_b_to_print],
            inheritance=[child_extends_parent]
        )
    """

    chunks: list["CodeChunk"] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    inheritance: list[InheritanceInfo] = field(default_factory=list)


class ChunkType(str, Enum):
    """
    Types of code constructs that can be indexed.

    Each type represents a distinct semantic unit in source code.
    The type affects how the chunk is displayed and filtered.

    Values:
        MODULE: A complete module or file
        CLASS: A class definition
        INTERFACE: An interface (TypeScript/Java)
        FUNCTION: A standalone function
        METHOD: A method within a class
        CONSTRUCTOR: A class constructor
        PROPERTY: A class property or field
        CONSTANT: A constant or enum value
        IMPORT: An import statement
        COMMENT: A significant comment block
        DOCUMENTATION: Documentation (README, etc.)
    """

    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    CONSTANT = "constant"
    IMPORT = "import"
    COMMENT = "comment"
    DOCUMENTATION = "documentation"


class CodeChunk(BaseModel):
    """
    A semantic unit of code extracted from a repository.

    This is the fundamental data model that gets embedded and stored.
    Each chunk represents a meaningful code construct (function, class,
    method, etc.) along with its metadata and context.

    The chunk is designed to contain everything needed for:
    - Generating meaningful embeddings
    - Displaying search results
    - Understanding code context
    - Building call graphs

    Attributes:
        id: Unique identifier (SHA256 hash of repo+file+name+type)
        repo_name: Repository name (e.g., 'Actions-Discovery')
        file_path: Relative path within the repository
        start_line: 1-indexed line where the chunk starts
        end_line: 1-indexed line where the chunk ends
        chunk_type: Type of code construct (function, class, etc.)
        name: Name of the construct (function name, class name, etc.)
        content: Full source code of the chunk
        signature: Function/method signature if applicable
        docstring: Documentation string if present
        parent_name: Name of parent construct (e.g., class for methods)
        parent_type: Type of parent construct
        language: Programming language (python, java, typescript)
        imports: List of imports used by this chunk

    Example:
        chunk = CodeChunk(
            id="a1b2c3...",
            repo_name="Actions",
            file_path="src/services/auth.py",
            start_line=45,
            end_line=67,
            chunk_type=ChunkType.METHOD,
            name="validate_token",
            content="def validate_token(self, token): ...",
            signature="def validate_token(self, token: str) -> bool",
            docstring="Validate a JWT token.",
            parent_name="AuthService",
            parent_type=ChunkType.CLASS,
            language="python",
        )
    """

    # ========================================================================
    # Identity Fields
    # ========================================================================

    id: str = Field(
        description="Unique identifier (SHA256 hash of repo+file+name+type)"
    )
    repo_name: str = Field(
        description="Repository name (e.g., 'Actions-Discovery')"
    )
    file_path: str = Field(
        description="Relative path within the repository"
    )

    # ========================================================================
    # Location Fields
    # ========================================================================

    start_line: int = Field(
        description="1-indexed line number where the chunk starts"
    )
    end_line: int = Field(
        description="1-indexed line number where the chunk ends"
    )

    # ========================================================================
    # Content Fields
    # ========================================================================

    chunk_type: ChunkType = Field(
        description="Type of code construct (function, class, method, etc.)"
    )
    name: str = Field(
        description="Name of the construct (function name, class name, etc.)"
    )
    content: str = Field(
        description="Full source code of the chunk"
    )
    signature: Optional[str] = Field(
        default=None,
        description="Function/method signature if applicable"
    )
    docstring: Optional[str] = Field(
        default=None,
        description="Documentation string if present"
    )

    # ========================================================================
    # Hierarchy Fields
    # ========================================================================

    parent_name: Optional[str] = Field(
        default=None,
        description="Name of parent construct (e.g., class name for methods)"
    )
    parent_type: Optional[ChunkType] = Field(
        default=None,
        description="Type of parent construct"
    )

    # ========================================================================
    # Metadata Fields
    # ========================================================================

    language: str = Field(
        description="Programming language (python, java, typescript, javascript)"
    )
    imports: list[str] = Field(
        default_factory=list,
        description="List of imports used by this chunk"
    )

    # Semantic enrichment (populated during indexing)
    summary: Optional[str] = Field(default=None, description="AI-generated summary of what this code does")

    def to_embedding_text(self) -> str:
        """
        Generate optimized text representation for embedding.

        The text is structured to maximize semantic search quality by:
        1. Starting with a purpose-focused description
        2. Including the signature and docstring prominently
        3. Adding extracted keywords from identifier names
        4. Minimizing noise from boilerplate code

        Returns:
            Optimized text string for embedding generation.
        """
        parts = []

        # Start with semantic purpose (most important for search)
        purpose = self._generate_purpose_description()
        if purpose:
            parts.append(purpose)

        # Include docstring (very valuable for semantic matching)
        if self.docstring:
            parts.append(f"Description: {self.docstring.strip()}")

        # Include signature (important for understanding function behavior)
        if self.signature:
            parts.append(f"Signature: {self.signature}")

        # Include AI-generated summary if available
        if self.summary:
            parts.append(f"Summary: {self.summary}")

        # Extract keywords from identifier name (camelCase/snake_case)
        keywords = self._extract_keywords_from_name()
        if keywords:
            parts.append(f"Keywords: {' '.join(keywords)}")

        # Add minimal context (file name is useful, full path less so)
        file_name = self.file_path.split('/')[-1] if '/' in self.file_path else self.file_path
        parts.append(f"File: {file_name}")
        parts.append(f"Language: {self.language}")

        # Add code content (truncated for very long functions)
        code_content = self.content
        if len(code_content) > 2000:
            # Keep first and last parts for very long code
            code_content = code_content[:1500] + "\n... (truncated) ...\n" + code_content[-400:]
        parts.append(f"Code:\n{code_content}")

        return "\n".join(parts)

    def _generate_purpose_description(self) -> str:
        """
        Generate a human-readable purpose description based on chunk type and name.

        Returns:
            Purpose description string.
        """
        name_words = self._extract_keywords_from_name()
        name_readable = ' '.join(name_words) if name_words else self.name

        type_descriptions = {
            ChunkType.FUNCTION: f"Function that handles {name_readable}",
            ChunkType.METHOD: f"Method for {name_readable}" + (f" in {self.parent_name}" if self.parent_name else ""),
            ChunkType.CLASS: f"Class representing {name_readable}",
            ChunkType.INTERFACE: f"Interface defining {name_readable}",
            ChunkType.CONSTRUCTOR: f"Constructor for {self.parent_name or name_readable}",
            ChunkType.MODULE: f"Module containing {name_readable}",
            ChunkType.PROPERTY: f"Property {name_readable}" + (f" of {self.parent_name}" if self.parent_name else ""),
            ChunkType.CONSTANT: f"Constant {name_readable}",
            ChunkType.IMPORT: f"Import statement for {name_readable}",
        }

        return type_descriptions.get(self.chunk_type, f"{self.chunk_type.value} {name_readable}")

    def _extract_keywords_from_name(self) -> list[str]:
        """
        Extract meaningful keywords from identifier name.

        Handles common naming conventions:
        - camelCase
        - PascalCase
        - snake_case
        - SCREAMING_SNAKE_CASE

        Returns:
            List of lowercase keywords.
        """
        import re

        name = self.name

        # Handle snake_case and SCREAMING_SNAKE_CASE
        if '_' in name:
            words = name.split('_')
        else:
            # Handle camelCase and PascalCase
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)

        # Filter and normalize
        keywords = [
            word.lower()
            for word in words
            if word and len(word) > 1 and word.lower() not in {'get', 'set', 'is', 'has', 'the', 'a', 'an'}
        ]

        return keywords

    def get_qualified_name(self) -> str:
        """Get fully qualified name (e.g., ClassName.method_name)."""
        if self.parent_name:
            return f"{self.parent_name}.{self.name}"
        return self.name
