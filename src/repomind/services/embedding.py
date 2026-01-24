"""
Embedding Service for Vector Generation.

This module provides the embedding generation service that converts
code chunks into vector representations for semantic search.

Supported Backends:
    1. **Local Models** (Privacy-First):
       - microsoft/codebert-base (default): Best for code understanding
       - all-MiniLM-L6-v2: Faster, smaller dimensions
       - all-mpnet-base-v2: Higher quality general embeddings

    2. **API Models** (Require API Keys):
       - voyage-code-2: Voyage AI's code-specialized model
       - text-embedding-3-small: OpenAI's efficient model
       - text-embedding-3-large: OpenAI's high-quality model

    3. **Mock** (Testing):
       - Deterministic hash-based vectors for reproducible tests

Privacy:
    By default, this service uses local models that process everything
    on your machine. No code is sent to external services unless you
    explicitly configure an API-based model.

Performance:
    - Local models: ~100ms per batch of 32 chunks
    - API models: ~200-500ms per batch (network latency)
    - Mock: <1ms per batch

Usage:
    from repomind.services.embedding import EmbeddingService

    # Default local embedding (privacy-first)
    service = EmbeddingService()
    embeddings = service.embed_chunks(chunks)

    # Query embedding for search
    query_vector = service.embed_query("authentication middleware")

    # Check if local
    if service.is_local:
        print("Running 100% locally - no API calls")

Author: RepoMind Team
"""

import os
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSION,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    LOCAL_EMBEDDING_MODELS,
    EmbeddingModelName,
)
from ..logging import get_logger, log_operation_start, log_operation_end
from ..models.chunk import CodeChunk


# Module logger
logger = get_logger(__name__)

# Rich console for progress display
console = Console()

# Global model cache for performance (singleton pattern)
# Prevents reloading the same model multiple times
_model_cache: dict[str, object] = {}


class EmbeddingModel(str, Enum):
    """
    Available embedding models for code vectorization.

    Choose based on your needs:
    - Privacy: Use LOCAL_* models (no API calls)
    - Quality: Use LOCAL_BGE_BASE for best code search results
    - Speed: Use LOCAL_MINILM for fastest processing
    - Testing: Use MOCK for deterministic results
    """

    # Local models (no API calls, 100% private)
    LOCAL_BGE_BASE = EmbeddingModelName.BGE_BASE.value
    LOCAL_BGE_LARGE = EmbeddingModelName.BGE_LARGE.value
    LOCAL_CODEBERT = EmbeddingModelName.CODEBERT.value
    LOCAL_MINILM = EmbeddingModelName.MINILM.value
    LOCAL_MPNET = EmbeddingModelName.MPNET.value

    # API-based models
    VOYAGE_CODE = EmbeddingModelName.VOYAGE_CODE.value
    OPENAI_SMALL = EmbeddingModelName.OPENAI_SMALL.value
    OPENAI_LARGE = EmbeddingModelName.OPENAI_LARGE.value

    # Testing
    MOCK = EmbeddingModelName.MOCK.value


class EmbeddingService:
    """
    Service for generating vector embeddings from code.

    This service converts code chunks into dense vector representations
    that capture semantic meaning, enabling similarity-based search.

    Privacy First:
        By default, uses local sentence-transformers models that run
        entirely on your machine. No code is sent to external services.

    Automatic Batching:
        Processes chunks in batches for memory efficiency and
        progress tracking on large repositories.

    Multi-Backend Support:
        Supports local models, Voyage AI, OpenAI, and mock embeddings
        with a consistent interface.

    Attributes:
        model: Name of the embedding model being used.
        dimension: Vector dimension produced by the model.
        is_local: Whether the model runs locally.

    Example:
        # Initialize with default CodeBERT model
        embedding_service = EmbeddingService()

        # Generate embeddings for code chunks
        embeddings = embedding_service.embed_chunks(chunks)

        # Generate embedding for a search query
        query_vector = embedding_service.embed_query("error handling")

        # Use embeddings for similarity search
        results = storage.search(query_vector, n_results=10)
    """

    def __init__(self, model: str | EmbeddingModel = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the embedding service with the specified model.

        Args:
            model: Embedding model to use. Options:

                **Local Models** (privacy-first, no API required):
                - "BAAI/bge-base-en-v1.5" (default): Best for code search
                - "BAAI/bge-large-en-v1.5": Higher quality, slower
                - "microsoft/codebert-base": Good for code understanding
                - "all-MiniLM-L6-v2": Fast, smaller dimensions
                - "all-mpnet-base-v2": Higher quality general

                **API Models** (require API keys):
                - "voyage-code-2": Voyage AI (VOYAGE_API_KEY)
                - "text-embedding-3-small": OpenAI (OPENAI_API_KEY)

                **Testing**:
                - "mock": Deterministic hash-based embeddings

        Example:
            # Default (BGE, local)
            service = EmbeddingService()

            # Specific model
            service = EmbeddingService("all-MiniLM-L6-v2")

            # API model
            service = EmbeddingService("voyage-code-2")
        """
        self.model = model if isinstance(model, str) else model.value
        self._client: Optional[object] = None
        self._dimension: Optional[int] = None

        # BGE models require special handling for queries
        self._is_bge_model = "bge" in self.model.lower()
        self._query_instruction = "Represent this code search query: " if self._is_bge_model else ""

        logger.info(
            "Embedding service initialized",
            extra={
                "model": self.model,
                "is_local": self.is_local,
                "is_bge": self._is_bge_model,
                "dimension": self.dimension,
            }
        )

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension for the current model.

        Returns:
            Number of dimensions in the embedding vectors.
        """
        if self._dimension:
            return self._dimension
        return EMBEDDING_DIMENSIONS.get(self.model, DEFAULT_EMBEDDING_DIMENSION)

    @property
    def is_local(self) -> bool:
        """
        Check if the current model runs locally.

        Local models:
        - Don't require API keys
        - Don't send data over the network
        - Are 100% private

        Returns:
            True if the model runs entirely locally.
        """
        return self.model in LOCAL_EMBEDDING_MODELS

    def embed_chunks(self, chunks: list[CodeChunk], show_progress: bool = True) -> list[list[float]]:
        """Generate embeddings for a list of code chunks.

        Args:
            chunks: List of CodeChunk objects to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (same order as input chunks)
        """
        texts = [chunk.to_embedding_text() for chunk in chunks]
        return self.embed_texts(texts, show_progress=show_progress)

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.model == "mock" or self.model == EmbeddingModel.MOCK.value:
            return self._embed_mock(texts)
        elif self.is_local:
            return self._embed_local(texts, show_progress=show_progress)
        elif self.model.startswith("voyage"):
            return self._embed_voyage(texts)
        elif self.model.startswith("text-embedding"):
            return self._embed_openai(texts)
        else:
            # Default to local
            return self._embed_local(texts, show_progress=show_progress)

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        For BGE models, automatically prepends the query instruction
        for better retrieval performance.

        Args:
            query: Search query text

        Returns:
            Embedding vector for the query
        """
        # Expand query for better semantic matching
        expanded_query = self._expand_query(query)

        # For BGE models, prepend instruction for better retrieval
        if self._is_bge_model and self._query_instruction:
            query_text = f"{self._query_instruction}{expanded_query}"
        else:
            query_text = expanded_query

        embeddings = self.embed_texts([query_text], show_progress=False)
        return embeddings[0]

    def _expand_query(self, query: str) -> str:
        """
        Expand query with code-related synonyms for better matching.

        Args:
            query: Original search query

        Returns:
            Expanded query string
        """
        # Common code search synonyms
        expansions = {
            "auth": "authentication authorization login",
            "error": "exception handling catch try",
            "validate": "validation check verify",
            "fetch": "get retrieve request api",
            "save": "store persist write",
            "delete": "remove destroy",
            "create": "new initialize construct",
            "update": "modify change edit",
            "parse": "deserialize decode extract",
            "format": "serialize encode stringify",
            "config": "configuration settings options",
            "middleware": "interceptor handler filter",
            "event": "listener callback handler trigger",
            "api": "endpoint route handler service",
            "test": "spec unit integration mock",
            "util": "utility helper function",
            "cache": "memoize store temporary",
        }

        query_lower = query.lower()
        extra_terms = []

        for term, synonyms in expansions.items():
            if term in query_lower:
                extra_terms.append(synonyms)

        if extra_terms:
            return f"{query} ({' '.join(extra_terms)})"
        return query

    def _embed_local(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers model.

        100% private - no data leaves your machine.

        Embeddings are L2-normalized for optimal cosine similarity search.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        if self._client is None:
            console.print(f"[dim]Loading local embedding model: {self.model}[/dim]")
            self._client = SentenceTransformer(self.model)
            self._dimension = self._client.get_sentence_embedding_dimension()
            console.print(f"[dim]Model loaded. Dimension: {self._dimension}[/dim]")

        # Process in batches for memory efficiency
        batch_size = 32
        all_embeddings = []

        if show_progress and len(texts) > batch_size:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating embeddings...", total=len(texts))

                for i in range(0, len(texts), batch_size):
                    batch = texts[i: i + batch_size]
                    # Normalize embeddings for cosine similarity
                    embeddings = self._client.encode(
                        batch,
                        show_progress_bar=False,
                        normalize_embeddings=True,  # L2 normalize for cosine similarity
                    )
                    all_embeddings.extend([emb.tolist() for emb in embeddings])
                    progress.update(task, advance=len(batch))
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            all_embeddings = [emb.tolist() for emb in embeddings]

        return all_embeddings

    def _embed_voyage(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Voyage AI API."""
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai is required for Voyage embeddings. "
                "Install with: pip install voyageai"
            )

        if self._client is None:
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError(
                    "VOYAGE_API_KEY environment variable not set. "
                    "For private/local embeddings, use model='all-MiniLM-L6-v2' instead."
                )
            self._client = voyageai.Client(api_key=api_key)

        batch_size = 128
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            result = self._client.embed(batch, model=self.model, input_type="document")
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "For private/local embeddings, use model='all-MiniLM-L6-v2' instead."
                )
            self._client = OpenAI()

        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self.model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_mock(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings based on text hash.

        For testing only - does not capture semantic meaning.
        """
        import hashlib

        embeddings = []
        for text in texts:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(self.dimension):
                byte_idx = i % len(hash_bytes)
                value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
                embedding.append(value)
            embeddings.append(embedding)

        return embeddings


class LocalEmbeddingService(EmbeddingService):
    """Convenience class for local-only embeddings.

    Guaranteed to never make API calls. All processing is local.
    """

    # All local model names
    LOCAL_MODELS = {
        "all-MiniLM-L6-v2",
        "microsoft/codebert-base",
        "all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    }

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        if model not in self.LOCAL_MODELS:
            raise ValueError(
                f"Model {model} is not a local model. "
                f"Valid options: {self.LOCAL_MODELS}. "
                "Use EmbeddingService for API-based models."
            )
        super().__init__(model=model)


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing without real embeddings."""

    def __init__(self, dimension: int = 768):
        super().__init__(model="mock")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension
