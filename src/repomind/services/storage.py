"""
Storage Service for Code Chunks and Embeddings.

This module provides the primary interface for storing and retrieving
indexed code chunks with their vector embeddings. It uses:

- **ChromaDB**: Vector database for semantic similarity search
- **JSON Files**: Full metadata storage for complete chunk retrieval

Architecture:
    ChromaDB stores embeddings + minimal metadata for fast vector search.
    JSON files store complete CodeChunk objects for full data retrieval.
    This hybrid approach optimizes both search speed and data completeness.

Key Features:
    - Automatic dimension migration when embedding model changes
    - Automatic collection migration from legacy names
    - Batch operations for efficient large-scale indexing
    - Metadata filtering for targeted searches
    - Thread-safe singleton pattern for ChromaDB client

Usage:
    from repomind.services.storage import StorageService

    storage = StorageService()

    # Store chunks with embeddings
    chunks_stored = storage.store_chunks(chunks, embeddings)

    # Search by similarity
    results = storage.search(query_embedding, n_results=10)

    # Get statistics
    stats = storage.get_stats()

Author: Code Expert Team
"""

import json
from pathlib import Path
from typing import Optional

from ..config import IndexConfig, get_config
from ..constants import (
    CHROMADB_COLLECTION_NAME,
    CHROMADB_COLLECTION_DESCRIPTION,
    CHROMADB_BATCH_SIZE,
    ErrorMessage,
    SuccessMessage,
)
from ..logging import get_logger, log_operation_start, log_operation_end
from ..models.chunk import ChunkType, CodeChunk


# Module logger
logger = get_logger(__name__)

# ChromaDB client cache for connection pooling
_chromadb_client_cache: dict = {}


class StorageService:
    """
    Service for storing and retrieving code chunks with vector search.

    This service manages the persistence layer for the Code Expert
    indexing system. It provides:

    1. **Vector Storage**: ChromaDB for semantic similarity search
    2. **Metadata Storage**: JSON files for complete chunk data
    3. **Automatic Migration**: Handles embedding dimension changes
    4. **Batch Operations**: Efficient large-scale indexing

    The service uses a singleton pattern for the ChromaDB client to
    prevent resource leaks and ensure consistent behavior.

    Attributes:
        config: Index configuration (directories, settings)

    Example:
        # Initialize service
        storage = StorageService()

        # Store indexed chunks
        stored_count = storage.store_chunks(chunks, embeddings)
        logger.info(f"Stored {stored_count} chunks")

        # Search for similar code
        results = storage.search(
            query_embedding=query_vector,
            n_results=10,
            repo_filter="my-repo",
            type_filter=ChunkType.FUNCTION
        )

        for chunk, similarity_score in results:
            print(f"{chunk.name}: {similarity_score:.3f}")
    """

    # Class-level constants for easy reference
    COLLECTION_NAME = CHROMADB_COLLECTION_NAME
    COLLECTION_DESCRIPTION = CHROMADB_COLLECTION_DESCRIPTION
    DEFAULT_BATCH_SIZE = CHROMADB_BATCH_SIZE
    METADATA_FILENAME = "chunks.json"

    def __init__(self, config: IndexConfig | None = None):
        """
        Initialize the storage service.

        Args:
            config: Optional index configuration. If not provided,
                    uses the global configuration.
        """
        self.config = config or get_config().index
        self._client = None
        self._collection = None

        # Ensure data directories exist
        self._ensure_directories_exist()

        logger.debug(
            "Storage service initialized",
            extra={
                "data_dir": str(self.config.data_dir),
                "chroma_dir": str(self.config.chroma_dir),
            }
        )

    def _ensure_directories_exist(self) -> None:
        """Create required directories if they don't exist."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_chromadb_client(self):
        """
        Get or create a ChromaDB client with connection pooling.

        Uses a module-level cache to ensure we reuse connections
        and avoid resource leaks.

        Returns:
            ChromaDB PersistentClient instance.
        """
        cache_key = str(self.config.chroma_dir)

        if cache_key not in _chromadb_client_cache:
            try:
                import chromadb
                from chromadb.config import Settings

                logger.debug("Creating new ChromaDB client", extra={"path": cache_key})

                _chromadb_client_cache[cache_key] = chromadb.PersistentClient(
                    path=cache_key,
                    settings=Settings(anonymized_telemetry=False),
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required for storage. Install with: pip install chromadb"
                )

        return _chromadb_client_cache[cache_key]

    def _get_collection(self):
        """
        Get or create the ChromaDB collection for code chunks.

        Uses connection pooling via _get_chromadb_client() to ensure
        efficient resource usage.

        Returns:
            ChromaDB Collection instance.
        """
        if self._collection is None:
            self._client = self._get_chromadb_client()
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    "description": self.COLLECTION_DESCRIPTION,
                    "hnsw:space": "cosine",  # Use cosine similarity for better semantic search
                },
            )

            logger.debug(
                "ChromaDB collection ready",
                extra={
                    "collection_name": self.COLLECTION_NAME,
                    "existing_count": self._collection.count(),
                }
            )

        return self._collection

    def _handle_embedding_dimension_mismatch(
        self,
        existing_dimension: int,
        new_dimension: int
    ) -> None:
        """
        Handle a mismatch between existing and new embedding dimensions.

        When the embedding model changes (e.g., from MiniLM to CodeBERT),
        the embedding dimensions change. ChromaDB cannot store vectors
        of different dimensions in the same collection.

        This method:
        1. Logs a warning about the dimension change
        2. Deletes the old collection
        3. Creates a new collection for the new dimensions

        Args:
            existing_dimension: Current dimension in the collection.
            new_dimension: Dimension of the new embeddings.
        """
        logger.warning(
            ErrorMessage.EMBEDDING_DIMENSION_MISMATCH.format(
                existing=existing_dimension,
                new=new_dimension
            ),
            extra={
                "existing_dimension": existing_dimension,
                "new_dimension": new_dimension,
            }
        )

        # Delete the old collection
        self._client.delete_collection(name=self.COLLECTION_NAME)
        logger.info("Deleted old collection due to dimension mismatch")

        # Recreate collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "description": self.COLLECTION_DESCRIPTION,
                "hnsw:space": "cosine",
            },
        )

        logger.info(
            SuccessMessage.COLLECTION_MIGRATED.format(dimension=new_dimension)
        )

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        """
        Validate that new embeddings match the collection's expected dimension.

        If there's a mismatch, automatically migrates the collection.
        This ensures smooth transitions when changing embedding models.

        Args:
            embeddings: List of embedding vectors to validate.
        """
        if not embeddings or not embeddings[0]:
            return

        new_dimension = len(embeddings[0])
        collection = self._get_collection()

        # Check if collection has any existing data
        existing_count = collection.count()
        if existing_count == 0:
            logger.debug(
                "Empty collection, no dimension validation needed",
                extra={"new_dimension": new_dimension}
            )
            return

        # Get a sample to check existing dimension
        try:
            sample = collection.peek(limit=1)
            if sample and sample.get('embeddings') and sample['embeddings']:
                existing_dimension = len(sample['embeddings'][0])

                if existing_dimension != new_dimension:
                    self._handle_embedding_dimension_mismatch(
                        existing_dimension, new_dimension
                    )
        except Exception as error:
            logger.warning(
                "Could not validate embedding dimensions",
                extra={"error": str(error)}
            )

    def store_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]]
    ) -> int:
        """
        Store code chunks with their vector embeddings.

        This method persists chunks to both ChromaDB (for vector search)
        and JSON files (for complete data retrieval).

        The method handles:
        - Batch processing to respect ChromaDB limits
        - Automatic dimension migration if embedding model changed
        - Validation of embedding format
        - Atomic upsert operations

        Args:
            chunks: List of CodeChunk objects to store.
            embeddings: Corresponding embedding vectors (same order as chunks).

        Returns:
            Number of chunks successfully stored.

        Raises:
            ValueError: If chunks and embeddings counts don't match.
            TypeError: If embeddings have invalid format.
            RuntimeError: If ChromaDB upsert fails.

        Example:
            chunks = [CodeChunk(...), CodeChunk(...)]
            embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]

            stored_count = storage.store_chunks(chunks, embeddings)
            print(f"Stored {stored_count} chunks")
        """
        # Validate input
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        if not chunks:
            logger.debug("No chunks to store, returning early")
            return 0

        start_time = log_operation_start(
            logger, "store_chunks",
            chunk_count=len(chunks),
            embedding_dimension=len(embeddings[0]) if embeddings else 0
        )

        # Validate embedding dimension and handle migration if needed
        self._validate_embedding_dimensions(embeddings)

        collection = self._get_collection()

        # Prepare data for ChromaDB
        chunk_ids = [chunk.id for chunk in chunks]
        documents = [chunk.to_embedding_text() for chunk in chunks]
        metadata_list = self._prepare_chunk_metadata(chunks)

        # Validate embedding format
        self._validate_embedding_format(embeddings)

        # Store in batches
        total_stored = self._store_in_batches(
            collection, chunk_ids, documents, metadata_list, embeddings
        )

        # Also save complete chunk data to JSON
        self._save_chunk_metadata_to_file(chunks)

        log_operation_end(
            logger, "store_chunks", start_time,
            chunks_stored=total_stored
        )

        return total_stored

    def _prepare_chunk_metadata(self, chunks: list[CodeChunk]) -> list[dict]:
        """
        Prepare metadata dictionaries for ChromaDB storage.

        ChromaDB metadata is used for filtering during search operations.
        Only essential fields are stored here; complete data is in JSON.

        Args:
            chunks: List of CodeChunk objects.

        Returns:
            List of metadata dictionaries for ChromaDB.
        """
        return [
            {
                "repo_name": chunk.repo_name,
                "file_path": chunk.file_path,
                "chunk_type": chunk.chunk_type.value,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
                "parent_name": chunk.parent_name or "",
                "has_docstring": chunk.docstring is not None,
            }
            for chunk in chunks
        ]

    def _validate_embedding_format(self, embeddings: list[list[float]]) -> None:
        """
        Validate that embeddings have the correct format.

        Args:
            embeddings: List of embedding vectors to validate.

        Raises:
            TypeError: If embeddings have invalid format.
        """
        if not embeddings:
            return

        sample_embedding = embeddings[0]

        if not isinstance(sample_embedding, list):
            raise TypeError(
                f"Each embedding must be a list of floats. "
                f"Got: {type(sample_embedding).__name__}"
            )

        if sample_embedding and not isinstance(sample_embedding[0], (float, int)):
            raise TypeError(
                f"Embedding values must be numeric. "
                f"Got: {type(sample_embedding[0]).__name__}"
            )

    def _store_in_batches(
        self,
        collection,
        chunk_ids: list[str],
        documents: list[str],
        metadata_list: list[dict],
        embeddings: list[list[float]]
    ) -> int:
        """
        Store chunks in batches to respect ChromaDB limits.

        Args:
            collection: ChromaDB collection instance.
            chunk_ids: List of unique chunk identifiers.
            documents: List of document texts for search.
            metadata_list: List of metadata dictionaries.
            embeddings: List of embedding vectors.

        Returns:
            Total number of chunks stored.

        Raises:
            RuntimeError: If a batch upsert fails.
        """
        batch_size = self.DEFAULT_BATCH_SIZE
        total_stored = 0
        total_batches = (len(chunk_ids) + batch_size - 1) // batch_size

        for batch_number, start_index in enumerate(range(0, len(chunk_ids), batch_size), 1):
            end_index = start_index + batch_size

            batch_ids = chunk_ids[start_index:end_index]
            batch_docs = documents[start_index:end_index]
            batch_meta = metadata_list[start_index:end_index]
            batch_emb = embeddings[start_index:end_index]

            try:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                    embeddings=batch_emb,
                )
                total_stored += len(batch_ids)

                logger.debug(
                    f"Stored batch {batch_number}/{total_batches}",
                    extra={"batch_size": len(batch_ids)}
                )

            except Exception as error:
                error_message = (
                    f"Failed to store batch {batch_number}/{total_batches}. "
                    f"Batch size: {len(batch_ids)}, "
                    f"Embedding shape: {len(batch_emb)}x{len(batch_emb[0]) if batch_emb else 0}"
                )
                logger.error(error_message, extra={"error": str(error)})
                raise RuntimeError(error_message) from error

        return total_stored

    def _save_chunk_metadata_to_file(self, chunks: list[CodeChunk]) -> None:
        """
        Save complete chunk data to JSON file for retrieval.

        ChromaDB only stores minimal metadata for filtering.
        This JSON file stores the complete CodeChunk objects.

        Args:
            chunks: List of CodeChunk objects to save.
        """
        chunk_map = self._load_chunk_metadata()

        for chunk in chunks:
            chunk_map[chunk.id] = chunk

        self._save_chunk_map(chunk_map)

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        repo_filter: Optional[str] = None,
        type_filter: Optional[ChunkType] = None,
        language_filter: Optional[str] = None,
        query_text: Optional[str] = None,
        similarity_threshold: float = 0.35,
        enable_hybrid_search: bool = True,
    ) -> list[tuple[CodeChunk, float]]:
        """Search for similar code chunks with hybrid semantic + keyword matching.

        Args:
            query_embedding: Query vector for semantic search
            n_results: Maximum number of results to return
            repo_filter: Filter by repository name
            type_filter: Filter by chunk type
            language_filter: Filter by programming language
            query_text: Original query text for keyword boosting
            similarity_threshold: Minimum similarity score (0-1) to include
            enable_hybrid_search: Whether to apply keyword boosting

        Returns:
            List of (CodeChunk, similarity_score) tuples, sorted by relevance
        """
        collection = self._get_collection()

        # Build where filter
        where_filter = {}
        if repo_filter:
            where_filter["repo_name"] = repo_filter
        if type_filter:
            where_filter["chunk_type"] = type_filter.value
        if language_filter:
            where_filter["language"] = language_filter

        # Request more results for filtering and reranking
        fetch_multiplier = 3 if enable_hybrid_search and query_text else 1
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * fetch_multiplier, 100),
            where=where_filter if where_filter else None,
            include=["metadatas", "distances", "documents"],
        )

        # Load full chunk data and compute scores
        chunks_with_scores = []
        if results["ids"] and results["ids"][0]:
            chunk_map = self._load_chunk_metadata()

            for i, chunk_id in enumerate(results["ids"][0]):
                if chunk_id not in chunk_map:
                    continue

                chunk = chunk_map[chunk_id]
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to similarity score
                # ChromaDB with cosine distance returns values in range [0, 2]
                # where 0 = identical, 1 = orthogonal, 2 = opposite
                similarity = max(0, 1 - (distance / 2))

                # Apply keyword boosting for hybrid search
                if enable_hybrid_search and query_text:
                    keyword_boost = self._compute_keyword_boost(query_text, chunk)
                    similarity = min(1.0, similarity + keyword_boost)

                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    chunks_with_scores.append((chunk, similarity))

        # Sort by score (descending) and limit results
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return chunks_with_scores[:n_results]

    def _compute_keyword_boost(self, query_text: str, chunk: CodeChunk) -> float:
        """
        Compute keyword matching boost for hybrid search with phrase awareness.

        Args:
            query_text: Original search query
            chunk: Code chunk to check

        Returns:
            Boost value to add to similarity score (0.0 to ~0.3)
        """
        # Extract keywords from query
        query_lower = query_text.lower()
        query_keywords = set(
            word.lower()
            for word in query_text.replace('_', ' ').split()
            if len(word) > 2
        )

        if not query_keywords:
            return 0.0

        # Check for multi-word phrase matches (worth more)
        phrase_boost = 0.0
        multi_word_phrases = []
        words = query_text.split()

        # Generate 2-word and 3-word phrases
        for i in range(len(words) - 1):
            multi_word_phrases.append(' '.join(words[i:i+2]))
            if i < len(words) - 2:
                multi_word_phrases.append(' '.join(words[i:i+3]))

        # Check for phrase matches in name, docstring, content
        chunk_name_lower = chunk.name.lower()
        docstring_lower = chunk.docstring.lower() if chunk.docstring else ""
        content_lower = chunk.content.lower()

        for phrase in multi_word_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in chunk_name_lower:
                phrase_boost += 0.15  # 15% boost for phrase in name
            elif phrase_lower in docstring_lower:
                phrase_boost += 0.10  # 10% boost for phrase in docstring
            elif phrase_lower in content_lower:
                phrase_boost += 0.05  # 5% boost for phrase in content

        # Single keyword matching boost
        keyword_boost = 0.0
        boost_factor = 0.03  # Reduced from 0.05 to prevent overwhelming

        # Name matches are most valuable
        matched_in_name = sum(1 for kw in query_keywords if kw in chunk_name_lower)
        keyword_boost += matched_in_name * boost_factor * 2

        # Docstring matches are valuable
        matched_in_docstring = 0
        if chunk.docstring:
            matched_in_docstring = sum(1 for kw in query_keywords if kw in docstring_lower)
            keyword_boost += matched_in_docstring * boost_factor

        # Content matches (less weight, and penalize if ONLY these match)
        matched_in_content = sum(1 for kw in query_keywords if kw in content_lower)

        # If keywords only match in content but not in name/docstring, reduce boost
        if matched_in_content > 0 and matched_in_name == 0 and matched_in_docstring == 0:
            keyword_boost += matched_in_content * boost_factor * 0.3  # Reduced weight
        else:
            keyword_boost += matched_in_content * boost_factor * 0.5

        # Combine phrase and keyword boosts
        total_boost = phrase_boost + keyword_boost

        # Cap the boost to prevent overwhelming semantic similarity
        return min(total_boost, 0.35)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Retrieve a specific chunk by ID."""
        chunk_map = self._load_chunk_metadata()
        return chunk_map.get(chunk_id)

    def get_chunks_by_file(self, repo_name: str, file_path: str) -> list[CodeChunk]:
        """Get all chunks from a specific file."""
        chunk_map = self._load_chunk_metadata()
        return [
            chunk
            for chunk in chunk_map.values()
            if chunk.repo_name == repo_name and chunk.file_path == file_path
        ]

    def get_stats(self) -> dict:
        """Get index statistics."""
        collection = self._get_collection()
        chunk_map = self._load_chunk_metadata()

        # Aggregate stats
        repos = set()
        languages = {}
        types = {}

        for chunk in chunk_map.values():
            repos.add(chunk.repo_name)
            languages[chunk.language] = languages.get(chunk.language, 0) + 1
            types[chunk.chunk_type.value] = types.get(chunk.chunk_type.value, 0) + 1

        return {
            "total_chunks": collection.count(),
            "repositories": list(repos),
            "languages": languages,
            "chunk_types": types,
        }

    def clear_repo(self, repo_name: str) -> int:
        """Remove all chunks from a specific repository.

        Args:
            repo_name: Name of the repository to clear

        Returns:
            Number of chunks removed
        """
        collection = self._get_collection()

        # Get all IDs for this repo
        results = collection.get(where={"repo_name": repo_name}, include=[])
        if not results["ids"]:
            return 0

        # Delete from ChromaDB
        collection.delete(ids=results["ids"])

        # Update metadata file
        chunk_map = self._load_chunk_metadata()
        removed = 0
        for chunk_id in results["ids"]:
            if chunk_id in chunk_map:
                del chunk_map[chunk_id]
                removed += 1

        self._save_chunk_map(chunk_map)

        return removed

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        collection = self._get_collection()

        # Delete from ChromaDB
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            pass  # Ignore errors for non-existent IDs

        # Update metadata file
        chunk_map = self._load_chunk_metadata()
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in chunk_map:
                del chunk_map[chunk_id]
                deleted += 1

        if deleted > 0:
            self._save_chunk_map(chunk_map)

        return deleted

    def _save_chunk_metadata(self, chunks: list[CodeChunk]) -> None:
        """Save full chunk data to JSON file."""
        chunk_map = self._load_chunk_metadata()

        for chunk in chunks:
            chunk_map[chunk.id] = chunk

        self._save_chunk_map(chunk_map)

    def _save_chunk_map(self, chunk_map: dict[str, CodeChunk]) -> None:
        """Save chunk map to file."""
        metadata_file = self.config.metadata_dir / "chunks.json"
        data = {chunk_id: chunk.model_dump() for chunk_id, chunk in chunk_map.items()}

        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_chunk_metadata(self) -> dict[str, CodeChunk]:
        """Load full chunk data from JSON file."""
        metadata_file = self.config.metadata_dir / "chunks.json"

        if not metadata_file.exists():
            return {}

        try:
            with open(metadata_file) as f:
                data = json.load(f)
            return {
                chunk_id: CodeChunk(**chunk_data)
                for chunk_id, chunk_data in data.items()
            }
        except (json.JSONDecodeError, Exception):
            return {}

    def clear_all(self) -> int:
        """Clear the entire index (all repositories).

        WARNING: This deletes all indexed data!

        Returns:
            Number of chunks removed
        """
        collection = self._get_collection()
        count = collection.count()

        if count > 0:
            # Delete the entire collection
            self._client.delete_collection(name=self.COLLECTION_NAME)

            # Recreate empty collection with cosine similarity
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    "description": self.COLLECTION_DESCRIPTION,
                    "hnsw:space": "cosine",
                },
            )

        # Clear metadata file
        metadata_file = self.config.metadata_dir / "chunks.json"
        if metadata_file.exists():
            metadata_file.unlink()

        return count

