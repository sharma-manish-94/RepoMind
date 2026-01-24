"""Manifest service for tracking indexed file states."""

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..config import get_config


class FileStatus(str, Enum):
    """Status of a file compared to the manifest."""
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    NEW = "new"
    DELETED = "deleted"


@dataclass
class FileEntry:
    """Entry for a tracked file in the manifest."""
    mtime: float
    size: int
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mtime": self.mtime,
            "size": self.size,
            "chunk_ids": self.chunk_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileEntry":
        return cls(
            mtime=data["mtime"],
            size=data["size"],
            chunk_ids=data.get("chunk_ids", []),
        )


@dataclass
class FileChange:
    """Represents a change to a file."""
    path: Path
    relative_path: str
    status: FileStatus
    old_chunk_ids: list[str] = field(default_factory=list)


class ManifestService:
    """Service for tracking indexed file states to enable incremental indexing."""

    def __init__(self):
        config = get_config()
        self.manifest_dir = config.index.metadata_dir
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def _get_manifest_path(self, repo_name: str) -> Path:
        """Get path to manifest file for a repository."""
        return self.manifest_dir / f"manifest_{repo_name}.json"

    def load_manifest(self, repo_name: str) -> dict[str, FileEntry]:
        """Load manifest for a repository."""
        manifest_path = self._get_manifest_path(repo_name)
        if not manifest_path.exists():
            return {}

        try:
            with open(manifest_path) as f:
                data = json.load(f)
            return {
                path: FileEntry.from_dict(entry)
                for path, entry in data.items()
            }
        except (json.JSONDecodeError, KeyError):
            return {}

    def save_manifest(self, repo_name: str, manifest: dict[str, FileEntry]) -> None:
        """Save manifest for a repository."""
        manifest_path = self._get_manifest_path(repo_name)
        data = {path: entry.to_dict() for path, entry in manifest.items()}
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_file_changes(
        self,
        repo_path: Path,
        repo_name: str,
        current_files: list[Path],
        use_git: bool = True,
    ) -> list[FileChange]:
        """Determine which files have changed since last index.

        Args:
            repo_path: Path to repository root
            repo_name: Repository name
            current_files: List of current source files
            use_git: Whether to use git for change detection

        Returns:
            List of FileChange objects describing what changed
        """
        manifest = self.load_manifest(repo_name)
        changes = []

        # Try git-based detection first (faster for large repos)
        if use_git:
            git_changes = self._get_git_changes(repo_path, repo_name, manifest)
            if git_changes is not None:
                return git_changes

        # Fall back to mtime-based detection
        current_paths = {str(f.relative_to(repo_path)): f for f in current_files}

        # Check for new and modified files
        for rel_path, abs_path in current_paths.items():
            try:
                stat = abs_path.stat()
                current_mtime = stat.st_mtime
                current_size = stat.st_size
            except OSError:
                continue

            if rel_path not in manifest:
                # New file
                changes.append(FileChange(
                    path=abs_path,
                    relative_path=rel_path,
                    status=FileStatus.NEW,
                ))
            else:
                entry = manifest[rel_path]
                # Check if modified (mtime or size changed)
                if current_mtime != entry.mtime or current_size != entry.size:
                    changes.append(FileChange(
                        path=abs_path,
                        relative_path=rel_path,
                        status=FileStatus.MODIFIED,
                        old_chunk_ids=entry.chunk_ids,
                    ))
                # else: unchanged, skip

        # Check for deleted files
        for rel_path, entry in manifest.items():
            if rel_path not in current_paths:
                changes.append(FileChange(
                    path=repo_path / rel_path,
                    relative_path=rel_path,
                    status=FileStatus.DELETED,
                    old_chunk_ids=entry.chunk_ids,
                ))

        return changes

    def _get_git_changes(
        self,
        repo_path: Path,
        repo_name: str,
        manifest: dict[str, FileEntry],
    ) -> Optional[list[FileChange]]:
        """Use git to detect changes since last index.

        Returns None if git detection fails (not a git repo, etc.)
        """
        # Check if this is a git repository
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return None

        # Get the last indexed commit (stored in manifest metadata)
        last_commit = self._get_last_indexed_commit(repo_name)

        if not last_commit:
            # No previous index - use mtime-based detection
            return None

        try:
            # Get committed changes since last indexed commit
            result_committed = subprocess.run(
                ["git", "diff", "--name-status", last_commit, "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Get uncommitted changes (working tree vs HEAD)
            result_uncommitted = subprocess.run(
                ["git", "diff", "--name-status", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result_committed.returncode != 0:
                return None

            # Combine both outputs
            combined_output = result_committed.stdout.strip()
            if result_uncommitted.returncode == 0 and result_uncommitted.stdout.strip():
                if combined_output:
                    combined_output += "\n" + result_uncommitted.stdout.strip()
                else:
                    combined_output = result_uncommitted.stdout.strip()

            changes = []
            seen_files = set()  # Avoid duplicates if file appears in both diffs
            source_extensions = {".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".mjs"}

            for line in combined_output.split("\n"):
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                status_code = parts[0][0]  # First char: A, M, D, R, etc.
                file_path = parts[-1]  # Last part is the file path

                # Only process source files
                if not any(file_path.endswith(ext) for ext in source_extensions):
                    continue

                # Skip duplicates (file may appear in both committed and uncommitted)
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)

                abs_path = repo_path / file_path
                old_chunks = manifest.get(file_path, FileEntry(0, 0, [])).chunk_ids

                if status_code == "A":
                    changes.append(FileChange(
                        path=abs_path,
                        relative_path=file_path,
                        status=FileStatus.NEW,
                    ))
                elif status_code == "M":
                    changes.append(FileChange(
                        path=abs_path,
                        relative_path=file_path,
                        status=FileStatus.MODIFIED,
                        old_chunk_ids=old_chunks,
                    ))
                elif status_code == "D":
                    changes.append(FileChange(
                        path=abs_path,
                        relative_path=file_path,
                        status=FileStatus.DELETED,
                        old_chunk_ids=old_chunks,
                    ))
                elif status_code == "R":
                    # Renamed: treat as delete old + add new
                    old_path = parts[1] if len(parts) > 2 else file_path
                    old_chunks = manifest.get(old_path, FileEntry(0, 0, [])).chunk_ids
                    changes.append(FileChange(
                        path=repo_path / old_path,
                        relative_path=old_path,
                        status=FileStatus.DELETED,
                        old_chunk_ids=old_chunks,
                    ))
                    changes.append(FileChange(
                        path=abs_path,
                        relative_path=file_path,
                        status=FileStatus.NEW,
                    ))

            return changes

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return None

    def _get_last_indexed_commit(self, repo_name: str) -> Optional[str]:
        """Get the commit hash from the last index."""
        meta_path = self.manifest_dir / f"meta_{repo_name}.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                data = json.load(f)
            return data.get("last_commit")
        except (json.JSONDecodeError, KeyError):
            return None

    def save_last_indexed_commit(self, repo_name: str, repo_path: Path) -> None:
        """Save the current HEAD commit as last indexed."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                meta_path = self.manifest_dir / f"meta_{repo_name}.json"
                with open(meta_path, "w") as f:
                    json.dump({"last_commit": commit}, f)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    def update_manifest_entry(
        self,
        repo_name: str,
        relative_path: str,
        file_path: Path,
        chunk_ids: list[str],
    ) -> None:
        """Update a single file entry in the manifest."""
        manifest = self.load_manifest(repo_name)

        try:
            stat = file_path.stat()
            manifest[relative_path] = FileEntry(
                mtime=stat.st_mtime,
                size=stat.st_size,
                chunk_ids=chunk_ids,
            )
            self.save_manifest(repo_name, manifest)
        except OSError:
            pass

    def remove_manifest_entry(self, repo_name: str, relative_path: str) -> None:
        """Remove a file entry from the manifest."""
        manifest = self.load_manifest(repo_name)
        if relative_path in manifest:
            del manifest[relative_path]
            self.save_manifest(repo_name, manifest)
