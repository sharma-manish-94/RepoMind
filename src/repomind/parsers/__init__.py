"""Code parsers for different languages."""

from .base import BaseParser
from .java_parser import JavaParser
from .python_parser import PythonParser
from .typescript_parser import TypeScriptParser

__all__ = ["BaseParser", "PythonParser", "JavaParser", "TypeScriptParser", "get_parser"]


_PARSERS: dict[str, type[BaseParser]] = {
    "python": PythonParser,
    "java": JavaParser,
    "typescript": TypeScriptParser,
    "javascript": TypeScriptParser,  # TS parser handles JS too
}


def get_parser(language: str) -> BaseParser | None:
    """Get a parser for the given language."""
    parser_class = _PARSERS.get(language.lower())
    if parser_class:
        return parser_class()
    return None


def get_language_for_file(file_path: str) -> str | None:
    """Determine language from file extension."""
    ext_map = {
        ".py": "python",
        ".java": "java",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
    }
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    return None
