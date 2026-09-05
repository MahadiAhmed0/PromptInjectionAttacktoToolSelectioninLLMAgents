"""Tool library model for the tool-selection robustness harness.

:class:`ToolLibrary` is an immutable-by-convention container of
:class:`ToolDocument` objects. All mutating-style operations return a new
library instead of modifying the receiver, which lets researchers snapshot
libraries before and after modifications when running controlled
tool-selection robustness experiments.

Purpose note: this module supports defensive research and benchmarking of
tool retrieval/selection robustness in LLM agents. It provides the data
plumbing for experiments (e.g., measuring how selection accuracy degrades
under modified tool registries); it does not generate exploits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tool_selection_harness.core.tool_document import ToolDocument


@dataclass
class ToolLibrary:
    """An ordered collection of tool documents available to an LLM agent.

    Methods that would conventionally mutate the library (``add``,
    ``remove``, ``inject``) instead return a new :class:`ToolLibrary`
    instance, leaving the original untouched. This copy-on-write behavior is
    deliberate: evaluation experiments frequently need to compare agent
    behavior against a pristine baseline library and one or more modified
    libraries simultaneously.
    """

    documents: list[ToolDocument] = field(default_factory=list)

    # -- construction helpers -------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "ToolLibrary":
        """Construct a library from a JSON file on disk.

        Args:
            path: Path to a JSON file containing a list of tool document
                objects, each with ``tool_name`` and ``tool_description``
                keys (and optionally ``metadata``).
        """
        data = _read_json(Path(path))
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list of tool documents in {path}, "
                f"got {type(data).__name__}"
            )
        documents = [_document_from_mapping(item) for item in data]
        return cls(documents=documents)

    # -- queries --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __contains__(self, name: str) -> bool:
        return any(doc.tool_name == name for doc in self.documents)

    def get(self, name: str) -> ToolDocument | None:
        """Return the document with ``name``, or ``None`` if absent."""
        for doc in self.documents:
            if doc.tool_name == name:
                return doc
        return None

    def names(self) -> list[str]:
        """Return the tool names in library order."""
        return [doc.tool_name for doc in self.documents]

    # -- copy-on-write operations ----------------------------------------

    def add(self, doc: ToolDocument) -> "ToolLibrary":
        """Return a new library with ``doc`` appended.

        The receiver is left unchanged. If a document with the same
        ``tool_name`` already exists, the new document is appended after it
        (callers who need strict uniqueness should check with ``get``).
        """
        return ToolLibrary(documents=self.documents + [doc])

    def remove(self, name: str) -> "ToolLibrary":
        """Return a new library without the document named ``name``.

        The receiver is left unchanged. Raises :class:`KeyError` if ``name``
        is not present.
        """
        if name not in self:
            raise KeyError(f"No tool named {name!r} in library")
        return ToolLibrary(
            documents=[doc for doc in self.documents if doc.tool_name != name]
        )

    def inject(self, doc: ToolDocument) -> "ToolLibrary":
        """Return a new library with ``doc`` injected, leaving this one intact.

        This is the primary entry point for robustness experiments: a
        researcher starts from a baseline library and produces modified
        variants by injecting additional tool documents, then observes
        whether the agent's tool-selection behavior changes. The operation
        never mutates the original library, so baseline and injected variants
        can be evaluated side by side.
        """
        return ToolLibrary(documents=self.documents + [doc])

    # -- persistence ------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize documents to plain dicts (for JSON export)."""
        result: list[dict[str, Any]] = []
        for doc in self.documents:
            item: dict[str, Any] = {
                "tool_name": doc.tool_name,
                "tool_description": doc.tool_description,
            }
            if doc.metadata:
                item["metadata"] = dict(doc.metadata)
            result.append(item)
        return result

    def save_json(self, path: str | Path) -> None:
        """Write the library to ``path`` as a JSON list of documents."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dicts(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    load_json = from_json


def _document_from_mapping(item: Any) -> ToolDocument:
    """Convert a decoded JSON object into a ToolDocument with validation."""
    if not isinstance(item, dict):
        raise ValueError(
            f"Each entry must be a JSON object, got {type(item).__name__}"
        )
    if "tool_name" not in item or "tool_description" not in item:
        raise ValueError(
            "Each tool document requires 'tool_name' and 'tool_description'"
        )
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"'metadata' for {item['tool_name']!r} must be an object")
    return ToolDocument(
        tool_name=item["tool_name"],
        tool_description=item["tool_description"],
        metadata={str(k): str(v) for k, v in metadata.items()},
    )


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file with helpful error messages."""
    if not path.exists():
        raise FileNotFoundError(f"Tool library file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
