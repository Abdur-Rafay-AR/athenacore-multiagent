"""Tools agents can call, via a text protocol that works on any model."""

from __future__ import annotations

from crucible.tools.base import (
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    parse_tool_calls,
    strip_tool_calls,
)
from crucible.tools.builtin import (
    CalculatorTool,
    ClockTool,
    MemorySearchTool,
    MemoryWriteTool,
    WebSearchTool,
    default_registry,
)

__all__ = [
    "CalculatorTool",
    "ClockTool",
    "MemorySearchTool",
    "MemoryWriteTool",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "default_registry",
    "parse_tool_calls",
    "strip_tool_calls",
]
