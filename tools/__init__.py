from .code_tools import compile_python_files, create_file, edit_code_file, list_workspace_files, read_code_file
from .feature_tools import extract_features_from_readme
from .math_tools import add
from .memory_tools import forget_memory, recall_memory, remember
from .readme_tools import fetch_readme
from .report_tools import generate_report
from .registry import ToolRegistry, create_default_tool_registry
from .scoring_tools import score_projects, validate_project_result, validate_report
from .security_tools import generate_runtime_secret
from .web_tools import web_search


TOOLS = [
    add,
    generate_runtime_secret,
    remember,
    recall_memory,
    forget_memory,
    read_code_file,
    list_workspace_files,
    create_file,
    edit_code_file,
    compile_python_files,
    web_search,
    score_projects,
    validate_project_result,
    validate_report,
    generate_report,
    fetch_readme,
    extract_features_from_readme,
]

TOOL_MAP = {tool.__name__: tool for tool in TOOLS}

__all__ = [
    "TOOLS",
    "TOOL_MAP",
    "ToolRegistry",
    "create_default_tool_registry",
    "score_projects",
    "validate_project_result",
    "validate_report",
    "generate_report",
    "fetch_readme",
    "extract_features_from_readme",
]
