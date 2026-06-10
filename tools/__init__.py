from .code_tools import compile_python_files, create_file, edit_code_file, list_workspace_files, read_code_file
from .math_tools import add
from .security_tools import generate_runtime_secret
from .web_tools import web_search


TOOLS = [
    add,
    generate_runtime_secret,
    read_code_file,
    list_workspace_files,
    create_file,
    edit_code_file,
    compile_python_files,
    web_search,
]

TOOL_MAP = {tool.__name__: tool for tool in TOOLS}
