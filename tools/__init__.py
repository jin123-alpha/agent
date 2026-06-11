import inspect
import types
from typing import Union, get_args, get_origin

from .code_tools import compile_python_files, create_file, edit_code_file, list_workspace_files, read_code_file
from .math_tools import add
from .memory_tools import forget_memory, recall_memory, remember
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
]

TOOL_MAP = {tool.__name__: tool for tool in TOOLS}


def annotation_to_json_schema(annotation) -> dict:
    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {Union, types.UnionType}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return annotation_to_json_schema(non_none_args[0])

    if origin is list:
        item_schema = annotation_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    return {"type": "string"}


def tool_description(tool) -> str:
    doc = inspect.getdoc(tool) or "无描述。"
    return doc.splitlines()[0][:1000]


def build_tool_spec(tool) -> dict:
    signature = inspect.signature(tool)
    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        properties[name] = annotation_to_json_schema(parameter.annotation)
        if parameter.default is inspect.Signature.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": tool_description(tool),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SPECS = [build_tool_spec(tool) for tool in TOOLS]
