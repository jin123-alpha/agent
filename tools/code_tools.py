import json
import py_compile
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_LINES = 200
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".agents",
    ".codex",
    "__pycache__",
}


def resolve_project_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()

    if PROJECT_ROOT not in target.parents and target != PROJECT_ROOT:
        raise ValueError("拒绝访问项目目录之外的路径。")

    return target


def read_code_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """
    read_code_file(path, start_line=1, end_line=None)：阅读项目内代码文件，返回带行号的指定行范围；一次最多返回 200 行。
    """
    try:
        target = resolve_project_path(path)
    except ValueError as exc:
        return str(exc)

    if not target.exists():
        return f"文件不存在：{path}"

    if not target.is_file():
        return f"目标不是文件：{path}"

    if start_line < 1:
        start_line = 1

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total_lines = len(lines)

    if end_line is None:
        end_line = min(start_line + MAX_LINES - 1, total_lines)
    else:
        end_line = min(end_line, start_line + MAX_LINES - 1, total_lines)

    if start_line > total_lines:
        return f"文件 {path} 只有 {total_lines} 行，无法从第 {start_line} 行开始读取。"

    selected = lines[start_line - 1:end_line]
    numbered_lines = [
        f"{line_number:>4}: {line}"
        for line_number, line in enumerate(selected, start=start_line)
    ]

    header = f"文件：{path}，行号：{start_line}-{end_line}，总行数：{total_lines}"
    return "\n".join([header, *numbered_lines])


def edit_code_file(path: str, old_text: str, new_text: str, expected_replacements: int = 1) -> str:
    """
    edit_code_file(path, old_text, new_text, expected_replacements=1)：编辑项目内文本文件，将 old_text 精确替换为 new_text，默认要求只替换 1 处。
    """
    if not old_text:
        return "old_text 不能为空。"

    if expected_replacements < 1:
        expected_replacements = 1

    try:
        target = resolve_project_path(path)
    except ValueError as exc:
        return str(exc)

    if not target.exists():
        return f"文件不存在：{path}"

    if not target.is_file():
        return f"目标不是文件：{path}"

    content = target.read_text(encoding="utf-8", errors="replace")
    actual_replacements = content.count(old_text)

    if actual_replacements != expected_replacements:
        return (
            f"替换数量不匹配：期望 {expected_replacements} 处，实际找到 "
            f"{actual_replacements} 处。文件未修改。"
        )

    updated = content.replace(old_text, new_text, expected_replacements)
    target.write_text(updated, encoding="utf-8")

    return f"已编辑文件：{path}，替换 {expected_replacements} 处。"


def compile_python_files(paths: list[str] | None = None) -> str:
    """
    compile_python_files(paths=None)：编译检查 Python 文件并返回报错信息；paths 为空时检查当前工作区内所有 .py 文件。
    """
    if paths is None:
        targets = [
            path
            for path in PROJECT_ROOT.rglob("*.py")
            if not any(part in DEFAULT_IGNORED_DIRS for part in path.relative_to(PROJECT_ROOT).parts)
        ]
    else:
        targets = []
        for path in paths:
            try:
                target = resolve_project_path(path)
            except ValueError as exc:
                return str(exc)

            if not target.exists():
                return f"文件不存在：{path}"
            if not target.is_file():
                return f"目标不是文件：{path}"
            if target.suffix != ".py":
                return f"目标不是 Python 文件：{path}"

            targets.append(target)

    errors = []
    checked = []

    for target in sorted(targets):
        relative_path = str(target.relative_to(PROJECT_ROOT))
        checked.append(relative_path)

        try:
            py_compile.compile(str(target), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append({
                "file": relative_path,
                "error": exc.msg,
                "traceback": traceback.format_exc(limit=2),
            })

    return json.dumps(
        {
            "ok": not errors,
            "checked": checked,
            "errors": errors,
        },
        ensure_ascii=False,
        indent=2,
    )


def list_workspace_files(max_depth: int = 5) -> str:
    """
    list_workspace_files(max_depth=5)：获取当前工作区的文件结构树，默认最多展开 5 层，并忽略缓存、Git 和 Agent 配置目录。
    """
    if max_depth < 1:
        max_depth = 1

    lines = [f"{PROJECT_ROOT.name}/"]

    def walk(directory: Path, depth: int, prefix: str = ""):
        if depth >= max_depth:
            return

        children = sorted(
            [
                child
                for child in directory.iterdir()
                if child.name not in DEFAULT_IGNORED_DIRS
            ],
            key=lambda child: (child.is_file(), child.name.lower()),
        )

        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            branch = "`-- " if is_last else "|-- "
            next_prefix = "    " if is_last else "|   "
            suffix = "/" if child.is_dir() else ""

            lines.append(f"{prefix}{branch}{child.name}{suffix}")

            if child.is_dir():
                walk(child, depth + 1, prefix + next_prefix)

    walk(PROJECT_ROOT, depth=0)
    return "\n".join(lines)
