"""
Git-Powered MCP Server with Simplified Single-Pass Truncation Safety
"""

import json
import logging
import os
import asyncio
import shlex
from pathlib import Path
from typing import Dict, List, Any
from fastmcp import FastMCP
import git
import re
import subprocess
from dataclasses import dataclass
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# SYSTEM PARAMETERS
# ----------------------------------------------------------

ALLOWED_ROOT_DIR = Path(r"path/to/project").resolve()

ALLOWED_EXTENSIONS = {
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".inl",
    ".ixx",
    ".cxx",
    ".hxx",
    ".cs",
    ".py",
    ".txt",
    ".md",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".v",
    ".sv",
    ".vh",  # Verilog / SystemVerilog HDL Formats
}

IGNORED_DIRS = {
    ".git",
    ".vs",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "build",
    "dist",
    "bin",
    "obj",
    "node_modules",
    "__pycache__",
}

MAX_FILE_SIZE_MB = 10
MAX_CHARS_PER_RESPONSE = (
    60000  # Generous cap (~15k tokens) - easily covers a few thousand lines of code
)

SANDBOX_BRANCH = "mcp-sandbox"

CPP_EXTENSIONS = {".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx", ".inl", ".ixx"}


def _is_cpp_file(path: str) -> bool:
    return Path(path).suffix.lower() in CPP_EXTENSIONS


# Used to build the dependency graph in the build dependencies
@dataclass
class DependencyNode:
    path: str
    children: List["DependencyNode"]
    cyclic: bool = False
    truncated: bool = False


@dataclass
class IncludeInfo:
    include: str
    resolved: str | None
    include_type: str  # SYSTEM | PROJECT | UNRESOLVED


@dataclass
class GitWorkspace:
    repo: git.Repo
    sandbox_branch: str


server_instructions = """
Search and inspect source code files inside the restricted workspace.
Files are returned in full single-pass outputs. If a file is exceptionally large, 
the server will automatically truncate the tail end and append an explicit notice.
"""

# ----------------------------------------------------------
# FILE SYSTEM SECURITY & ACCESS HELPERS
# ----------------------------------------------------------


def secure_path(relative_path: str) -> Path:
    clean_rel = relative_path.lstrip("/\\")
    target = (ALLOWED_ROOT_DIR / clean_rel).resolve()
    if not str(target).startswith(str(ALLOWED_ROOT_DIR)):
        raise PermissionError("Access Denied: Path escapes restricted root namespace.")
    return target


def is_allowed_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def enforce_truncation_safety(text_data: str) -> str:
    """
    If the code output fits within the generous limit, pass it through completely.
    Otherwise, truncate it cleanly to keep the payload stable over ngrok.
    """
    if len(text_data) > MAX_CHARS_PER_RESPONSE:
        truncated = text_data[:MAX_CHARS_PER_RESPONSE]
        warning = f"\n\n⚠️ [SERVER NOTICE: Output exceeded {MAX_CHARS_PER_RESPONSE} characters and was safely truncated to protect the network payload pipeline.]"
        return truncated + warning
    return text_data


# ----------------------------------------------------------
# COMPILE COMMANDS DATABASE
# ----------------------------------------------------------

# Resolved at import time. Maps absolute resolved path -> compile_commands entry dict.
# O(1) lookup per file; entries look like:
#   {"file": "...", "command": "clang++ -I... -c foo.cpp -o foo.obj", "directory": "..."}
_COMPILE_LOOKUP: Dict[str, dict] = {}
_COMPILE_DB_PATH: Path | None = None
_COMPILE_DB_MTIME: float = 0.0


def _find_compile_commands() -> Path | None:
    try:
        matches = []
        for root in (
            ALLOWED_ROOT_DIR / "out" / "build",
            ALLOWED_ROOT_DIR / "build",
        ):
            if not root.exists():
                continue

            process = subprocess.run(
                [
                    "rg",
                    "--files",
                    "-g",
                    "compile_commands.json",
                    str(root),
                ],
                capture_output=True,
                text=True,
            )
            for line in process.stdout.splitlines():
                matches.append(Path(line))

        if not matches:
            return None

        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return matches[0]
    except Exception:
        return None


def _load_compile_commands() -> None:
    """
    Loads compile_commands.json into _COMPILE_LOOKUP once at startup.
    Keys are resolved absolute paths so both relative and absolute entries match.
    """
    global _COMPILE_DB_PATH
    _COMPILE_LOOKUP.clear()
    compile_db = _find_compile_commands()

    if compile_db is None:
        logger.warning(
            "No compile_commands.json found. "
            "Clang tools will fall back to bare invocation."
        )
        return

    _COMPILE_DB_PATH = compile_db

    try:
        with open(_COMPILE_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)

        build_dir = _COMPILE_DB_PATH.parent
        for entry in db:
            raw_file = entry.get("file", "")
            if not raw_file:
                continue

            raw_path = Path(raw_file)
            if raw_path.is_absolute():
                resolved = raw_path.resolve()
            else:
                resolved = (build_dir / raw_path).resolve()

            _COMPILE_LOOKUP[str(resolved)] = entry

        logger.info(
            f"Compile commands database loaded: {len(_COMPILE_LOOKUP)} entries from {_COMPILE_DB_PATH}"
        )
    except Exception as exc:
        logger.error(f"Failed to load compile_commands.json: {exc}")


def _ensure_compile_db_loaded() -> None:
    global _COMPILE_DB_MTIME
    path = _find_compile_commands()

    if path is None:
        return

    mtime = path.stat().st_mtime
    if path != _COMPILE_DB_PATH or mtime != _COMPILE_DB_MTIME:
        _load_compile_commands()


def get_compile_command_for_file(target: Path) -> List[str]:
    """
    Looks up the compile command for *target* in the cached database.
    Returns the tokenised argument list (argv-style, compiler as argv[0]).
    Returns an empty list if no entry is found.

    The entry's "command" string is shell-split.  If the entry uses "arguments"
    (array form) that is used directly instead.
    """
    _ensure_compile_db_loaded()

    key = str(target.resolve())
    entry = _COMPILE_LOOKUP.get(key)
    if entry is None:
        return []

    if "arguments" in entry:
        return list(entry["arguments"])

    raw_cmd = entry.get("command", "")
    if not raw_cmd:
        return []

    # shlex.split handles quoted paths with spaces correctly.
    # posix=False keeps Windows-style backslash paths intact.
    try:
        return shlex.split(raw_cmd, posix=False)
    except ValueError:
        # Fall back to naive whitespace split if shlex chokes on the string
        return raw_cmd.split()


def sanitize_compile_command(command: List[str]) -> List[str]:
    """
    Strips build-only flags from a compile command so it can be reused for
    analysis passes (syntax-only, AST dump, preprocessor, dependency scan).

    Preserves the original compiler from compile_commands.json.

    Removes:
        /c
        -c
        /Fo*
        /Fd*
        -o <file>
        -o<file>
        @response_files
        --driver-mode=*
    """
    if not command:
        return []

    result: List[str] = [command[0]]  # normalise compiler
    skip_next = False

    for token in command[1:]:  # skip original compiler at index 0
        if skip_next:
            skip_next = False
            continue
        normalized = token.lower()

        #
        # GCC / Clang build outputs
        #
        if token == "-c":
            continue
        if token == "-o":
            skip_next = True
            continue
        if token.startswith("-o"):
            continue

        #
        # MSVC build outputs
        #
        if normalized == "/c":
            continue
        if token.startswith("/Fo"):
            continue
        if token.startswith("/Fd"):
            continue

        #
        # Response / module files generated by VS+CMake
        #
        if token.startswith("@"):
            continue

        #
        # Clang driver noise
        #
        if token.startswith("--driver-mode"):
            continue
        result.append(token)
    return result


# Load the database immediately when the module is imported.
_load_compile_commands()


# ----------------------------------------------------------
# HEADER INCLUDES AND DEPENDENCY RESOLVER
# ----------------------------------------------------------


_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')


async def resolve_include_path(include: str, include_dirs: List[Path]) -> str | None:
    """
    Resolves an include using:

        1. Compiler include directories from compile_commands.json.
        2. Workspace-wide ripgrep fallback.

    Returns the fully resolved absolute path if found.
    """

    include = include.strip()

    if not include:
        return None

    include_path = Path(include)

    #
    # Already absolute.
    #
    if include_path.is_absolute():
        return str(include_path.resolve())

    #
    # First try compiler include directories.
    #
    expected_rel = include.replace("\\", "/")
    filename = include_path.name

    for include_dir in include_dirs:
        process = await asyncio.create_subprocess_exec(
            "rg",
            "--files",
            "-g",
            filename,
            str(include_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()
        matches = [
            Path(line.strip())
            for line in stdout.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]

        #
        # Prefer exact relative-path match.
        #
        for match in matches:
            candidate = match if match.is_absolute() else include_dir / match
            try:
                rel = candidate.resolve().relative_to(include_dir.resolve()).as_posix()
                if rel == expected_rel:
                    return str(candidate.resolve())
            except Exception:
                pass

        #
        # Fallback to first filename match
        # inside this include directory.
        #
        if matches:
            candidate = matches[0]
            if not candidate.is_absolute():
                candidate = include_dir / candidate
            return str(candidate.resolve())

    #
    # No compile_commands match.
    # Fallback to full workspace search.
    #
    process = await asyncio.create_subprocess_exec(
        "rg",
        "--files",
        "-g",
        filename,
        str(ALLOWED_ROOT_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    matches = [
        Path(line.strip())
        for line in stdout.decode(
            "utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    #
    # Prefer exact relative include match.
    #
    for match in matches:
        candidate = match if match.is_absolute() else ALLOWED_ROOT_DIR / match
        normalized = str(candidate).replace("\\", "/")
        if normalized.endswith(expected_rel):
            return str(candidate.resolve())

    #
    # Last resort.
    #
    if matches:
        candidate = matches[0]
        if not candidate.is_absolute():
            candidate = ALLOWED_ROOT_DIR / candidate
        return str(candidate.resolve())

    return None


def extract_include_dirs(compile_command: List[str]) -> List[Path]:
    """
    Extract include search directories from a compile command.

    Supports:

        MSVC:
            /IC:\\foo
            /I C:\\foo

        GCC/Clang:
            -IC:\\foo
            -I C:\\foo
            -isystem C:\\foo

    Returns:
        List[Path] preserving compiler search order.
    """
    include_dirs: List[Path] = []

    i = 0
    while i < len(compile_command):
        token = compile_command[i]

        #
        # MSVC:
        #
        # /IC:\foo
        #
        if token.startswith("/I") and len(token) > 2:
            include_path = token[2:].strip()
            if include_path:
                candidate = Path(include_path)
                try:
                    candidate = candidate.resolve()
                except Exception:
                    pass
                include_dirs.append(candidate)
            i += 1
            continue

        #
        # MSVC:
        #
        # /I C:\foo
        #
        if token == "/I":
            if i + 1 < len(compile_command):
                candidate = Path(compile_command[i + 1])
                try:
                    candidate = candidate.resolve()
                except Exception:
                    pass
                include_dirs.append(candidate)
                i += 2
                continue

        #
        # GCC/Clang:
        #
        # -IC:\foo
        #
        if token.startswith("-I") and len(token) > 2:
            candidate = Path(token[2:])
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            include_dirs.append(candidate)
            i += 1
            continue

        #
        # GCC/Clang:
        #
        # -I C:\foo
        #
        if token == "-I":
            if i + 1 < len(compile_command):
                candidate = Path(compile_command[i + 1])
                try:
                    candidate = candidate.resolve()
                except Exception:
                    pass
                include_dirs.append(candidate)
                i += 2
                continue

        #
        # GCC/Clang:
        #
        # -isystem C:\foo
        #
        if token == "-isystem":
            if i + 1 < len(compile_command):
                candidate = Path(compile_command[i + 1])
                try:
                    candidate = candidate.resolve()
                except Exception:
                    pass

                include_dirs.append(candidate)
                i += 2
                continue
        i += 1

    #
    # Remove duplicates while preserving order.
    #
    seen = set()
    result: List[Path] = []

    for include_dir in include_dirs:
        key = str(include_dir).lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(include_dir)
    return result


async def parse_direct_includes(target: Path) -> List[str]:
    with open(
        target,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        lines = f.readlines()

    includes = []
    for line in lines:
        match = _INCLUDE_RE.match(line)
        if match:
            includes.append(match.group(1).strip())
    return includes


async def build_dependency_tree(
    target: Path, depth: int, max_depth: int, visited: set[str]
) -> DependencyNode:
    canonical = str(target.resolve()).lower()
    if canonical in visited:
        return DependencyNode(
            path=str(target),
            children=[],
            cyclic=True,
        )

    if depth >= max_depth:
        return DependencyNode(
            path=str(target),
            children=[],
            truncated=True,
        )

    visited.add(canonical)
    compile_cmd = get_compile_command_for_file(target)
    include_dirs = extract_include_dirs(compile_cmd)
    direct_includes = await parse_direct_includes(target)

    children = []
    for include in direct_includes:
        resolved = await resolve_include_path(include, include_dirs)
        if not resolved:
            continue

        resolved_path = Path(resolved)
        child = await build_dependency_tree(
            resolved_path, depth + 1, max_depth, visited.copy()
        )
        children.append(child)
    return DependencyNode(path=str(target.resolve()), children=children)


def format_dependency_tree(
    node: DependencyNode, prefix: str = "", is_last: bool = True
) -> List[str]:
    marker = "└── " if is_last else "├── "
    logger.info(node.path)
    try:
        label = Path(node.path).relative_to(ALLOWED_ROOT_DIR).as_posix()
    except ValueError:
        label = node.path
    if node.cyclic:
        label += " [CYCLIC]"
    if node.truncated:
        label += " [MAX DEPTH]"

    lines = [prefix + marker + label]
    child_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(node.children):
        lines.extend(
            format_dependency_tree(
                child,
                child_prefix,
                i == len(node.children) - 1,
            )
        )
    return lines


# ----------------------------------------------------------
# GIT SANDBOXING
# ----------------------------------------------------------


def initialize_git_workspace() -> GitWorkspace | None:
    try:
        repo = git.Repo(ALLOWED_ROOT_DIR)
        branch_names = {branch.name for branch in repo.branches}

        current_branch = repo.active_branch.name
        logger.info(f"Current branch before MCP startup: {current_branch}")

        #
        # Create sandbox branch if missing.
        #
        if SANDBOX_BRANCH not in branch_names:
            logger.info(f"Creating sandbox branch '{SANDBOX_BRANCH}'")
            repo.git.checkout("-b", SANDBOX_BRANCH)

        #
        # Otherwise switch to it.
        #
        else:
            logger.info(f"Switching to sandbox branch '{SANDBOX_BRANCH}'")
            repo.git.checkout(SANDBOX_BRANCH)

        logger.info(
            f"MCP workspace initialized on branch: " f"{repo.active_branch.name}"
        )

        return GitWorkspace(repo=repo, sandbox_branch=SANDBOX_BRANCH)

    except git.InvalidGitRepositoryError:
        logger.warning("Workspace is not a git repository.")
        return None

    except Exception as exc:
        logger.error(f"Failed to initialize sandbox: {exc}")
        return None


def create_server():
    # Setup sandbox git branch
    workspace = initialize_git_workspace()
    repo = workspace.repo if workspace else None

    def _blocking_git_ls(target_dir: Path) -> List[str]:
        if not repo:
            output = []
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if not f.startswith("."):
                        full_p = Path(root) / f
                        output.append(full_p.relative_to(ALLOWED_ROOT_DIR).as_posix())
            return output
        relative_target = target_dir.relative_to(ALLOWED_ROOT_DIR).as_posix()
        tracked_files = repo.git.ls_files(relative_target).splitlines()
        return [f_str for f_str in tracked_files if not f_str.startswith(".")]

    # Setup MCP Server
    mcp = FastMCP(name="Local Source Tree Explorer", instructions=server_instructions)

    # MCP Tools

    @mcp.tool(annotations={"destructiveHint": False})
    async def workspace_status() -> str:
        """
        Returns git workspace information.
        """
        if not repo:
            return "Git repository unavailable."

        def _status():
            status = repo.git.status("--short")
            return (
                f"Branch: {repo.active_branch.name}\n\n"
                f"{status or 'Workspace Clean'}"
            )

        return await asyncio.to_thread(_status)

    @mcp.tool(annotations={"destructiveHint": False})
    async def list_directory(relative_path: str = "") -> str:
        """Recursively lists all files tracked in the repository workspace."""
        try:
            target = secure_path(relative_path)
            if not target.exists():
                return "Error: Directory target does not exist."
            files = await asyncio.to_thread(_blocking_git_ls, target)
            output = [f"[FILE] {f}" for f in sorted(files)]
            return enforce_truncation_safety(
                "\n".join(output) if output else "Directory layout is empty."
            )
        except Exception as e:
            return f"Scan failed: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def read_entire_file(relative_path: str) -> str:
        """
        Reads the requested source file completely into the context window.
        """

        def _blocking_read():
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Path targeted is not a valid or accessible source file."

            size_mb = target.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                return f"Error: Payload size ({size_mb:.2f}MB) crosses maximum server memory caps."

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Format rows with clear line numbers so the LLM can reference bugs accurately
            formatted_lines = [
                f"{idx + 1:5}: {line.rstrip()}" for idx, line in enumerate(lines)
            ]
            return "\n".join(formatted_lines)

        try:
            raw_out = await asyncio.to_thread(_blocking_read)
            if raw_out.startswith("Error:"):
                return raw_out
            return enforce_truncation_safety(raw_out)
        except Exception as e:
            return f"File transmission error: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def search_text(query: str, relative_path: str = "") -> str:
        try:
            import json as _json

            target = secure_path(relative_path)
            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--column",
                "--line-number",
                "--smart-case",
                "--fixed-strings",
                query,
                str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 2:
                return f"Ripgrep Failure:\n{stderr.decode()}"

            raw = stdout.decode("utf-8", errors="replace")
            if not raw.strip():
                return "No string matches found."

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            results = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                for sub in data.get("submatches", []):
                    col = sub["start"] + 1
                    text = data["lines"]["text"].rstrip()
                    results.append(f"{path}:{line_no}:{col}: {text}")

            return (
                enforce_truncation_safety("\n".join(results))
                if results
                else "No string matches found."
            )
        except Exception as e:
            return f"Grep analysis error: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def search_text_ripgrep(query: str, relative_path: str = "") -> str:
        """
        Ultra-fast substring or regex grepping across repository files using Ripgrep (rg).
        Automatically respects .gitignore rules and enforces a clean truncation safety window.
        """
        try:
            import json as _json

            # 1. Anchor and validate target path boundaries
            target_directory = secure_path(relative_path)
            if not target_directory.exists():
                return "Error: Targeted directory scope does not exist."

            logger.info(
                f"Initiating Ripgrep subprocess for query: '{query}' in {relative_path}"
            )

            # 2. Spawn the Ripgrep subprocess asynchronously
            # --json: Structured NDJSON output, one object per line
            # --column: Include column numbers
            # --line-number: Include line matching numbers
            # --smart-case: Case-insensitive searching unless query contains uppercase letters
            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--column",
                "--line-number",
                "--smart-case",
                query,
                str(target_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 3. Await execution and catch byte buffers cleanly without blocking the event loop
            stdout_bytes, stderr_bytes = await process.communicate()

            if process.returncode == 2:
                error_msg = stderr_bytes.decode("utf-8", errors="replace")
                return f"Ripgrep Execution Failure: {error_msg.strip()}"

            raw_output = stdout_bytes.decode("utf-8", errors="replace")

            if not raw_output.strip():
                return f"No matches found for string pattern: '{query}'"

            # 4. Parse NDJSON and relativize paths
            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            clean_lines = []
            for line in raw_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                for sub in data.get("submatches", []):
                    col = sub["start"] + 1
                    text = data["lines"]["text"].rstrip()
                    clean_lines.append(f"{path}:{line_no}:{col}: {text}")

            if not clean_lines:
                return f"No matches found for string pattern: '{query}'"

            # 5. Route through your single-pass truncation manager
            return enforce_truncation_safety("\n".join(clean_lines))

        except FileNotFoundError:
            return "Error: 'rg' executable not found on your System PATH. Please verify Ripgrep installation variables."
        except Exception as e:
            return f"Search execution halted: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def get_outline(relative_path: str) -> str:
        """
        Generates a semantic outline using Clang AST.

        Unlike regex-based scanning, this correctly identifies:

        - classes
        - structs
        - enums
        - namespaces
        - functions
        - methods
        - constructors
        - destructors
        - templates

        while ignoring local variables and implementation details.
        """

        try:
            target = secure_path(relative_path)

            if not target.is_file() or not is_allowed_file(target):
                return "Error: Invalid or inaccessible source file descriptor."

            logger.info(f"Generating semantic Clang AST outline for: {relative_path}")

            base_cmd = get_compile_command_for_file(target)
            if base_cmd:
                cmd = sanitize_compile_command(base_cmd) + [
                    "-fsyntax-only",
                    "-Xclang",
                    "-ast-dump",
                    "-fno-color-diagnostics",
                ]
            else:
                logger.warning(
                    f"No compile_commands entry for {target}; falling back to bare clang invocation."
                )
                cmd = [
                    "clang",
                    "-fsyntax-only",
                    "-Xclang",
                    "-ast-dump",
                    "-fno-color-diagnostics",
                    str(target),
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            ast_text = stdout.decode("utf-8", errors="replace")

            if process.returncode != 0 and not ast_text.strip():
                return "AST Generation Failed:\n" + stderr.decode(
                    "utf-8", errors="replace"
                )

            interesting_nodes = (
                "NamespaceDecl",
                "ClassDecl",
                "CXXRecordDecl",
                "StructDecl",
                "EnumDecl",
                "FunctionDecl",
                "CXXMethodDecl",
                "CXXConstructorDecl",
                "CXXDestructorDecl",
                "FunctionTemplateDecl",
                "ClassTemplateDecl",
            )

            outline = []
            for line in ast_text.splitlines():
                if not any(node in line for node in interesting_nodes):
                    continue
                stripped = line.strip()
                indent = len(line) - len(line.lstrip(" |`-"))
                outline.append(f"{'  ' * (indent // 2)}{stripped}")
            if not outline:
                return "No structural symbols were identified by Clang AST."
            return enforce_truncation_safety("\n".join(outline))

        except FileNotFoundError:
            return "Error: 'clang' executable not found on your System PATH."
        except Exception as e:
            return f"Outline compilation failed: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def find_definition(symbol: str) -> str:
        """
        Attempts to locate the definition of a function, method,
        class, struct, enum, namespace, or variable.
        """
        try:
            import json as _json

            if not symbol.strip():
                return "Error: Symbol cannot be empty."
            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--line-number",
                "--column",
                "--smart-case",
                "-g",
                "*.cpp",
                "-g",
                "*.c",
                "-g",
                "*.h",
                "-g",
                "*.hpp",
                "-g",
                "*.ixx",
                "-g",
                "*.inl",
                "-e",
                rf"\b{re.escape(symbol)}\b",
                str(ALLOWED_ROOT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 2:
                return stderr.decode()

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            candidates = []

            for line in stdout.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                text = data["lines"]["text"].rstrip()
                lower = text.lower()

                if any(
                    x in lower
                    for x in [
                        "class ",
                        "struct ",
                        "enum ",
                        "namespace ",
                        f"{symbol}(",
                        f"::{symbol}(",
                    ]
                ):
                    candidates.append(f"{path}:{line_no}: {text}")

            if not candidates:
                return f"No definition found for '{symbol}'."

            return enforce_truncation_safety("\n".join(candidates[:100]))

        except Exception as e:
            return f"Definition lookup failed: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def find_callers(symbol: str) -> str:
        """
        Finds call sites of a function or method.
        """
        try:
            import json as _json

            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--line-number",
                "--column",
                "--smart-case",
                "-e",
                rf"\b{re.escape(symbol)}\s*\(",
                str(ALLOWED_ROOT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            if process.returncode == 2:
                return stderr.decode()

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            matches = []

            for line in stdout.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                text = data["lines"]["text"].rstrip()

                if f"{symbol}(" in text or f"{symbol} (" in text:
                    matches.append(f"{path}:{line_no}: {text}")

            if not matches:
                return f"No callers found for '{symbol}'."

            return enforce_truncation_safety("\n".join(matches))

        except Exception as e:
            return f"Caller lookup failed: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def find_includes(relative_path: str) -> str:
        """
        Lists all #includes and resolves them
        to their actual filesystem location.
        """
        try:
            target = secure_path(relative_path)

            with open(
                target,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:
                lines = f.readlines()

            compile_cmd = get_compile_command_for_file(target)
            include_dirs = extract_include_dirs(compile_cmd)

            results = []
            for line_no, line in enumerate(lines, start=1):
                match = _INCLUDE_RE.match(line)
                if not match:
                    continue

                include = match.group(1)
                resolved = await resolve_include_path(include, include_dirs)
                results.append(f"{line_no}: {include} -> {resolved}")
            if not results:
                return "No includes found."

            return "\n".join(results)
        except Exception as exc:
            return f"Failed: {exc}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def find_implementations(class_name: str) -> str:
        """
        Finds method implementations belonging
        to a class.
        """
        try:
            import json as _json

            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--line-number",
                "--column",
                "--smart-case",
                "-g",
                "*.cpp",
                "-g",
                "*.cxx",
                "-g",
                "*.cc",
                "-g",
                "*.ixx",
                "-e",
                rf"{re.escape(class_name)}::",
                str(ALLOWED_ROOT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 2:
                return stderr.decode()

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            results = []

            for line in stdout.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                text = data["lines"]["text"].rstrip()
                results.append(f"{path}:{line_no}: {text}")

            if not results:
                return f"No implementations found " f"for class '{class_name}'."

            return enforce_truncation_safety("\n".join(results))

        except Exception as e:
            return f"Implementation lookup failed: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def get_git_diff(cached: bool = False) -> str:
        """
        Returns the active structural modifications in the repository (staged or unstaged).
        Extremely token-efficient for debugging recent implementation edits.
        """
        if not repo:
            return "Error: This workspace root is not initialized as a git repository repository workspace."

        def _blocking_diff():
            logger.info(f"Fetching repository delta snapshot (staged={cached})")
            if cached:
                # View changes staged for commit
                return repo.git.diff("--cached")
            # View raw unstaged workspace deviations
            return repo.git.diff()

        try:
            raw_diff = await asyncio.to_thread(_blocking_diff)
            return (
                enforce_truncation_safety(raw_diff)
                if raw_diff.strip()
                else "Workspace is completely clean. No active diff structures detected."
            )
        except Exception as e:
            return f"Diff generation aborted: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def read_file_surround(
        relative_path: str, target_line: int, radius: int = 35
    ) -> str:
        """
        Reads a highly localized snippet around a target line index (line +/- radius).
        Perfect for inspecting specific error lines or grep hits with minimal token overhead.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Targeted path is not a valid code file descriptor."

            def _blocking_bounded_read():
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                total_lines = len(lines)
                if total_lines == 0:
                    return "Target file is empty."

                # Convert to 0-indexed values safely bounded by file depth
                start_idx = max(0, (target_line - 1) - radius)
                end_idx = min(total_lines, (target_line - 1) + radius + 1)

                formatted = []
                for idx in range(start_idx, end_idx):
                    # Add a visible visual pointer tracking the precise point of interest
                    pointer = "👉 " if (idx + 1) == target_line else "   "
                    formatted.append(f"{pointer}{idx + 1:5}: {lines[idx].rstrip()}")

                return "\n".join(formatted)

            raw_out = await asyncio.to_thread(_blocking_bounded_read)
            return enforce_truncation_safety(raw_out)
        except Exception as e:
            return f"Targeted window read failed: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def cpp_lint_file(relative_path: str) -> str:
        """
        Runs a dry-run syntax and semantic check on a targeted C/C++ file using MSVC CL/Clang.
        Automatically uses the project's compile_commands.json so all include paths,
        defines, and language standard flags are correct.
        Returns precise compilation errors, warning alerts, and suggestions.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Invalid or inaccessible file descriptor."

            logger.info(f"Executing Clang semantic lint pass for: {relative_path}")

            base_cmd = get_compile_command_for_file(target)
            if base_cmd:
                cmd = sanitize_compile_command(base_cmd)
                compiler = Path(cmd[0]).name.lower()
                if compiler in ("cl.exe", "cl"):
                    cmd.append("/Zs")
                else:
                    cmd.extend(
                        [
                            "-fsyntax-only",
                            "-fno-color-diagnostics",
                        ]
                    )
            else:
                logger.warning(
                    f"No compile_commands entry for {target}; falling back to bare clang invocation."
                )
                cmd = [
                    "clang",
                    "-fsyntax-only",
                    "-fno-color-diagnostics",
                    str(target),
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            raw_stdout = stdout.decode("utf-8", errors="replace")
            raw_stderr = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return enforce_truncation_safety(
                    raw_stderr if raw_stderr.strip() else raw_stdout
                )

            if process.returncode == 0:
                return "SUCCESS: Syntax and semantic analysis passed with 0 errors/warnings."

            combined = ""
            if raw_stdout.strip():
                combined += raw_stdout

            if raw_stderr.strip():
                if combined:
                    combined += "\n"
                combined += raw_stderr

            if not combined.strip():
                return "SUCCESS: Syntax and semantic analysis passed with 0 errors/warnings."

            return enforce_truncation_safety(combined)
        except FileNotFoundError:
            return "Error: 'clang' executable not found on your System PATH. Please verify installation variables."
        except Exception as e:
            return f"Clang linting aborted: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def clang_dump_ast(relative_path: str, filter_symbol: str = "") -> str:
        """
        Dumps the Clang Abstract Syntax Tree (AST) of a source file in JSON format.
        Automatically uses the project's compile_commands.json for correct include paths.
        Pass a filter_symbol string to target a specific class, function, or scope block.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Invalid file descriptor targeting."

            logger.info(
                f"Dumping Clang AST for file {relative_path} (Filter: '{filter_symbol}')"
            )

            base_cmd = get_compile_command_for_file(target)
            if base_cmd:
                # -ast-dump=json emits structured JSON the LLM can reason over directly.
                cmd = sanitize_compile_command(base_cmd) + [
                    "-fsyntax-only",
                    "-Xclang",
                    "-ast-dump=json",
                    "-fno-color-diagnostics",
                ]
            else:
                logger.warning(
                    f"No compile_commands entry for {target}; falling back to bare clang invocation."
                )
                cmd = [
                    "clang",
                    "-fsyntax-only",
                    "-Xclang",
                    "-ast-dump=json",
                    "-fno-color-diagnostics",
                    str(target),
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            raw_ast = stdout.decode("utf-8", errors="replace")

            if process.returncode != 0 and not raw_ast.strip():
                return f"AST Generation Failed:\n{stderr.decode('utf-8', errors='replace')}"

            # Filter the JSON AST by symbol name if requested.
            # Because -ast-dump=json produces one large JSON object, we do a simple
            # line-scan for the symbol rather than fully parsing the tree — this keeps
            # token cost low while still narrowing the result to the relevant subtree.
            if filter_symbol:
                lines = raw_ast.splitlines()
                filtered_lines = []
                capture = False
                brace_depth = 0

                for line in lines:
                    if filter_symbol in line:
                        capture = True
                        brace_depth = 0

                    if capture:
                        filtered_lines.append(line)
                        brace_depth += line.count("{") - line.count("}")
                        # Stop capturing once the JSON object that matched is closed
                        if brace_depth <= 0 and filtered_lines:
                            capture = False

                return (
                    enforce_truncation_safety("\n".join(filtered_lines))
                    if filtered_lines
                    else f"Symbol '{filter_symbol}' not isolated inside the AST dump structure."
                )

            return enforce_truncation_safety(raw_ast)
        except Exception as e:
            return f"AST compilation pass aborted: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def clang_expand_preprocessor(relative_path: str) -> str:
        """
        Runs Clang's preprocessor pass (-E) to expand all macros, includes, and
        conditional compilation blocks. Uses compile_commands.json so all project
        defines and include paths are correctly applied.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: File destination invalid."

            logger.info(
                f"Expanding preprocessor tokens using Clang for: {relative_path}"
            )

            base_cmd = get_compile_command_for_file(target)
            if base_cmd:
                # -E: preprocessor only  -P: suppress line markers for clean output
                cmd = sanitize_compile_command(base_cmd) + ["-E", "-P"]
            else:
                logger.warning(
                    f"No compile_commands entry for {target}; falling back to bare clang invocation."
                )
                cmd = ["clang", "-E", "-P", str(target)]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            raw_expansion = stdout.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return f"Preprocessor Expansion Failed:\n{stderr.decode('utf-8', errors='replace')}"

            # Strip excessive consecutive empty newline sequences to save token limits
            cleaned_lines = []
            for line in raw_expansion.splitlines():
                if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
                    cleaned_lines.append(line)

            return enforce_truncation_safety("\n".join(cleaned_lines))
        except Exception as e:
            return f"Preprocessor pipeline error: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def get_file_dependencies(relative_path: str, max_depth: int = 5) -> str:
        """
        Builds a recursive header dependency tree.

        - Resolves includes using compile_commands.json.
        - Traverses transitive dependencies.
        - Detects cyclic includes.
        - Limits recursion depth to prevent runaway graphs.

        Unlike find_includes(), this follows includes recursively
        and reports the full dependency hierarchy.
        """
        try:
            target = secure_path(relative_path)
            tree = await build_dependency_tree(
                target,
                depth=0,
                max_depth=max_depth,
                visited=set(),
            )

            output = [Path(tree.path).name]

            for i, child in enumerate(tree.children):
                output.extend(
                    format_dependency_tree(child, "", i == len(tree.children) - 1)
                )
            return "\n".join(output)

        except Exception as exc:
            return f"Failed to build dependency graph: {exc}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def get_git_log(max_commits: int = 10, relative_path: str = "") -> str:
        """
        Retrieves the repository commit log history. Pass a relative file path
        to isolate changes made specifically to that targeted file context.
        """
        if not repo:
            return "Error: Target workspace root is not an active git repository."

        def _blocking_log():
            logger.info(f"Pulling Git telemetry log context (limit={max_commits})")
            args = [
                "-n",
                str(max_commits),
                "--oneline",
                "--decorate",
            ]
            if relative_path:
                target = secure_path(relative_path)
                args.append(f"-- {str(target)}")
            return repo.git.log(*args)

        try:
            raw_log = await asyncio.to_thread(_blocking_log)
            return (
                enforce_truncation_safety(raw_log)
                if raw_log.strip()
                else "No commit entry records found."
            )
        except Exception as e:
            return f"Telemetry fetch failed: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def find_symbol_references(symbol_name: str) -> str:
        """
        Scans the workspace and lists only the filenames that contain references to the given symbol.
        Highly efficient for tracking down file dependencies or mapping signal/function usage.
        """
        try:
            import json as _json

            logger.info(f"Locating files referencing symbol: '{symbol_name}'")

            process = await asyncio.create_subprocess_exec(
                "rg",
                "--json",
                "--line-number",
                "--column",
                "--smart-case",
                "--fixed-strings",
                symbol_name,
                str(ALLOWED_ROOT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 2:
                return f"Ripgrep Failure:\n{stderr.decode('utf-8', errors='replace')}"

            raw_output = stdout.decode("utf-8", errors="replace")
            if not raw_output.strip():
                return f"No references found for symbol: '{symbol_name}'"

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            seen_files: dict[str, int] = {}  # path -> match count

            for line in raw_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                path = obj["data"]["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                seen_files[path] = seen_files.get(path, 0) + 1

            if not seen_files:
                return f"No references found for symbol: '{symbol_name}'"

            lines = [
                f"  📍 {path}  ({count} match{'es' if count != 1 else ''})"
                for path, count in sorted(seen_files.items())
            ]
            return enforce_truncation_safety("\n".join(lines))
        except Exception as e:
            return f"Symbol locator failed: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def get_directory_tree(relative_path: str = "", depth: int = 3) -> str:
        """
        Generates a visual directory tree structure up to a specified depth.
        Includes tracked and untracked files.
        Optimized using os.scandir().
        """
        try:
            target = secure_path(relative_path)
            if not target.exists():
                return "Error: Target path does not exist."
            if not target.is_dir():
                return "Error: Target path is not a directory."

            def _build_tree() -> str:
                output = []
                root_name = target.name if relative_path else ALLOWED_ROOT_DIR.name
                output.append(f"{root_name}/")

                def walk(directory: Path, prefix: str, current_depth: int):
                    if current_depth > depth:
                        return
                    try:
                        dirs = []
                        files = []
                        with os.scandir(directory) as entries:
                            for entry in entries:
                                if entry.name.startswith("."):
                                    continue
                                if entry.name in IGNORED_DIRS:
                                    continue
                                try:
                                    if entry.is_dir(follow_symlinks=False):
                                        dirs.append(entry)
                                    elif (
                                        Path(entry.name).suffix.lower()
                                        in ALLOWED_EXTENSIONS
                                    ):
                                        files.append(entry)
                                except OSError:
                                    continue

                        dirs.sort(key=lambda e: e.name.lower())
                        files.sort(key=lambda e: e.name.lower())
                        items = dirs + files

                        for index, entry in enumerate(items):
                            is_last = index == len(items) - 1
                            connector = "└── " if is_last else "├── "
                            if entry.is_dir(follow_symlinks=False):
                                output.append(f"{prefix}{connector}{entry.name}/")
                                next_prefix = (
                                    prefix + "    " if is_last else prefix + "│   "
                                )
                                walk(Path(entry.path), next_prefix, current_depth + 1)
                            else:
                                output.append(f"{prefix}{connector}{entry.name}")
                    except PermissionError:
                        return

                walk(target, "", 1)
                return "\n".join(output)

            result = await asyncio.to_thread(_build_tree)
            return enforce_truncation_safety(result)
        except Exception as e:
            return f"Tree compilation failed: {str(e)}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def search_text_multi_pattern(
        patterns: list[str], relative_path: str = ""
    ) -> str:
        """
        Executes a high-performance multi-pattern literal search using Ripgrep.

        Unlike regex-based matching, this performs fixed-string searches,
        making it ideal for source code symbols such as:

            RunJobs(
            JobSystem::Get()
            std::atomic<u32>
            operator<<

        All patterns are treated as literal strings.
        """
        try:
            target = secure_path(relative_path)
            if not patterns:
                return "Error: Pattern matching array cannot be empty."
            logger.info(
                f"Executing multi-pattern Ripgrep search for {len(patterns)} patterns"
            )
            cmd = [
                "rg",
                "--json",
                "--column",
                "--line-number",
                "--smart-case",
                "--fixed-strings",
            ]

            for pattern in patterns:
                if pattern.strip():
                    cmd.extend(["-e", pattern])

            cmd.append(str(target))
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            if process.returncode == 2:
                return "Ripgrep Failure:\n" + stderr.decode("utf-8", errors="replace")
            raw_output = stdout.decode("utf-8", errors="replace")

            if not raw_output.strip():
                return (
                    "No matches found matching any of the "
                    f"conditions inside: {patterns}"
                )

            import json as _json

            root_str = ALLOWED_ROOT_DIR.as_posix().lower()
            cleaned_lines = []
            for line in raw_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                path = data["path"]["text"].replace("\\", "/")
                idx = path.lower().find(root_str)
                if idx != -1:
                    path = path[idx + len(root_str) :].lstrip("/")
                line_no = data["line_number"]
                for sub in data.get("submatches", []):
                    col = sub["start"] + 1
                    text = data["lines"]["text"].rstrip()
                    cleaned_lines.append(f"{path}:{line_no}:{col}: {text}")

            return (
                enforce_truncation_safety("\n".join(cleaned_lines))
                if cleaned_lines
                else (
                    "No matches found matching any of the "
                    f"conditions inside: {patterns}"
                )
            )

        except FileNotFoundError:
            return "Error: 'rg' executable not found on " "your System PATH."
        except Exception as e:
            return f"Multi-search execution aborted: {e}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def apply_patch(unified_diff: str) -> str:
        """
        Applies a unified git patch.
        """
        if not repo:
            return "Error: This workspace root is not initialized as a git repository repository workspace."

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".patch", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(unified_diff)
                patch_path = f.name

            def _apply():
                repo.git.apply(
                    "--verbose",
                    "--ignore-whitespace",
                    "--whitespace=nowarn",
                    patch_path,
                )

            # apply the git patch
            await asyncio.to_thread(_apply)
            return "Patch applied successfully."

        except Exception as exc:
            return f"Patch apply failed:\n{exc}"
        # clean up patch files afterwards
        finally:
            try:
                os.remove(patch_path)
            except:
                pass

    @mcp.tool(annotations={"destructiveHint": False})
    async def apply_patch_and_validate(
        unified_diff: str, files_to_validate: list[str]
    ) -> str:
        """
        Applies a unified git patch and validates if the patched files compiles without any errors.
        """
        apply_result = await apply_patch(unified_diff)
        if "failed" in apply_result.lower():
            return apply_result

        # try compiling each file and check if the patch is compiling
        diagnostics = []
        for file in files_to_validate:
            if not _is_cpp_file(file):
                diagnostics.append(
                    {
                        "file": file,
                        "status": "skipped",
                        "reason": "Not a C/C++ source file",
                    }
                )
                continue

            try:
                result = await cpp_lint_file(file)
                diagnostics.append({"file": file, "diagnostics": result})
            except Exception as exc:
                diagnostics.append(
                    {"file": file, "status": "failed", "reason": str(exc)}
                )
        return json.dumps({"success": True, "validation": diagnostics}, indent=2)

    @mcp.tool(annotations={"destructiveHint": False})
    async def replace_text(
        relative_path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> str:
        """
        Replaces text inside a source file.

        By default exactly one occurrence must exist.
        This prevents accidental modifications.

        Set replace_all=True to replace every occurrence.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Invalid source file."

            def _replace():
                with open(
                    target,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    contents = f.read()

                occurrence_count = contents.count(old_text)
                if occurrence_count == 0:
                    return "Error: Search text not found.\n" f"File: {relative_path}"

                if not replace_all and occurrence_count > 1:
                    return (
                        "Error: Search text is ambiguous.\n"
                        f"Found {occurrence_count} occurrences.\n"
                        "Use replace_all=True or provide a more specific anchor."
                    )

                if replace_all:
                    updated = contents.replace(old_text, new_text)
                    replacements = occurrence_count
                else:
                    updated = contents.replace(old_text, new_text, 1)
                    replacements = 1

                with open(
                    target,
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    f.write(updated)
                return (
                    f"Success: Replaced {replacements} "
                    f"occurrence(s) in '{relative_path}'."
                )

            return await asyncio.to_thread(_replace)
        except Exception as exc:
            return f"Replace failed: {exc}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def insert_after_text(
        relative_path: str,
        anchor_text: str,
        text_to_insert: str,
        occurrence: int = 1,
    ) -> str:
        """
        Inserts text immediately after a matching anchor string.

        Example
            insert_after_text(
                "foo.cpp",
                "void Foo()",
                "\n// Inserted text\n"
            )

        Rules:
            - Anchor must exist.
            - occurrence is 1-based.
            - Fails if the requested occurrence does not exist.
        """
        try:
            target = secure_path(relative_path)
            if not target.is_file() or not is_allowed_file(target):
                return "Error: Invalid source file."

            if not anchor_text:
                return "Error: Anchor text cannot be empty."

            if occurrence < 1:
                return "Error: occurrence must be >= 1."

            def _insert() -> str:
                with open(
                    target,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    contents = f.read()

                occurrence_count = contents.count(anchor_text)
                if occurrence_count == 0:
                    return "Error: Anchor text not found.\n" f"File: {relative_path}"

                if occurrence > occurrence_count:
                    return (
                        f"Error: Requested occurrence {occurrence}, "
                        f"but only {occurrence_count} occurrence(s) exist."
                    )

                search_start = 0
                anchor_index = -1

                for _ in range(occurrence):
                    anchor_index = contents.find(anchor_text, search_start)
                    if anchor_index == -1:
                        return (
                            "Error: Internal search failure while "
                            "locating anchor occurrence."
                        )

                    search_start = anchor_index + len(anchor_text)
                insert_position = anchor_index + len(anchor_text)

                updated_contents = (
                    contents[:insert_position]
                    + text_to_insert
                    + contents[insert_position:]
                )

                with open(
                    target,
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    f.write(updated_contents)

                return (
                    f"Success: Inserted text after occurrence "
                    f"{occurrence} of anchor in '{relative_path}'."
                )

            return await asyncio.to_thread(_insert)

        except Exception as exc:
            return f"Insert failed: {exc}"

    @mcp.tool(annotations={"destructiveHint": False})
    async def validate_files(files: list[str]) -> str:
        """
        Runs compilation/syntax validation on a set of files.
        """
        diagnostics = []
        for file in files:
            if not _is_cpp_file(file):
                diagnostics.append(
                    {
                        "file": file,
                        "status": "skipped",
                        "reason": "Not a C/C++ source file",
                    }
                )
                continue

            try:
                result = await cpp_lint_file(file)
                diagnostics.append(
                    {
                        "file": file,
                        "success": result.startswith("SUCCESS"),
                        "diagnostics": result,
                    }
                )
            except Exception as exc:
                diagnostics.append(
                    {"file": file, "success": False, "diagnostics": str(exc)}
                )

        return json.dumps(
            {"success": True, "validation": diagnostics},
            indent=2,
        )

    return mcp


# ==========================================================
# EXECUTION ENTRYPOINT
# ==========================================================


def main():
    server = create_server()
    logger.info("Starting One-Pass FastMCP Server on 0.0.0.0:8000")
    server.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
