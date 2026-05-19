#!/usr/bin/env python3
"""Check all prerequisites for bo-mcp-ui installation.

This script verifies that all requirements are met before running the MCP server.
Run with: uv run python scripts/check_prerequisites.py
"""

import shutil
import socket
import subprocess
import sys
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = "\u2713"
CROSS = "\u2717"
WARN = "\u26a0"

# Required Python version
REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 11

# Default ports used by the MCP server
DEFAULT_MCP_PORT = 8001
DEFAULT_API_PORT = 8000


def check_python_version() -> tuple[bool, str]:
    """Check if Python version meets requirements (3.11+)."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"

    if major >= REQUIRED_PYTHON_MAJOR and minor >= REQUIRED_PYTHON_MINOR:
        return True, f"Python {version_str}"
    return False, f"Python {version_str} (need {REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}+)"


def check_uv_installed() -> tuple[bool, str]:
    """Check if uv package manager is installed."""
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            result = subprocess.run(  # noqa: S603 - uv_path from shutil.which is trusted
                [uv_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return True, f"uv found at {uv_path}"
        version = result.stdout.strip()
        return True, f"uv {version}"
    return False, "uv not found (install: curl -LsSf https://astral.sh/uv/install.sh | sh)"


def check_git_available() -> tuple[bool, str]:
    """Check if Git is available."""
    git_path = shutil.which("git")
    if git_path:
        try:
            result = subprocess.run(  # noqa: S603 - git_path from shutil.which is trusted
                [git_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return True, f"git found at {git_path}"
        version = result.stdout.strip()
        return True, version
    return False, "git not found"


def check_port_available(port: int) -> tuple[bool, str]:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            if result != 0:
                return True, f"Port {port} is available"
            return False, f"Port {port} is in use"
    except OSError as e:
        return False, f"Port {port} check failed: {e}"


def check_workspace_structure() -> tuple[bool, str]:
    """Check if required project directories exist."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    required_dirs = [
        "packages/bo-engine",
        "packages/bo-mcp-server",
        "packages/bo-mcp-api",
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing.append(dir_path)

    if not missing:
        return True, "All package directories found"
    return False, f"Missing: {', '.join(missing)}"


def check_pyproject_exists() -> tuple[bool, str]:
    """Check if root pyproject.toml exists."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        return True, "pyproject.toml found"
    return False, "pyproject.toml not found (run from project root)"


def check_uv_sync_status() -> tuple[bool, str]:
    """Check if uv sync has been run (uv.lock exists)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    lock_path = project_root / "uv.lock"

    if lock_path.exists():
        return True, "uv.lock exists (dependencies synced)"
    return False, "uv.lock not found (run: uv sync)"


def print_result(name: str, passed: bool, details: str, is_warning: bool = False) -> None:
    """Print a formatted check result."""
    if passed:
        symbol = f"{GREEN}{CHECK}{RESET}"
    elif is_warning:
        symbol = f"{YELLOW}{WARN}{RESET}"
    else:
        symbol = f"{RED}{CROSS}{RESET}"

    print(f"  {symbol} {name}: {details}")


def main() -> int:
    """Run all prerequisite checks and return exit code."""
    print("\nBO-MCP-UI Prerequisites Check")
    print("=" * 40)

    checks = [
        ("Python 3.11+", check_python_version, False),
        ("uv installed", check_uv_installed, False),
        ("Git available", check_git_available, False),
        ("Project structure", check_workspace_structure, False),
        ("pyproject.toml", check_pyproject_exists, False),
        ("Dependencies synced", check_uv_sync_status, False),
        (
            f"Port {DEFAULT_MCP_PORT} (MCP SSE)",
            lambda: check_port_available(DEFAULT_MCP_PORT),
            True,
        ),
        (
            f"Port {DEFAULT_API_PORT} (API)",
            lambda: check_port_available(DEFAULT_API_PORT),
            True,
        ),
    ]

    all_passed = True
    has_warnings = False

    print("\nChecking prerequisites:\n")

    for name, check_fn, is_optional in checks:
        passed, details = check_fn()
        print_result(name, passed, details, is_warning=is_optional and not passed)

        if not passed:
            if is_optional:
                has_warnings = True
            else:
                all_passed = False

    print()

    if all_passed and not has_warnings:
        print(f"{GREEN}All checks passed - ready to run MCP server{RESET}")
        print("\nNext steps:")
        print("  1. Verify installation: uv run bo-mcp-server --verify")
        print("  2. Run MCP server: uv run bo-mcp-server")
        print("  3. Or with SSE transport: uv run bo-mcp-server --transport sse --port 8001")
        return 0
    if all_passed:
        print(f"{YELLOW}Core checks passed with warnings{RESET}")
        print("\nYou can proceed, but some ports may be in use.")
        print("If running SSE transport, ensure ports are free or use --port.")
        return 0
    print(f"{RED}Some checks failed - please fix issues above{RESET}")
    print("\nCommon fixes:")
    print("  - Python version: Use pyenv or asdf to install Python 3.11+")
    print("  - uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("  - Dependencies not synced: uv sync")
    return 1


if __name__ == "__main__":
    sys.exit(main())
