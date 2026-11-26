#!/usr/bin/env python3
"""Automatic checks for the team rules in RULES.md (standard library only).

Run from the repository root:  python tools/check_rules.py
This does not replace validate.py or a human reading the code.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

BANNED_IMPORTS = {
    "vizdoom": "the agent must not access the game engine",
    "train": "submission/ must not depend on training code",
    "training": "submission/ must not depend on training code",
    "requests": "no network access",
    "urllib": "no network access",
    "http": "no network access",
    "socket": "no network access",
    "ftplib": "no network access",
    "subprocess": "no host access",
    "multiprocessing": "no extra processes in the agent",
    "importlib": "dynamic imports hide what the agent uses",
    "stable_baselines3": "not in our approved library list",
    "sample_factory": "not in our approved library list",
    "envpool": "not in our approved library list",
    "omg": "not in our approved library list (Omgifol)",
    "tensorboard": "training utility, not for the agent",
    "matplotlib": "training utility, not for the agent",
    "tqdm": "training utility, not for the agent",
}
BANNED_ATTRS = {
    "atan2": "no trigonometry aiming",
    "arctan2": "no trigonometry aiming",
    "unwrapped": "do not reach into wrapper internals",
    "environ": "do not read the host environment",
    "getenv": "do not read the host environment",
}
BANNED_DOTTED = {
    "os.system": "no host access",
    "os.popen": "no host access",
    "torch.hub": "no network access",
    "torch.save": "the repository is read-only during evaluation",
}
BANNED_CALLS = {"eval", "exec", "__import__"}
WRITE_METHODS = {"write_text", "write_bytes"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".msgpack", ".npz", ".npy",
                   ".safetensors", ".bin", ".pkl", ".onnx"}
SECRET_NAMES = {"kaggle.json", ".env", "id_rsa", "id_ed25519"}
SECRET_PATTERN = re.compile(
    r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}|hf_[A-Za-z0-9]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")
KAGGLE_PATTERN = re.compile(
    r"^\s*[!%]*\s*(?:import|from)\s+kaggle|kagglehub|/kaggle/"
    r"|\bkaggle\s+(?:datasets|kernels|competitions)\b|pip\s+install[^\n]*\bkaggle",
    re.MULTILINE | re.IGNORECASE)
CODE_SUFFIXES = {".py", ".ipynb", ".sh", ".txt", ".toml", ".yaml", ".yml", ".json"}
SIZE_LIMIT = 1_000_000_000  # strictest reading of "1 GB"
LARGE_FILE = 10_000_000

errors, warnings = [], []


def git(*args):
    result = subprocess.run(["git", "-c", "core.quotepath=off", *args],
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check_python(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: cannot be parsed ({exc})")
        return

    def bad(node, what, why):
        errors.append(f"{path}:{node.lineno}: {what} ({why})")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                module = node.module or ""
                names = [module] if module else [a.name for a in node.names]
                for alias in node.names:
                    full = f"{module}.{alias.name}"
                    if alias.name in BANNED_ATTRS:
                        bad(node, f"imports {alias.name}", BANNED_ATTRS[alias.name])
                    if full in BANNED_DOTTED:
                        bad(node, f"imports {full}", BANNED_DOTTED[full])
            for name in names:
                for part in (name.split(".")[0], name.split(".")[-1]):
                    if part in BANNED_IMPORTS:
                        bad(node, f"imports {name}", BANNED_IMPORTS[part])
                        break
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                bad(node, f"uses .{node.attr}", BANNED_ATTRS[node.attr])
            if dotted(node) in BANNED_DOTTED:
                bad(node, f"uses {dotted(node)}", BANNED_DOTTED[dotted(node)])
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name in BANNED_CALLS:
                bad(node, f"calls {name}()", "hidden code execution")
            if isinstance(func, ast.Attribute) and func.attr in WRITE_METHODS:
                bad(node, f"calls .{func.attr}()", "no file writes in the agent")
            if name == "open":
                mode = node.args[1] if len(node.args) > 1 else next(
                    (k.value for k in node.keywords if k.arg == "mode"), None)
                if isinstance(mode, ast.Constant) and set(str(mode.value)) & set("wax+"):
                    bad(node, "opens a file for writing", "no file writes in the agent")


def check_agent_class():
    path = Path("submission/agent.py")
    if not path.exists():
        errors.append("submission/agent.py is missing")
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return  # already reported
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "DoomAgent"), None)
    if cls is None:
        errors.append("submission/agent.py has no class DoomAgent")
        return
    methods = {n.name: [a.arg for a in n.args.args]
               for n in cls.body if isinstance(n, ast.FunctionDef)}
    expected = {
        "__init__": ["self", "observation_space", "action_space", "device"],
        "reset": ["self"],
        "act": ["self", "observation"],
    }
    for name, args in expected.items():
        if name not in methods:
            errors.append(f"DoomAgent.{name}() is missing")
        elif methods[name] != args:
            errors.append(f"DoomAgent.{name} arguments are {methods[name]}, expected {args}")


def check_submission():
    folder = Path("submission")
    if not folder.is_dir():
        errors.append("submission/ folder is missing (run from the repository root)")
        return
    for path in sorted(folder.rglob("*.py")):
        check_python(path)
    for path in folder.rglob("*"):
        if "dockerfile" in path.name.lower():
            errors.append(f"{path}: do not submit a Dockerfile")
    reqs = folder / "requirements.txt"
    if reqs.exists():
        for line in reqs.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.\-\[\],]+==[A-Za-z0-9_.+!\-]+", line):
                errors.append(f"{reqs}: '{line}' is not pinned as name==version")
            else:
                warnings.append(f"{reqs}: confirm that '{line}' is approved")
    check_agent_class()


def check_repository():
    listing = git("ls-files", "-z")
    if listing is None:
        errors.append("not a git repository, so repository checks could not run")
        return
    files = [Path(p) for p in listing.split("\0") if p and Path(p).is_file()]
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        attr = git("check-attr", "filter", "--", str(path)) or ""
        in_lfs = attr.strip().endswith(": lfs")
        if path.suffix.lower() in WEIGHT_SUFFIXES and not in_lfs:
            errors.append(f"{path}: weight or data file is not stored with Git LFS")
        elif size > LARGE_FILE and not in_lfs:
            errors.append(f"{path}: larger than 10 MB and not in Git LFS")
        if path.name in SECRET_NAMES:
            errors.append(f"{path}: looks like a secrets file")
        elif size < 1_000_000 and SECRET_PATTERN.search(
                path.read_text(encoding="utf-8", errors="ignore")):
            errors.append(f"{path}: contains what looks like a token or private key")
        if (path.suffix.lower() in CODE_SUFFIXES and path.parts[0] != "tools"
                and size < 5_000_000 and KAGGLE_PATTERN.search(
                    path.read_text(encoding="utf-8", errors="ignore"))):
            errors.append(f"{path}: uses Kaggle (the team uses Google Colab only)")
    git_dir = git("rev-parse", "--git-common-dir")
    if git_dir:
        lfs_dir = Path(git_dir.strip()) / "lfs" / "objects"
        lfs = sum(p.stat().st_size for p in lfs_dir.rglob("*") if p.is_file()) if lfs_dir.is_dir() else 0
    else:
        lfs = 0
    # Tracked files plus local LFS objects: a cautious estimate, may count weights twice.
    estimate = total + lfs
    message = f"estimated repo size {estimate / 1e6:.0f} MB (limit 1 GB)"
    if total > SIZE_LIMIT or lfs > SIZE_LIMIT:
        errors.append(message)
    elif estimate > SIZE_LIMIT // 2:
        warnings.append(message + "; confirm with validate.py")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "submission")
    for name in (untracked or "").splitlines():
        warnings.append(f"{name}: not committed yet, so it will not be submitted")


def main():
    check_submission()
    check_repository()
    for item in warnings:
        print("WARNING:", item)
    for item in errors:
        print("ERROR:  ", item)
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s). Do not push to main.")
        return 1
    print("\nPASSED automatic checks. Hand-written behaviour still needs a human review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
