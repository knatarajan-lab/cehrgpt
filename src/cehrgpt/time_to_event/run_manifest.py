"""
The manifest of a time-to-event prediction run: which code, model, tokenizer, dataset and cohort.

Two runs that should agree can then be told apart by something other than their predictions.
The manifest is a json next to the prediction folder with one entry per session, a resumed run
adds an entry rather than replacing the earlier one.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any, Dict, List, Optional

# Files up to this size are hashed, the weights of a model are a few hundred megabytes
MAX_BYTES_TO_HASH = 4 * 1024**3
# The folders are described by the files directly in them. The tokenizer folder of a pretrained
# model also holds the prepared datasets, which are not what the tokenizer is.
MAX_FILES_TO_LIST = 200


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str, with_hash: bool = True) -> Dict[str, Any]:
    stat = os.stat(path)
    description = {
        "bytes": stat.st_size,
        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        ),
    }
    if with_hash and stat.st_size <= MAX_BYTES_TO_HASH:
        description["sha256"] = sha256_of_file(path)
    return description


def describe_folder(
    path: Optional[str], with_hash: bool = True, recursive: bool = False
) -> Optional[Dict[str, Any]]:
    """
    The files of a folder with their size, modification time and, when asked for, sha256.

    Only the files directly in the folder unless recursive. Without the hashes (a tokenized
    dataset is too large to read again) the number of files, their total size and the latest
    modification still tell a changed folder from an unchanged one.
    """
    if not path:
        return None
    absolute_path = os.path.abspath(path)
    if os.path.isfile(absolute_path):
        return {"path": absolute_path, **describe_file(absolute_path, with_hash)}
    if not os.path.isdir(absolute_path):
        return {"path": absolute_path, "exists": False}
    relative_paths: List[str] = []
    if recursive:
        for root, _, names in os.walk(absolute_path):
            for name in names:
                relative_paths.append(
                    os.path.relpath(os.path.join(root, name), absolute_path)
                )
    else:
        relative_paths = [
            name
            for name in os.listdir(absolute_path)
            if os.path.isfile(os.path.join(absolute_path, name))
        ]
    relative_paths.sort()
    stats = [os.stat(os.path.join(absolute_path, _)) for _ in relative_paths]
    description: Dict[str, Any] = {
        "path": absolute_path,
        "num_files": len(relative_paths),
        "total_bytes": sum(_.st_size for _ in stats),
        "latest_modified": (
            datetime.datetime.fromtimestamp(max(_.st_mtime for _ in stats)).isoformat(
                timespec="seconds"
            )
            if stats
            else None
        ),
    }
    if len(relative_paths) <= MAX_FILES_TO_LIST:
        description["files"] = {
            relative_path: describe_file(
                os.path.join(absolute_path, relative_path), with_hash
            )
            for relative_path in relative_paths
        }
    return description


def describe_tokenized_dataset(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The splits of a dataset saved with save_to_disk, with the fingerprint datasets gave each."""
    description = describe_folder(path, with_hash=False)
    if not description or "num_files" not in description:
        return description
    splits = {}
    for name in sorted(os.listdir(description["path"])):
        split_path = os.path.join(description["path"], name)
        if not os.path.isdir(split_path):
            continue
        split = describe_folder(split_path, with_hash=False)
        split.pop("files", None)
        split.pop("path", None)
        state_path = os.path.join(split_path, "state.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path) as f:
                    split["fingerprint"] = json.load(f).get("_fingerprint")
            except (OSError, ValueError):
                pass
        splits[name] = split
    if splits:
        description["splits"] = splits
    return description


def describe_code() -> Dict[str, Any]:
    """
    The cehrgpt code that is running, which is not always the installed package.

    PYTHONPATH can put another checkout in front of the installed package, so the folder the
    module was loaded from is recorded, and its git commit when that folder is a checkout.
    """
    import cehrgpt

    code_folder = os.path.dirname(os.path.abspath(cehrgpt.__file__))
    description: Dict[str, Any] = {"cehrgpt_folder": code_folder}
    try:
        description["cehrgpt_installed_version"] = metadata.version("cehrgpt")
    except metadata.PackageNotFoundError:
        description["cehrgpt_installed_version"] = None
    try:
        # An installed package inside a checkout (a venv in the repository) is not that checkout
        tracked = subprocess.run(
            ["git", "-C", code_folder, "ls-files", "--error-unmatch", "__init__.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = subprocess.run(
            ["git", "-C", code_folder, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if tracked.returncode == 0 and commit.returncode == 0:
            description["git_commit"] = commit.stdout.strip()
            status = subprocess.run(
                ["git", "-C", code_folder, "status", "--porcelain", "--", "."],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if status.returncode == 0:
                description["git_uncommitted_changes"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "flash-attn", "cehrbert"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    description["versions"] = versions
    return description


def build_run_manifest(args, task_config: Dict[str, Any]) -> Dict[str, Any]:
    """One session of a prediction run: the command, the code and what it read."""
    return {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "command": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "python_path": os.environ.get("PYTHONPATH"),
        "code": describe_code(),
        "model": describe_folder(getattr(args, "model_folder", None)),
        "tokenizer": describe_folder(getattr(args, "tokenizer_folder", None)),
        "tokenized_dataset": describe_tokenized_dataset(
            getattr(args, "tokenized_full_dataset_path", None)
        ),
        "cohort": describe_folder(getattr(args, "cohort_folder", None), recursive=True),
        "dataset": describe_folder(
            getattr(args, "dataset_folder", None), with_hash=False, recursive=True
        ),
        "concept_ancestor": describe_folder(
            getattr(args, "concept_ancestor", None), with_hash=False, recursive=True
        ),
        "task_config_file": describe_folder(getattr(args, "task_config", None)),
        # After --task_config_override, so what the run used rather than what the file says
        "task_config": task_config,
        "arguments": {
            key: value
            for key, value in sorted(vars(args).items())
            if isinstance(value, (str, int, float, bool, type(None), list))
        },
    }


def write_run_manifest(args, task_config: Dict[str, Any], manifest_path: str) -> None:
    """Add this session to the manifest, keeping the sessions of the run it may be resuming."""
    sessions = []
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                sessions = json.load(f).get("sessions", [])
        except (OSError, ValueError, AttributeError):
            sessions = []
    sessions.append(build_run_manifest(args, task_config))
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    temporary_path = f"{manifest_path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump({"sessions": sessions}, f, indent=2, default=str)
    os.replace(temporary_path, manifest_path)
