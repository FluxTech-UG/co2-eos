"""No tracked text file names the private working setup of the machine this repo is worked on.

The repo is public. Its text carries the package, never the names of the hooks,
mods and tools of the private setup it is developed in. Every file git tracks, and
every untracked file it does not ignore (so a new file is read before it is added),
is read; a file with a NUL byte is binary and is not read (as ``git grep -I`` has
it), a tracked path absent from the working tree has no text to read, and this
file, which holds the word list, is exempt by path. The match ignores case. Every
hit is reported as ``path:line``.

Skips when git is not on PATH or the tree is not a git checkout (an sdist).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PRIVATE_MACHINERY = (
    ".fluxtech-outside",
    "dangerouslyDisableSandbox",
    "sandbox-hatch-gate",
    "bash-shape-guard",
    "fluxtech-meta",
    "mirror install-guards",
    "install-checkout",
    "permission-triage",
    "guardrail/inbox",
    "cache-keeper",
    "flux-compute",
)
EXEMPT = ("tests/test_public_text.py",)


def _tracked_files() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    top = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=False)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != REPO:
        pytest.skip(f"{REPO} is not a git checkout's top")
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--cached",
                             "--others", "--exclude-standard"],
                            capture_output=True, check=True)
    return sorted(n for n in listed.stdout.decode("utf-8", "surrogateescape").split("\0") if n)


def test_no_tracked_text_names_the_private_setup():
    hits = []
    for rel in _tracked_files():
        path = REPO / rel
        if rel in EXEMPT or path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            hits.append(f"{rel}: a text file that is not UTF-8 cannot be checked ({exc})")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            named = [word for word in PRIVATE_MACHINERY if word.lower() in lowered]
            if named:
                hits.append(f"{rel}:{number}: names {', '.join(map(repr, named))}")
    assert not hits, ("public text names the private working setup:\n" + "\n".join(hits))
