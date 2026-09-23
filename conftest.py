"""Pytest configuration: make the repository root importable regardless of
where pytest is invoked from (repo root, tests/, another directory, or a CI
container). Without this, `import src...` / `import app...` inside the test
suite only works when the current working directory happens to be the repo
root -- a fragile, environment-dependent assumption.

Also register shared markers (none used yet beyond future-proofing).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
