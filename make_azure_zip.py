"""Builds analytixa-azure.zip next to this folder, ready for: az webapp deploy --type zip.

It leaves out everything that must not or need not go to Azure: config.env (your passwords and keys), logs,
tests, documents, caches and Vercel files. Azure installs the packages from requirements.txt when it receives the zip.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "analytixa-azure.zip"
SKIP_DIRS = {"logs", "tests", "docs", "__pycache__", ".git", ".vercel", "data", "uploads", "antenv", ".venv", "venv"}
SKIP_FILES = {"config.env", ".vercelignore", "vercel.json", "start.bat", "start.sh", "make_azure_zip.py"}


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir() or set(rel.parts[:-1]) & SKIP_DIRS or rel.parts[0] in SKIP_DIRS:
                continue
            if p.name in SKIP_FILES or p.suffix in (".pyc", ".db", ".log"):
                continue
            z.write(p, rel.as_posix())
            n += 1
    names = zipfile.ZipFile(OUT).namelist()
    assert "config.env" not in names and not any(x.startswith("logs/") for x in names)
    print(f"Wrote {OUT} with {n} files. config.env, logs and tests are not in it.")


if __name__ == "__main__":
    main()
