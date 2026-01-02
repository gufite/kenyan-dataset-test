#!/usr/bin/env python3
"""
Convert a Markdown file to PDF using pandoc.
Usage:
  python code/model/md_to_pdf.py path/to/file.md [-o output.pdf]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert(md_path: Path, out_path: Path) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        sys.exit("pandoc not found. Install pandoc (https://pandoc.org/installing.html) and retry.")

    result = subprocess.run(
        [pandoc, str(md_path), "-o", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(f"Conversion failed with exit code {result.returncode}")

    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF using pandoc.")
    parser.add_argument("markdown_path", help="Path to the Markdown file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF path (defaults to same name with .pdf)",
    )
    args = parser.parse_args()

    md_path = Path(args.markdown_path).resolve()
    if not md_path.is_file():
        sys.exit(f"Markdown file not found: {md_path}")

    out_path = Path(args.output).resolve() if args.output else md_path.with_suffix(".pdf")
    convert(md_path, out_path)


if __name__ == "__main__":
    main()
