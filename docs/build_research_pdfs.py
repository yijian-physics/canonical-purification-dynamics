#!/usr/bin/env python3
"""Typeset the two research reports from their authoritative Markdown sources.

Requires Python 3, Pandoc >= 3, and XeLaTeX with the standard TeX Live packages.
Run from any directory: python docs/build_research_pdfs.py
Build intermediates and logs go to run_outputs/research_pdfs (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "run_outputs" / "research_pdfs"
REPO_URL = "https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/blob/research/factor-tdvp-benchmarks/"
REPORTS = [
    ("factor_tdvp/report.md", "factor_tdvp/factor_tdvp_note.pdf"),
    ("factor_tdvp_comparison/report.md", "factor_tdvp_comparison/comparison_report.pdf"),
]


def run(args, *, env=None, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Command failed: " + " ".join(map(str, args)) + "\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr


def markdown_for_pdf(text, source):
    # Github's code-backed inline math avoids Markdown emphasis/backslash parsing.
    text = re.sub(r"\$`([^`\n]+)`\$", lambda m: "$" + m[1] + "$", text)
    # GitHub keeps equation numbers outside the MathJax block. In the PDF,
    # restore ordinary right-aligned LaTeX tags without duplicating the label.
    text = re.sub(
        r"(?m)^\*\*Equation \((\d+)\)\.\*\*\s*\n```math\s*\n(.*?)^```\s*$",
        lambda m: "$$\n" + m[2].rstrip() + "\n\\tag{" + m[1] + "}\n$$",
        text, flags=re.S,
    )
    text = re.sub(r"(?m)^```math\s*\n(.*?)^```\s*$", lambda m: "$$\n" + m[1].rstrip() + "\n$$", text, flags=re.S)
    # Keep relative repository links useful in a downloaded PDF. Images stay local.
    def link(match):
        prefix, label, dest = match.groups()
        if prefix == "!" or re.match(r"(?:[a-z]+:|#)", dest):
            return match[0]
        path = (source.parent / dest).resolve().relative_to(ROOT)
        return "[" + label + "](" + REPO_URL + urllib.parse.quote(path.as_posix()) + ")"
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^ )]+)\)", link, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", choices=["factor_tdvp", "factor_tdvp_comparison"], help="Build only this report")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # This fallback is only for minimal containers whose TeX source tree exists
    # but whose system ls-R database and prebuilt format were not installed.
    probe = subprocess.run(["kpsewhich", "article.cls"], text=True, capture_output=True)
    if not probe.stdout.strip() and Path("/usr/share/texlive/texmf-dist").exists():
        env["TEXMF"] = "{/usr/share/texmf,/usr/share/texlive/texmf-dist}"
        env["TEXFORMATS"] = str(BUILD / "tex-format") + ":"
        fmt_dir = BUILD / "tex-format"
        fmt_dir.mkdir(exist_ok=True)
        if not (fmt_dir / "xelatex.fmt").exists():
            run(["xetex", "-ini", "-etex", "-progname=xelatex", "-output-directory=" + str(fmt_dir), "xelatex.ini"], env=env, cwd=fmt_dir)
    manifest = []
    for rel_source, rel_output in REPORTS:
        source = ROOT / rel_source
        name = source.parent.name
        if args.report and name != args.report:
            continue
        outdir = BUILD / name
        outdir.mkdir(exist_ok=True)
        md = outdir / "report.md"
        tex = outdir / "report.tex"
        md.write_text(markdown_for_pdf(source.read_text(), source))
        command = [
            "pandoc", str(md), "--from=markdown+tex_math_dollars", "--to=latex", "--standalone",
            "--resource-path=" + str(source.parent), "--output=" + str(tex),
            "--include-in-header=" + str(ROOT / "docs/research_pdf_header.tex"),
            "--lua-filter=" + str(ROOT / "docs/research_pdf.lua"),
            "--no-highlight", "--top-level-division=section",
            "-V", "documentclass=article", "-V", "papersize=a4", "-V", "fontsize=11pt",
            "-V", "geometry:margin=19mm", "-V", "colorlinks=true", "-V", "linkcolor=NavyBlue",
            "-V", "urlcolor=NavyBlue", "-V", "mainfont=lmroman10-regular.otf",
            "-V", "mainfontoptions=BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf,BoldItalicFont=lmroman10-bolditalic.otf",
            "-V", "mathfont=latinmodern-math.otf", "-V", "monofont=DejaVu Sans Mono",
            "-V", "monofontoptions=Scale=0.78", "-V", "linestretch=1.04",
        ]
        run(command, env=env)
        # XeLaTeX must resolve figures relative to the original report directory.
        for _ in range(2):
            log = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + str(outdir), str(tex)], env=env, cwd=source.parent)
        (outdir / "build.log").write_text(log)
        warnings = [line for line in log.splitlines() if re.search(r"(?:Overfull|Missing character|LaTeX Warning)", line)]
        shutil.copyfile(outdir / "report.pdf", ROOT / rel_output)
        manifest.append({"source": rel_source, "pdf": rel_output, "warnings": warnings})
        print(json.dumps(manifest[-1]))
    (BUILD / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
