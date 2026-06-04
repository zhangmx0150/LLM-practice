from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "C8_RAG_AI_interview_guide.md"
HTML_PATH = ROOT / "docs" / "C8_RAG_AI_interview_guide.html"
PDF_PATH = ROOT / "docs" / "C8_RAG_AI_interview_guide.pdf"


def find_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_html(markdown_text: str) -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
        ],
        output_format="html5",
    )

    css = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}

* {
  box-sizing: border-box;
}

html {
  color: #172033;
  background: #f4f6f8;
  font-family: "Noto Sans SC", "Microsoft YaHei", "DengXian", "SimHei", sans-serif;
  font-size: 15px;
  line-height: 1.72;
}

body {
  margin: 0;
  padding: 28px;
}

.page {
  max-width: 980px;
  margin: 0 auto;
  padding: 44px 54px 58px;
  background: white;
  border: 1px solid #d8dee8;
  box-shadow: 0 10px 34px rgba(20, 35, 65, 0.10);
}

h1, h2, h3, h4 {
  color: #0f2747;
  line-height: 1.34;
  margin: 1.35em 0 0.55em;
  font-weight: 750;
}

h1 {
  margin-top: 0;
  padding-bottom: 14px;
  border-bottom: 3px solid #1f6fb2;
  font-size: 31px;
}

h2 {
  margin-top: 2em;
  padding: 10px 0 8px;
  border-bottom: 1px solid #dbe3ef;
  font-size: 23px;
  page-break-after: avoid;
}

h3 {
  font-size: 18px;
  page-break-after: avoid;
}

h4 {
  font-size: 16px;
}

p {
  margin: 0.58em 0;
}

ul, ol {
  padding-left: 1.5em;
  margin: 0.55em 0 0.85em;
}

li {
  margin: 0.22em 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 18px;
  page-break-inside: avoid;
  font-size: 14px;
}

th, td {
  border: 1px solid #cfd8e5;
  padding: 8px 10px;
  vertical-align: top;
}

th {
  background: #edf4fb;
  color: #102a45;
  font-weight: 700;
}

tr:nth-child(even) td {
  background: #fafcff;
}

code {
  font-family: Consolas, "Cascadia Mono", "Microsoft YaHei", monospace;
  background: #eef3f8;
  color: #0f3f6f;
  padding: 0.1em 0.32em;
  border-radius: 4px;
  font-size: 0.92em;
}

pre {
  margin: 13px 0 18px;
  padding: 13px 15px;
  border: 1px solid #cbd7e6;
  border-left: 4px solid #1f6fb2;
  background: #f7f9fc;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  page-break-inside: avoid;
}

pre code {
  padding: 0;
  background: transparent;
  color: #152235;
}

blockquote {
  margin: 14px 0;
  padding: 10px 16px;
  border-left: 4px solid #5792c8;
  background: #f3f8fd;
}

hr {
  border: 0;
  border-top: 1px solid #d8e0ec;
  margin: 26px 0;
}

strong {
  color: #102a45;
}

.meta {
  margin: -8px 0 24px;
  padding: 13px 16px;
  background: #f1f6fb;
  border: 1px solid #d8e5f2;
  color: #28435f;
}

@media print {
  html, body {
    background: white;
  }
  body {
    padding: 0;
  }
  .page {
    max-width: none;
    margin: 0;
    padding: 0;
    border: 0;
    box-shadow: none;
  }
  h2 {
    break-after: avoid;
  }
  table, pre {
    break-inside: avoid;
  }
}
"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C8 食谱 RAG 系统 AI 开发岗位面试文档</title>
  <style>{css}</style>
</head>
<body>
  <main class="page">
    {body}
  </main>
</body>
</html>
"""


def main() -> int:
    if not MD_PATH.exists():
        print(f"Markdown source not found: {MD_PATH}", file=sys.stderr)
        return 1

    markdown_text = MD_PATH.read_text(encoding="utf-8")
    HTML_PATH.write_text(build_html(markdown_text), encoding="utf-8")

    browser = find_browser()
    if browser is None:
        print("No supported browser found for PDF rendering.", file=sys.stderr)
        return 1

    if PDF_PATH.exists():
        PDF_PATH.unlink()

    with tempfile.TemporaryDirectory(prefix="c8-pdf-") as user_data_dir:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={user_data_dir}",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.as_uri(),
        ]
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        print("PDF rendering finished but output file is missing or empty.", file=sys.stderr)
        return 1

    print(f"HTML: {HTML_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"PDF size: {PDF_PATH.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
