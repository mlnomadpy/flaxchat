"""Render the YAT embedding article as a self-contained static Space page."""

from __future__ import annotations

from pathlib import Path

import mistune


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/blog/YAT_MMBERT_EMBEDDING_V1.md"
DEST = ROOT / "docs/blog/yat-embedding-space"


def render() -> None:
    article = SOURCE.read_text()
    markdown = mistune.create_markdown(escape=True, plugins=["table", "math"])
    body = markdown(article)
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f6f9fa">
  <meta name="description" content="The YAT mmBERT embedding v1 release: YAT attention and gated FFN equations, TPU results, and comparisons with mmBERT and EmbeddingGemma.">
  <meta property="og:type" content="article">
  <meta property="og:title" content="YAT mmBERT Embedding v1">
  <meta property="og:description" content="A multilingual retrieval encoder, its block equations, measured gains, and open limitations.">
  <title>YAT mmBERT Embedding v1 — Model notes</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    :root { color-scheme: light; --paper:#f6f9fa; --ink:#172a34; --soft:#4c626d; --line:#d4e2e6; --accent:#176d80; --panel:#e9f3f5; --white:#fff; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans',system-ui,sans-serif; font-size:18px; line-height:1.65; }
    a { color:#075d76; text-decoration-thickness:1px; text-underline-offset:3px; }
    a:hover { color:#063f52; }
    a:focus-visible { outline:3px solid #d36d3f; outline-offset:3px; }
    .masthead { border-bottom:1px solid var(--line); background:var(--white); }
    .masthead-inner { max-width:1150px; margin:auto; padding:18px 32px; display:flex; align-items:center; justify-content:space-between; gap:20px; }
    .brand { display:flex; align-items:center; gap:12px; color:var(--ink); text-decoration:none; font-size:15px; font-weight:700; letter-spacing:.01em; }
    .mark { width:30px; height:30px; border:2px solid var(--accent); border-radius:50%; display:grid; place-items:center; color:var(--accent); font-family:'IBM Plex Mono',monospace; font-size:18px; line-height:1; }
    .masthead nav { display:flex; gap:20px; font-size:14px; }
    .masthead nav a { text-decoration:none; font-weight:600; }
    .layout { max-width:1150px; margin:auto; padding:0 32px 110px; display:grid; grid-template-columns:170px minmax(0,780px); gap:72px; }
    .rail { padding-top:80px; }
    .rail-inner { position:sticky; top:30px; border-left:2px solid var(--line); padding-left:17px; font-size:13px; color:var(--soft); }
    .rail-inner strong { display:block; color:var(--ink); margin-bottom:12px; }
    .rail-inner a { display:block; color:var(--soft); text-decoration:none; margin:8px 0; line-height:1.3; }
    .rail-inner a:hover { color:var(--accent); }
    article { min-width:0; padding-top:60px; }
    article h1 { font-size:clamp(2.5rem,5vw,4.35rem); line-height:1.08; letter-spacing:-.055em; max-width:14ch; margin:0 0 22px; font-weight:600; }
    article h1 + p { color:var(--accent); font-size:15px; font-family:'IBM Plex Mono',monospace; margin:0 0 35px; }
    article h2 { margin:68px 0 19px; padding-top:14px; border-top:1px solid var(--line); font-size:clamp(1.7rem,3vw,2.25rem); line-height:1.2; letter-spacing:-.035em; }
    article p { max-width:70ch; margin:0 0 22px; }
    article h1 + p + p { font-size:21px; line-height:1.58; color:#273d47; }
    article strong { font-weight:700; }
    .math { margin:26px 0 30px; background:var(--panel); border-left:4px solid var(--accent); border-radius:0 10px 10px 0; padding:24px 20px; overflow-x:auto; color:#123944; font-size:1.06rem; }
    span.math { margin:0; padding:0; background:none; border:0; color:inherit; }
    .math mjx-container { max-width:100%; }
    table { display:block; width:100%; overflow-x:auto; border-collapse:collapse; font-size:15px; margin:26px 0 32px; }
    th, td { padding:11px 14px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
    th { background:var(--panel); color:#234650; font-weight:600; }
    tbody tr:nth-child(even) { background:#eff5f6; }
    td:not(:first-child), th:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
    code { font-family:'IBM Plex Mono',monospace; font-size:.83em; background:#e8eef0; border-radius:3px; padding:.12em .3em; }
    .footer { border-top:1px solid var(--line); padding:28px 32px 40px; color:var(--soft); font-size:14px; }
    .footer-inner { max-width:1150px; margin:auto; }
    @media (max-width:850px) { .layout { display:block; max-width:800px; padding:0 24px 80px; } .rail { display:none; } article { padding-top:48px; } }
    @media (max-width:580px) { body { font-size:16px; } .masthead-inner { padding:15px 18px; } .masthead nav { gap:12px; font-size:12px; } .layout { padding:0 18px 70px; } article h1 + p + p { font-size:18px; } article h2 { margin-top:52px; } .math { font-size:.86rem; padding:18px 12px; } }
    @media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
  </style>
  <script>
    window.MathJax = { tex: { inlineMath: [[String.raw`\\(`, String.raw`\\)`]], displayMath: [['$$','$$']] }, svg: { fontCache: 'global' } };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <header class="masthead"><div class="masthead-inner">
    <a class="brand" href="https://huggingface.co/mlnomad"><span class="mark">Y</span> YAT model notes</a>
    <nav aria-label="Model links"><a href="https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1">JAX weights</a><a href="https://huggingface.co/mlnomad/yat-mmbert-base-embedding-v1-pytorch">PyTorch weights</a></nav>
  </div></header>
  <div class="layout">
    <aside class="rail" aria-label="Article contents"><div class="rail-inner">
      <strong>On this page</strong>
      <a href="#inside-the-yat-blocks">Inside the YAT blocks</a>
      <a href="#what-improved">What improved</a>
      <a href="#where-it-stands-against-mmbert-and-embeddinggemma">Against mmBERT and EmbeddingGemma</a>
      <a href="#using-the-release">Using the release</a>
      <a href="#what-we-will-improve-next">What comes next</a>
    </div></aside>
    <article id="article">{{ARTICLE}}</article>
  </div>
  <footer class="footer"><div class="footer-inner">YAT mmBERT Embedding v1 · Release notes and measured results · September 2026</div></footer>
</body>
</html>
"""
    # Mistune does not assign heading IDs; link the rail to the actual sections.
    for title, slug in (
        ("Inside the YAT blocks", "inside-the-yat-blocks"),
        ("What improved", "what-improved"),
        ("Where it stands against mmBERT and EmbeddingGemma", "where-it-stands-against-mmbert-and-embeddinggemma"),
        ("Using the release", "using-the-release"),
        ("What we will improve next", "what-we-will-improve-next"),
    ):
        heading = f"<h2>{title}</h2>"
        if body.count(heading) != 1:
            raise ValueError(f"Missing or duplicated heading: {title}")
        body = body.replace(heading, f'<h2 id="{slug}">{title}</h2>')
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "index.html").write_text(page.replace("{{ARTICLE}}", body))
    (DEST / "article.md").write_text(article)


if __name__ == "__main__":
    render()
