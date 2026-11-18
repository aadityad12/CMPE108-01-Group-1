"""Build a printable sheet of the formulas each writer copies by hand.

Reads prompts/formulas.txt (one LaTeX label per line, in the model's
vocabulary) and writes prompts/sheet.html. Open it in a browser and print.
Every formula is shown with its number, so each handwritten line can be
matched back to its label later.
"""
import html
import json
from pathlib import Path

PER_PAGE = 10
formulas = [line.strip() for line in Path("prompts/formulas.txt").read_text().splitlines() if line.strip()]

pages = []
for start in range(0, len(formulas), PER_PAGE):
    rows = "\n".join(
        f'<tr><td class="n">{i + 1}</td><td class="f" data-tex="{html.escape(tex)}"></td></tr>'
        for i, tex in enumerate(formulas[start:start + PER_PAGE], start)
    )
    pages.append(f'<section><h2>Page {start // PER_PAGE + 1} '
                 f'(formulas {start + 1}-{min(start + PER_PAGE, len(formulas))})</h2>'
                 f'<table>{rows}</table></section>')

Path("prompts/sheet.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Handwriting prompts</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
section {{ page-break-after: always; }}
td {{ padding: 0.6em 1em; font-size: 1.3em; }}
td.n {{ color: #888; width: 3em; }}
</style></head><body>
<h1>Handwriting prompts ({len(formulas)} formulas)</h1>
<p>Write each formula on its own line, in order, with a clear gap between lines.
One page of paper per page below. Write the page number in the top corner.</p>
{''.join(pages)}
<script>
document.querySelectorAll("td.f").forEach(td =>
  katex.render(td.dataset.tex, td, {{throwOnError: false}}));
</script>
<script>window.FORMULAS = {json.dumps(formulas)};</script>
</body></html>
""")
print(f"Wrote prompts/sheet.html with {len(formulas)} formulas on {len(pages)} pages")
