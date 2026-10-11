# Paper project page

Open `index.html` in a web browser. Keep the adjacent `assets/` directory with
it: the page uses relative paths to the PDF, figures, and downloadable archives.
No build, framework, third-party JavaScript, or external font service is needed.

For a local preview:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8765/`.

## Contents

- `index.html`: page content, responsive styles, accessible example tabs, and
  citation-copy control (with a manual-copy fallback for local files).
- `assets/paper.pdf`: the revised manuscript.
- `assets/fig-main.png`, `assets/fig-frontier.png`: high-resolution exports of
  the figures in the manuscript.
- `assets/code.zip`: experiment scripts, tests, saved results, dependencies,
  and reproduction README.
- `assets/paper-source.zip`: editable manuscript and its supporting files.

The website presents the exact paper title, author, affiliation, workshop
status, abstract, and cited numerical examples. It does not invent an arXiv
identifier, DOI, public repository URL, or main-conference publication.
The citation uses an accepted-workshop note.

## Publishing

This is a static site. Copy the contents of this directory to the root of a
static host such as GitHub Pages, retaining the asset paths. Nothing has been
published by creating this package. Set a canonical URL and public citation URL
only after a real hosting address is chosen. Source changes can be made directly
in `index.html`; figures and downloadable archives should be refreshed if the
paper or code changes.

The layout is inspired by https://video-msg.github.io/. The HTML and CSS are
original, and the scientific figures come from this paper.
