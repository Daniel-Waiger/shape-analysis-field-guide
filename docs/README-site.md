# The reading site (`docs/`)

The site is a single static page (`index.html` + `style.css` + `app.js`) that fetches the chapters in
`chapters/*.md`, renders them with [marked](https://marked.js.org/) and [KaTeX](https://katex.org/), and
mounts the interactive demos from `demos/`. There is no build step.

## Preview locally

Browsers refuse `fetch()` from `file://` pages, so serve the folder:

```bash
cd docs && python3 -m http.server 8000
```

Then open <http://localhost:8000/>. Routes are hash-based: `#02` opens chapter 02, `#06/modes-of-variation`
jumps to a section, no hash shows the landing page. The arrow keys move between chapters.

The fonts and the two libraries load from Google Fonts and cdnjs (pinned versions: marked 12.0.2,
KaTeX 0.16.11). Offline, the page still renders: chapters fall back to raw markdown and maths stays as
`$…$` source.

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. **Settings → Pages → Build and deployment → Source: "Deploy from a branch"**.
3. Branch: `main` (or your default), folder: **`/docs`**. Save.
4. After a minute the site is at `https://<user>.github.io/<repo>/`.

Note: GitHub Pages for a **private** repository requires GitHub Pro, Team or Enterprise. On a Free plan
either make the repository public or keep previewing locally. The site uses only relative paths, so it
works both at the repo subpath and at a custom domain.

## How demos are embedded in chapters

A chapter embeds a demo with a single `div`, placed right after the related static figure so GitHub
readers see the figure and site readers additionally get the live version:

```markdown
![Modes 1–4 at ±1 SD](../figures/06_modes.png)
*Modes of variation around the mean.*

<div class="demo" data-demo="pca-modes"></div>
```

After rendering a chapter, `app.js` finds every `div.demo[data-demo="NAME"]`, imports
`demos/NAME.js` as an ES module and calls its default export `mount(el, ctx)`:

- `el` is the `div` to render into (the demo replaces its contents with a `.demo-card`);
- `ctx.colors` holds the resolved theme colours (`paper`, `ink`, `teal`, `coral`, `gold`, `slate`, …);
- `ctx.loadJSON(path)` fetches a JSON file relative to `docs/` (for example `data/worms.json`);
- `ctx.reducedMotion` is `true` when the reader prefers reduced motion;
- `ctx.onThemeChange(cb)` calls `cb(colors)` whenever the theme toggles.

`mount` may return a cleanup function; it runs when the reader leaves the chapter. If a demo throws or
its module is missing, the site shows a short note in its place instead of breaking the page.

Demos draw with inline SVG and read colours from the CSS variables (`var(--teal)` and so on), so they
follow the light/dark theme automatically. Shared helpers (SVG builders, pointer drag, sliders, the
preshape maths and the built-in sample shapes) live in `demos/_lib.js`.

| `data-demo` | File | Data it tries first | Fallback |
|---|---|---|---|
| `preshape` | `demos/preshape.js` | `data/worms.json` | built-in bent worm |
| `rotate-align` | `demos/rotate-align.js` | `data/worms.json` | built-in straight/bent worms |
| `geodesic` | `demos/geodesic.js` | `data/worms.json` | built-in straight/bent worms |
| `pca-modes` | `demos/pca-modes.js` | `data/worms.json` (`pca` block) | four illustrative hand-made modes |
| `bspline` | `demos/bspline.js` | `data/spline.json` | six built-in control points |
| `contrastive` | `demos/contrastive.js` | none (toy data) | — |

When a demo runs on built-in shapes it shows a small "built-in sample shapes" badge, so nobody mistakes
the illustrative numbers for results.

### Data file notes

- `worms.json`: `mean` and `examples.{dead,live}` are `[[x, y] × 12]` preshapes. The `pca` block is used as
  `pca.components` (up to four 24-vectors), `pca.sdev`, `pca.explained` and `pca.mean_tangent`. The
  24-vector layout follows `pca.tangent_layout` (`re_then_im` = 12 real parts then 12 imaginary parts;
  anything containing `xy`/`inter` = interleaved); without that field it is guessed. The demo shows
  `exp_map(mean, mean_tangent + Σ wᵢ sᵢ cᵢ)`; `mean_tangent` is ≈ 0 at the Fréchet mean.
- `spline.json`: control points are read from `control_points` (also accepted: `control`, `coefficients`,
  `c`), landmarks from `landmarks` (or `points`), and an optional `closed` flag. An open curve is drawn on
  `t ∈ [0, M − 1]` with symmetric mirrored end coefficients (`c₋₁ = c₁`, `c_M = c_{M−2}`), the same
  convention as `shapekit.bspline` (checked against the exported `curve` samples to 1e-5 px).

## Chapter conventions the site relies on

- `# NN · Title` on the first line, then the "In one breath" blockquote (styled as the lede callout).
- Figures: `![alt](../figures/NAME)` followed by an italic caption line; the site rewrites `../figures/` to
  `figures/`, wraps the pair in `<figure>`, and shows a small note if the file is not generated yet.
- Maths: `$inline$` and `$$ display $$` on their own lines.
- `<details><summary>…</summary>` blocks become "reveal answer" cards (a missing summary gets one).
- Links to other chapters by file name (`[next](03-rotation-and-complex-numbers.md)`) are rewritten to `#03`.
- Every `##` heading appears in the "On this page" list.
