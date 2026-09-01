# Project dashboard

A self-contained, no-build static dashboard for the ConditionalGeneEssentiality
project — a working reference + presentation deck in one page. Pure HTML/CSS/JS,
no dependencies (matches what `trojai2` has: python3, no node/nginx/sudo).

## View it (SSH port-forward — recommended)

On the **server**:
```bash
./dashboard/serve.sh            # serves http://127.0.0.1:8080
# survive logout:  nohup ./dashboard/serve.sh >/tmp/dash.log 2>&1 &   (or run in tmux)
```
On your **laptop** (IU VPN if off-campus):
```bash
ssh -L 8080:localhost:8080 ds85@trojai2.luddy.indiana.edu
# then open http://localhost:8080
```
Nothing is exposed to the campus network — the server binds to 127.0.0.1 and the
page only reaches your browser through the encrypted SSH tunnel.

## Pages
Overview · The key result (8 grouped comparison tables) · Data · Experiment
timeline (scrollable, filterable, deep-linkable expandable cards) · What we've
learned · Future directions · Status & next steps · Limitations & verification ·
References.

## Features
- **Deep links** — `index.html#exp-rcold` opens the timeline *and* expands that
  card; every card has a `#` permalink.
- **Timeline filters** — all / ✓ worked / ✗ didn't / infra; plus expand/collapse-all.
- **Glossary tooltips** — hover dotted terms (e.g. NDCG@5) for a definition.
- **Light/dark**, **presentation mode** (▶ present, then arrow keys), **print/PDF** (⎙).
- **Mobile** — sidebar collapses to a top menu under ~860px.

## How it's built (this is the seed of the reusable skill)
Rendered by a **pure-stdlib Python generator** — no build tools, matches what the
server has (python3 only):
```bash
python dashboard/build.py     # content model -> index.html
```
Content model:
- `content/experiments.json` — 13 per-experiment records (hypothesis · how-tested ·
  explained numbers · analysis · meaning · verification), extracted from the
  decision ledger. **This is the persistent plain-language synthesis** — edit here,
  re-run `build.py`.
- `content/learnings.json` — 8 enriched durable learnings.
- Static pages (Overview / Key result / Directions / Status / Limitations / Refs /
  Data) are authored inline in `build.py`; theme/behaviour in `assets/{style.css,app.js}`.

To **skillify**: swap the content model + the STATIC blocks in `build.py`; the same
template renders a dashboard for any project (ideally driven from that project's
memory files — `progress.md`, `decisions.md`, `SCIENTIFIC_SYNTHESIS.md`).

## Content provenance
Every claim carries a verification badge: **code-verified** (confirmed in source),
**artifact-verified** (reproduced from a committed run file), **web-verified**
(citation checked online), or **ledger** (from the decision log, not re-run). See
the *Limitations & verification* page. NB: headline numbers were computed on the
original 48-org DB; the re-downloaded DB is a newer 62-org release, so they need
re-deriving before they're treated as current.
