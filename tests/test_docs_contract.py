"""The documentation contract.

A documentation system with no test rots. This runs at the end of every session
(see the Wrap-Up Protocol in CLAUDE.md) and fails loudly when the docs and the
repository disagree.

Every failure message says what to do. The standing instruction is:
    the docs and the repo disagree. Fix the docs or fix the code --
    do not quote a number until this passes.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

CLAUDE_MD = REPO / "CLAUDE.md"
PLAN, TERMS, DATA, MEMORY, BUGS = (
    DOCS / n for n in ("plan.md", "terms.md", "data.md", "memory.md", "bugs.md")
)
THE_SIX = [CLAUDE_MD, PLAN, TERMS, DATA, MEMORY, BUGS]

CLAUDE_MD_LINE_BUDGET = 160

# Directories excluded from every scan: duplicate doc trees, superseded docs, VCS.
EXCLUDED_DIR_NAMES = {".git", ".claude", "archive_docs", "__pycache__",
                      ".pytest_cache", ".ipynb_checkpoints", "node_modules"}

FIX = "the docs and the repo disagree. Fix the docs or fix the code -- do not quote a number until this passes."


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The board is not stale
# ---------------------------------------------------------------------------

def test_plan_last_updated_is_not_older_than_the_newest_memory_entry():
    """Catches "work was done and the board was never reset"."""
    m = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", _read(PLAN))
    assert m, f"plan.md has no '**Last updated:** YYYY-MM-DD' stamp. {FIX}"
    plan_date = date.fromisoformat(m.group(1))

    entry_dates = [
        date.fromisoformat(d)
        for d in re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})\s", _read(MEMORY), re.M)
    ]
    assert entry_dates, f"memory.md has no dated '## YYYY-MM-DD' entries. {FIX}"
    newest = max(entry_dates)

    assert plan_date >= newest, (
        f"plan.md Last updated ({plan_date}) predates the newest memory.md entry "
        f"({newest}). Work was archived but the board was never reset. "
        f"Run step 6 of the Wrap-Up Protocol: rewrite plan.md Current State, add the "
        f"ledger row, bump Last updated."
    )


# ---------------------------------------------------------------------------
# 2-3. CLAUDE.md stays a contract, not a results file
# ---------------------------------------------------------------------------

def test_claude_md_is_within_its_line_budget():
    n = len(_read(CLAUDE_MD).splitlines())
    assert n <= CLAUDE_MD_LINE_BUDGET, (
        f"CLAUDE.md is {n} lines, over the {CLAUDE_MD_LINE_BUDGET}-line budget. "
        f"It costs context on every single turn. Move detail to docs/plan.md "
        f"(state), docs/terms.md (definitions) or docs/data.md (paths)."
    )


# Metric-ish words that must never appear next to a figure in the contract.
_RESULT_WORDS = r"(ndcg|spearman|kendall|precision_at|auroc|aupr|ceiling|noise.?floor|delta|coverage)"
_NUMBER = r"\d*\.\d+"


def test_claude_md_contains_no_results():
    """The contract must stay findings-free or it becomes a stale results file."""
    offenders = []
    for i, line in enumerate(_read(CLAUDE_MD).splitlines(), 1):
        low = line.lower()
        for m in re.finditer(_NUMBER, low):
            window = low[max(0, m.start() - 60): m.end() + 60]
            if re.search(_RESULT_WORDS, window):
                offenders.append(f"  CLAUDE.md:{i}: {line.strip()}")
                break
    assert not offenders, (
        "CLAUDE.md contains result-like figures:\n" + "\n".join(offenders) +
        "\nThe contract holds ZERO findings, results and numbers. Move the value to "
        "docs/terms.md (as a definition) or docs/plan.md (as current state), and refer "
        "to it by name from CLAUDE.md."
    )


# ---------------------------------------------------------------------------
# 4. Every path the docs reference actually exists
# ---------------------------------------------------------------------------

# A row explicitly marked as absent/planned is not required to exist -- the doc is
# being honest about the gap, which is the point.
_ABSENT_MARKERS = ("MISSING", "PLANNED", "DEPRECATED", "Removed", "removed", "was a symlink",
                   "existed in no commit", "no longer")

_PATHY = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./<>*-]*\.[A-Za-z0-9_]+|[a-z_]+/[A-Za-z0-9_./-]*)`")


def _candidate_paths(text: str):
    for line in text.splitlines():
        if any(mark in line for mark in _ABSENT_MARKERS):
            continue
        for raw in _PATHY.findall(line):
            # skip globs, placeholders and bare module dotted-names
            if any(ch in raw for ch in "<>*") or raw.endswith("/"):
                continue
            if "..." in raw:
                continue
            # A bare basename in prose ("re-pin `reval_baseline.json`") is a reference,
            # not a path claim. Only strings with a directory component are checked.
            if "/" not in raw:
                continue
            if ":" in raw:
                raw = raw.split(":", 1)[0]
            yield line, raw


def test_every_referenced_path_exists():
    missing = []
    for doc in THE_SIX:
        for line, raw in _candidate_paths(_read(doc)):
            p = REPO / raw
            if p.exists() or p.is_symlink():
                continue
            missing.append(f"  {doc.relative_to(REPO)}: `{raw}`  <- in: {line.strip()[:100]}")
    assert not missing, (
        "Documented paths do not exist:\n" + "\n".join(sorted(set(missing))) +
        f"\n{FIX} If the path is genuinely gone, mark its row MISSING in docs/data.md "
        "rather than deleting the row."
    )


# ---------------------------------------------------------------------------
# 5. Every directory that exists is registered
# ---------------------------------------------------------------------------

_REGISTERED_ROOTS = ("artifacts", "data_contract", "research_log", "docs", "paper")


def test_every_directory_is_registered_in_data_md():
    data_txt = _read(DATA)
    unregistered = []
    for root in _REGISTERED_ROOTS:
        root_dir = REPO / root
        if not root_dir.is_dir():
            continue
        for child in sorted(root_dir.iterdir()):
            if not child.is_dir() or _excluded(child):
                continue
            rel = f"{root}/{child.name}"
            # runs/ holds timestamped per-invocation dirs, covered by the layout rule
            if root == "artifacts" and child.name == "runs":
                if "artifacts/runs" in data_txt:
                    continue
            if rel in data_txt or child.name in data_txt:
                continue
            unregistered.append(f"  {rel}")
    assert not unregistered, (
        "Directories exist but are not registered in docs/data.md:\n" +
        "\n".join(unregistered) +
        "\nRun step 5 of the Wrap-Up Protocol: every new output, dataset, checkpoint or "
        "directory gets a docs/data.md row with its state."
    )


# ---------------------------------------------------------------------------
# 6. Terms used in reports resolve to a glossary entry
# ---------------------------------------------------------------------------

def _terms_defined() -> set[str]:
    return set(re.findall(r"^###\s+([a-z0-9_]+)\s", _read(TERMS), re.M))


def test_report_row_and_column_labels_resolve_to_terms():
    """Row labels are the exact terms.md identifier. No prose synonyms."""
    defined = _terms_defined()
    used: set[str] = set()
    for line in _read(PLAN).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        for cell in cells:
            for ident in re.findall(r"`([a-z][a-z0-9_]{2,})`", cell):
                used.add(ident)
    unresolved = sorted(u for u in used if u not in defined)
    assert not unresolved, (
        "Report labels in docs/plan.md do not resolve to a docs/terms.md entry:\n" +
        "\n".join(f"  {u}" for u in unresolved) +
        "\nRun step 4 of the Wrap-Up Protocol: any new metric, concept or pipeline step "
        "gets a full six-field terms.md entry. Do not use a prose synonym in a table."
    )


# ---------------------------------------------------------------------------
# 7. The board agrees with the pinned regression gate
# ---------------------------------------------------------------------------

def test_headline_table_matches_the_pinned_baseline():
    """A number in the board that drifts from reval_baseline.json is the highest-signal
    docs failure there is: it means a quoted result no longer matches the artifact."""
    baseline_p = REPO / "data_contract" / "ranking" / "reval_baseline.json"
    full = json.loads(_read(baseline_p))["full"]
    plan_txt = _read(PLAN)

    mismatches = []
    for method in ("chem_null", "model", "linear_mf", "chem_knn"):
        for metric, key in (("ndcg_at_5", "ndcg_at_5"),
                            ("within_gene_spearman_mean", "spearman"),
                            ("within_gene_kendall_mean", "kendall"),
                            ("precision_at_5", "precision_at_5")):
            expected = f"{full[method][key]:.4f}"
            if expected not in plan_txt:
                mismatches.append(f"  {method}.{metric} = {expected} (from reval_baseline.json) "
                                  f"is not present in plan.md's headline table")
    assert not mismatches, (
        "docs/plan.md disagrees with the pinned regression gate:\n" + "\n".join(mismatches) +
        f"\n{FIX}"
    )


# ---------------------------------------------------------------------------
# 8. Corrections do not survive elsewhere (the known failure mode)
# ---------------------------------------------------------------------------

# Values that are REAL and current, but mean the wrong thing in the wrong context.
# These are not retractions -- the number is legitimate -- so the correction probe
# cannot see them. Each entry says where the value must NOT appear without an
# explicit annotation. This is the harder failure mode: an ambiguous number is more
# dangerous than a retracted one, because it survives every "is this still true?" check.
CONTESTED_VALUES = {
    "0.3214": dict(
        forbidden_near=r"(ceiling|noise.?floor)",
        needs=r"(median|unfiltered|DIAGNOSTIC|diagnostic|not comparable|all val genes)",
        why="0.3214 is the MEDIAN over ALL val genes. Quoted as 'the ceiling' next to a "
            "model score it is an apples-to-oranges comparison on two axes at once. The "
            "parity-comparable ceiling is 0.393 (MEAN over parity-eligible val genes).",
    ),
    "0.358": dict(
        forbidden_near=r"(ceiling|noise.?floor)",
        needs=r"(8 high|high.?rep|subset|DIAGNOSTIC|diagnostic)",
        why="0.358 is the 8-high-replicate-organism subset, not the 23-organism panel.",
    ),
}

# A mention is fine when it is explicitly flagged as historical.
_ANNOTATION = re.compile(
    r"(INCORRECT|CORRECTION|superseded|SUPERSEDES|stale|STALE|RETIRED|DIAGNOSTIC|DEMOTED|"
    r"historical|no longer|was the|formerly|deprecated|DEPRECATED)", re.I)


def _retracted_probes() -> list[str]:
    """Distinctive tokens from retracted lines.

    Two rules learned the hard way:
      * A number needs >=4 decimals to be distinctive. Probing "0.023" matches
        "0.0238095..." and every unrelated value that happens to share three digits.
      * A number the paired [CORRECTION] restates is NOT retracted -- what was wrong
        was the interpretation, not the value. Probing it only generates noise.
    """
    lines = _read(MEMORY).splitlines()
    # Anything stated anywhere in the ledger OUTSIDE an [INCORRECT] line is a live
    # fact, not a retraction. A retracted line routinely contains still-valid numbers
    # -- what was retracted was the interpretation. Probing those generates pure noise.
    elsewhere = " ".join(l for l in lines if not l.startswith("[INCORRECT]"))
    probes: set[str] = set()
    for line in lines:
        if not line.startswith("[INCORRECT]"):
            continue
        for num in re.findall(r"\d+\.\d{4,}", line):
            if num not in elsewhere:
                probes.add(num)
        # Genuine snake_case identifiers are distinctive; plain English words are not.
        # The underscore requirement is what separates "delta_concrete_spearman" from
        # "essentiality" and "confirmatory".
        for ident in re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}", line):
            if ident not in elsewhere:
                probes.add(ident)
    return sorted(probes)


def test_corrections_do_not_survive_elsewhere():
    """Grep the refuted NUMBER, not the topic. Numbers are the high-signal probe."""
    numbers = _retracted_probes()
    assert numbers, ("memory.md has no [INCORRECT] lines carrying a distinctive probe. If "
                     "nothing has ever been retracted, delete this assertion -- otherwise "
                     "the correction rule is not being applied.")

    scan_roots = [REPO / "src", REPO / "scripts", REPO / "dashboard", REPO / "data_contract",
                  CLAUDE_MD, PLAN, TERMS, DATA, BUGS]
    survivors = []
    for root in scan_roots:
        files = [root] if root.is_file() else [
            f for f in root.rglob("*")
            if f.is_file() and not _excluded(f)
            and f.suffix in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}
        ] if root.exists() else []
        for f in files:
            rel = str(f.relative_to(REPO))
            try:
                lines = _read(f).splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                for num in numbers:
                    if num not in line:
                        continue
                    # An annotation on an adjacent line counts -- a YAML key is
                    # routinely explained by the comment directly above it.
                    ctx = " ".join(lines[max(0, i - 4): i + 3])
                    if _ANNOTATION.search(ctx):
                        continue
                    survivors.append(f"  {rel}:{i}: {line.strip()[:110]}")
    assert not survivors, (
        "Retracted values are still asserted as fact outside memory.md:\n" +
        "\n".join(sorted(set(survivors))) +
        "\nA correction was recorded but never propagated. Either update the line, or "
        "annotate it as historical (superseded / STALE / DIAGNOSTIC / RETIRED)."
    )


def test_contested_values_are_never_quoted_bare():
    """The harder failure mode: a number that is real, but wrong in this context."""
    scan_roots = [REPO / "src", REPO / "scripts", REPO / "dashboard",
                  REPO / "data_contract", REPO / "paper",
                  CLAUDE_MD, PLAN, TERMS, DATA, BUGS, MEMORY]
    offenders = []
    for root in scan_roots:
        files = [root] if root.is_file() else [
            f for f in root.rglob("*")
            if f.is_file() and not _excluded(f)
            and f.suffix in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}
        ] if root.exists() else []
        for f in files:
            try:
                lines = _read(f).splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                for value, rule in CONTESTED_VALUES.items():
                    if value not in line:
                        continue
                    # look at the line plus its two neighbours for the annotation
                    ctx = " ".join(lines[max(0, i - 5): i + 4])
                    if not re.search(rule["forbidden_near"], ctx, re.I):
                        continue
                    if re.search(rule["needs"], ctx, re.I):
                        continue
                    offenders.append(
                        f"  {f.relative_to(REPO)}:{i}: {line.strip()[:100]}\n"
                        f"      -> {rule['why']}")
    assert not offenders, (
        "Contested values quoted without the annotation that makes them safe:\n" +
        "\n".join(sorted(set(offenders))))


# ---------------------------------------------------------------------------
# 9-12. Naming and filesystem hygiene
# ---------------------------------------------------------------------------

def test_no_two_paths_differ_only_in_case():
    seen: dict[str, str] = {}
    collisions = []
    for p in REPO.rglob("*"):
        if _excluded(p):
            continue
        rel = str(p.relative_to(REPO))
        low = rel.lower()
        if low in seen and seen[low] != rel:
            collisions.append(f"  {seen[low]}  vs  {rel}")
        seen[low] = rel
    assert not collisions, (
        "Paths differing only in case:\n" + "\n".join(collisions) +
        "\nRename one. Case-only differences break on case-insensitive filesystems."
    )


def test_no_loose_files_at_a_results_root():
    """Everything belongs in an owning directory, or it becomes unattributable."""
    offenders = []
    runs = REPO / "artifacts" / "runs"
    if runs.is_dir():
        for f in runs.iterdir():
            if f.is_file() and f.name != ".gitkeep" and not f.name.startswith("DEPRECATED_"):
                offenders.append(f"  artifacts/runs/{f.name}")
    for f in REPO.iterdir():
        if f.is_file() and f.suffix in {".parquet", ".csv", ".npz", ".pt"} \
                and not f.name.startswith("DEPRECATED_"):
            offenders.append(f"  {f.name}")
    assert not offenders, (
        "Loose files at a results root:\n" + "\n".join(offenders) +
        "\nMove each into an owning run directory, or rename it DEPRECATED_* and register "
        "it in docs/data.md."
    )


def test_no_brace_shorthand_in_docs():
    """Shorthand is not greppable and a verifier cannot check it."""
    offenders = []
    for doc in THE_SIX:
        for i, line in enumerate(_read(doc).splitlines(), 1):
            for m in re.finditer(r"`[^`]*\{[^`}]*,[^`}]*\}[^`]*`", line):
                offenders.append(f"  {doc.relative_to(REPO)}:{i}: {m.group(0)}")
    assert not offenders, (
        "Brace/glob shorthand in docs:\n" + "\n".join(offenders) +
        "\nWrite the paths out in full (run_a/, run_b/) so they are greppable and "
        "checkable."
    )


def test_no_self_referential_symlinks():
    """A symlink pointing at its own ancestor makes every recursive walk infinite."""
    offenders = []
    for p in REPO.rglob("*"):
        if _excluded(p) or not p.is_symlink():
            continue
        try:
            target = p.resolve()
        except (OSError, RuntimeError):
            offenders.append(f"  {p.relative_to(REPO)} -> <unresolvable>")
            continue
        if target == p.parent.resolve() or target in p.parent.resolve().parents \
                or p.parent.resolve() == target or str(p.parent.resolve()).startswith(str(target) + "/"):
            offenders.append(f"  {p.relative_to(REPO)} -> {target}")
    assert not offenders, (
        "Self-referential symlinks (infinite recursion in any tree walk):\n" +
        "\n".join(offenders) + "\nRemove the link. See docs/bugs.md."
    )


def test_the_six_files_exist_and_nothing_else_is_a_working_doc():
    for f in THE_SIX:
        assert f.exists(), f"Missing documentation file: {f.relative_to(REPO)}"
    strays = [
        p.relative_to(REPO) for p in DOCS.rglob("*.md")
        if p.resolve() not in {f.resolve() for f in THE_SIX}
    ]
    assert not strays, (
        f"Extra documentation files in docs/: {strays}. The system is exactly six files. "
        "Content belongs in one of them, in paper/, or in archive_docs/."
    )
