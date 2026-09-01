#!/usr/bin/env python3
"""Render the project dashboard (index.html) from the content model.

Content model:
  dashboard/content/experiments.json   # per-experiment extracts (from the ledger)
  dashboard/content/learnings.json     # enriched durable learnings
  + static page content authored below (home / key-result tables / directions /
    status / limitations / references / data).

This is deliberately dependency-free (pure stdlib) so it runs anywhere python does,
and is the seed of a reusable "project-dashboard" skill: swap the content model +
the STATIC blocks and the same template/renderer produces a dashboard for any project.

Usage:  python dashboard/build.py   ->   writes dashboard/index.html
"""
from __future__ import annotations
import html, json, re, pathlib

ROOT = pathlib.Path(__file__).parent
EXPERIMENTS = json.load(open(ROOT / "content" / "experiments.json"))
LEARNINGS = json.load(open(ROOT / "content" / "learnings.json")).get("learnings", [])

# ----------------------------------------------------------------------------- helpers
def esc(s: str) -> str:
    return html.escape(s or "", quote=False)

def md_lite(text: str, paras: bool = True) -> str:
    """Escape, then render `code` spans and **bold**; split blank lines into <p>."""
    if not text:
        return ""
    def inline(t: str) -> str:
        t = esc(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return t
    if not paras:
        return inline(text)
    blocks = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{inline(b.strip())}</p>" for b in blocks if b.strip())

# ----------------------------------------------------------------------------- experiment display metadata
# order, short display tag, family, verdict, one-line reasoning shown in the summary
META = {
  "cross_org":  dict(n=1, tag="T-regime", family="cross_organism", verdict="neg",
                     short="Predict held-out organisms' gene×condition fitness from embedding+chemistry."),
  "rlock_setup":dict(n=2, tag="R-LOCK", family="infrastructure", verdict="neu",
                     short="How every within-organism experiment is tested: the split, eligibility, baselines, and the gate."),
  "r1_encoder": dict(n=3, tag="R1", family="within_org_warm", verdict="neg",
                     short="Is the chemistry representation the bottleneck? Richer fingerprints should beat multihot."),
  "capacity":   dict(n=4, tag="capacity", family="within_org_warm", verdict="neg",
                     short="Is the model underpowered? More capacity / nonlinearity should help."),
  "rloss":      dict(n=5, tag="R-LOSS", family="within_org_warm", verdict="neg",
                     short="Does a ranking-aware loss (matching the metric) beat pointwise and the lookup?"),
  "rtopk":      dict(n=6, tag="R-TOPK", family="within_org_warm", verdict="neg",
                     short="Does truncating the loss to the top-5 (exactly the metric) close the gap?"),
  "rhybrid_a":  dict(n=7, tag="R-HYBRID-A", family="within_org_warm", verdict="neu",
                     short="Do the model and kNN carry complementary signal a static blend can exploit?"),
  "rhybrid_b":  dict(n=8, tag="R-HYBRID-B", family="within_org_warm", verdict="neg",
                     short="Does a LEARNED fusion (residual / retrieval / gating) beat the static blend?"),
  "raug":       dict(n=9, tag="R-AUG", family="within_org_warm", verdict="neg",
                     short="Does training on all 48 organisms narrow the gap?"),
  "rconf":      dict(n=10, tag="R-CONF", family="within_org_warm", verdict="neg",
                     short="Is the gap just label noise? On the cleanest-measured genes the model should win."),
  "rcold":      dict(n=11, tag="R-COLD", family="cold_gene", verdict="pos",
                     short="Hold out WHOLE genes so the lookup can't apply — can the model beat the population null?"),
  "rdark":      dict(n=12, tag="R-DARK", family="discovery", verdict="neg",
                     short="Are unannotated 'dark' genes enriched for surprising, chemically-specific essentiality?"),
  "literature": dict(n=13, tag="lit", family="literature", verdict="neu",
                     short="Is 'local memorization beats global learning' our failure, or a known phenomenon?"),
}
FAMILY_LABEL = {
  "cross_organism":"cross-organism", "within_org_warm":"within-org (warm)",
  "cold_gene":"cold-gene", "infrastructure":"infrastructure",
  "discovery":"discovery", "literature":"literature",
}
VERDICT_LABEL = {"neg":"Negative", "pos":"Positive", "neu":"Neutral / infra"}
VBADGE = {
  "code":'<span class="badge b-code">code-verified</span>',
  "artifact":'<span class="badge b-art">artifact-verified</span>',
  "web":'<span class="badge b-web">web-verified</span>',
  "ledger":'<span class="badge b-ledger">ledger</span>',
  "mixed":'<span class="badge b-art">artifact</span> <span class="badge b-code">code</span>',
}

def render_results_table(results):
    rows = []
    for r in results:
        rows.append(
          "<tr><td>%s</td><td class='num'>%s</td><td class='exp'>%s</td></tr>" % (
            esc(r.get("label","")), esc(r.get("number","")), md_lite(r.get("plain_explanation",""), paras=False)))
    return ("<table class='rtbl'><thead><tr><th>Metric</th><th class='num'>Value</th>"
            "<th>What it means</th></tr></thead><tbody>%s</tbody></table>" % "".join(rows))

def render_experiment(e):
    m = META.get(e["id"], dict(tag="", family="within_org_warm", verdict="neu", short=""))
    caveats = e.get("caveats","")
    body = []
    body.append("<h4 class='hyp'>Hypothesis / reasoning</h4>" + md_lite(e.get("hypothesis","")))
    body.append("<h4 class='how'>How we tested this</h4>" + md_lite(e.get("how_tested","")))
    body.append("<h4 class='res'>Results</h4>" + render_results_table(e.get("results",[])))
    body.append("<h4 class='ana'>Analysis — why it came out this way</h4>" + md_lite(e.get("analysis","")))
    body.append("<h4 class='mean'>What it means for the project</h4>" + md_lite(e.get("meaning","")))
    if caveats:
        body.append("<div class='caveat'><b>Caveats.</b> " + md_lite(caveats, paras=False) + "</div>")
    src = e.get("source_files", [])
    if src:
        body.append("<div class='srcs'>Sources: " + " · ".join("<code>%s</code>" % esc(s) for s in src) + "</div>")
    return (
      "<details class='exp' id='exp-%s' data-family='%s' data-verdict='%s'>"
      "<summary><span class='node'></span>"
      "<div class='top'><div><span class='name'>%d · %s</span>"
      "<span class='tag'>%s</span> %s"
      "<a class='permalink' href='#exp-%s' title='link to this experiment' onclick='event.stopPropagation()'>#</a>"
      "<div class='hypline'><b>Hypothesis:</b> %s</div></div>"
      "<span class='chev'>▶</span></div></summary>"
      "<div class='body'>%s</div></details>"
    ) % (
      e["id"], m["family"], m["verdict"], m.get("n",0), esc(e.get("name","")),
      esc(m["tag"]), VBADGE.get(e.get("verification","ledger"), ""), e["id"],
      esc(m["short"]), "".join(body))

def render_learning(l, i):
    return (
      "<div class='panel learn'><h3>%d · %s %s</h3>"
      "<p>%s</p>"
      "<div class='ev'><span class='evk'>Evidence</span> %s</div>"
      "<div class='ev'><span class='evk'>How we know</span> %s</div></div>"
    ) % (i, esc(l.get("title","")), VBADGE.get(l.get("verification","ledger"),""),
         md_lite(l.get("detail",""), paras=False),
         md_lite(l.get("evidence",""), paras=False),
         md_lite(l.get("how_we_know",""), paras=False))

# order experiments
EX_ORDER = sorted(EXPERIMENTS, key=lambda e: META.get(e["id"],{}).get("n",99))

# ----------------------------------------------------------------------------- STATIC page content
def T(rows, head):
    h = "".join("<th class='%s'>%s</th>" % ("num" if c.get("num") else "", c["t"]) for c in head)
    body = []
    for r in rows:
        tds = []
        for c in head:
            v = r.get(c["k"], "")
            cls = "num" if c.get("num") else ""
            if r.get("_hl") and c.get("k")==head[0]["k"]:
                cls += " hl"
            if c.get("win") and r.get("_win"):
                cls += " win"
            tds.append("<td class='%s'>%s</td>" % (cls.strip(), v))
        body.append("<tr>%s</tr>" % "".join(tds))
    return "<table>%s<tbody>%s</tbody></table>" % (
        "<thead><tr>%s</tr></thead>" % h, "".join(body))

# Grouped comparison tables for the Key Result page
TBL_WARM = T([
    dict(m="chem-NULL", a="0.343", b="0.023", w="population profile (gene-blind)"),
    dict(m="deep model (emb ⊕ chemistry)", a="0.432", b="0.152", w="the intended model"),
    dict(m="linear-MF (learned latents)", a="0.429", b="0.143", w="a learned fitness-aware gene rep"),
    dict(m="chem-kNN — the gate", a="0.485", b="0.240", w="the gene's OWN history, chemistry lookup", _hl=1, _win=1),
    dict(m="replicate ceiling", a="0.658", b="0.393", w="biological-replicate agreement (the max)"),
], [dict(t="Method",k="m"),dict(t="NDCG@5",k="a",num=1,win=1),dict(t="Spearman",k="b",num=1),dict(t="what it is",k="w")])

TBL_CROSSORG = T([
    dict(m="embedding-NN baseline (S2 best)", a="0.588", b="—"),
    dict(m="deep model (locked, T4/T5)", a="≈0.499–0.502", b="—"),
    dict(m="within-gene ranking diagnostic", a="—", b="≈0.045", _hl=1),
    dict(m="replicate noise floor (ceiling)", a="—", b="≈0.43"),
], [dict(t="Method",k="m"),dict(t="RMSE ↓",k="a",num=1),dict(t="within-gene Spearman ↑",k="b",num=1)])

TBL_ENCODER = T([
    dict(m="multihot_425 (best arm)", a="0.422"),
    dict(m="morgan+multihot", a="0.420"),
    dict(m="maccs+multihot", a="0.419"),
    dict(m="morgan_2048", a="0.415"),
    dict(m="maccs_167", a="0.414"),
    dict(m="rdkit_2048", a="0.406"),
    dict(m="chem-kNN — gate", a="0.485", _hl=1, _win=1),
], [dict(t="Chemistry encoder",k="m"),dict(t="NDCG@5",k="a",num=1,win=1)])

TBL_LOSS = T([
    dict(m="pointwise_huber (locked)", a="0.435", b="0.152"),
    dict(m="lambdarank", a="0.431", b="0.080  ⚠ collapses"),
    dict(m="lambdarank_top5", a="0.424", b="—"),
    dict(m="pointwise_mse", a="0.422", b="0.130"),
    dict(m="pairwise_ranknet", a="0.420", b="—"),
    dict(m="listmle", a="0.399", b="—"),
    dict(m="approxndcg_top5", a="0.370", b="—"),
    dict(m="approxndcg", a="0.368", b="—"),
    dict(m="chem-kNN — gate", a="0.485", b="0.240", _hl=1, _win=1),
], [dict(t="Loss",k="m"),dict(t="NDCG@5",k="a",num=1,win=1),dict(t="Spearman",k="b",num=1)])

TBL_HYBRID = T([
    dict(m="static ensemble (α≈0.8)", a="+0.008", w="the crude blend — the best of the lot"),
    dict(m="residual (learned)", a="+0.0009", w="statistical tie; sign-flips across seeds"),
    dict(m="retrieval-augmented (learned)", a="−0.0057", w="worse than the lookup alone"),
    dict(m="learned gating", a="−0.0081", w="every bit of model-weight hurt (mean α≈0.46)"),
    dict(m="promotion gate", a="≥ +0.026", w="the bar a hybrid must clear", _hl=1),
], [dict(t="Hybrid",k="m"),dict(t="honest held-out Δ NDCG@5 vs kNN",k="a",num=1),dict(t="note",k="w")])

TBL_AUG = T([
    dict(m="base — train on 23 eval orgs", a="0.4319", b="0.1522", _hl=1),
    dict(m="aug — train on all 48 orgs", a="0.4166", b="0.1270"),
    dict(m="Δ (aug − base)", a="−0.0152", b="−0.0252  ⚠ negative transfer"),
], [dict(t="Training set",k="m"),dict(t="NDCG@5",k="a",num=1),dict(t="Spearman",k="b",num=1)])

TBL_CONF = T([
    dict(q="Q1 (|t|≈0.52)", m="0.359", k="0.414", c="0.586", g="0.055"),
    dict(q="Q2 (|t|≈0.70)", m="0.375", k="0.431", c="0.562", g="0.056"),
    dict(q="Q3 (|t|≈0.93)", m="0.423", k="0.473", c="0.626", g="0.050"),
    dict(q="Q4 (|t|≈2.06)", m="0.593", k="0.633", c="0.835", g="0.040"),
], [dict(t="Confidence quartile",k="q"),dict(t="model",k="m",num=1),dict(t="chem-kNN",k="k",num=1),
    dict(t="ceiling",k="c",num=1),dict(t="gap (kNN−model)",k="g",num=1)])

TBL_COLD = T([
    dict(m="chem-kNN / linear-MF", a="inapplicable", b="—", w="0% coverage — no own-history to retrieve"),
    dict(m="chem-NULL (the gate here)", a="0.2447 [.212,.282]", b="0.0359 [.023,.051]", w="population profile"),
    dict(m="model (emb ⊕ chemistry)", a="0.2748 [.242,.318]", b="0.0735 [.052,.103]", w="beats the null — but NDCG CIs overlap", _hl=1, _win=1),
], [dict(t="Method (n=11,761 unseen genes)",k="m"),dict(t="NDCG@5 [95% CI]",k="a",num=1,win=1),
    dict(t="Spearman [95% CI]",k="b",num=1),dict(t="note",k="w")])

# ----------------------------------------------------------------------------- assemble sections
def section(id, inner):
    return "<section id='%s'>%s</section>" % (id, inner)

HOME = """
<h1>Predicting conditional gene essentiality</h1>
<p class='lede'>Can a frozen protein-language-model embedding of a gene, plus a description of a growth condition's chemistry, predict <b>when</b> that gene becomes important? Across 48–62 bacteria/archaea and tens of millions of transposon-fitness measurements, the honest answer is subtle: the signal is real but <span class='gl' data-tip='The predictive structure lives in each gene&#39;s own private deviations, not in any cross-gene pattern a global model could learn.'>local and memorization-dominated</span>, and a simple nearest-neighbour lookup beats every global model we built.</p>
<div class='grid c3' style='margin-top:18px'>
  <div class='stat'><div class='n'>48→62</div><div class='l'>organisms (original 48; the freshly re-downloaded DB has 62)</div></div>
  <div class='stat'><div class='n'>7.5–9.5k</div><div class='l'>growth-condition experiments</div></div>
  <div class='stat'><div class='n'>27–34M</div><div class='l'>gene × condition fitness rows</div></div>
</div>
<h2>The headline comparison <span class='badge b-art'>artifact-verified</span></h2>
<p>Within-organism ranking of a gene's conditions ("find its top stressors"), 23 replicate organisms, 3 seeds. Primary metric <span class='gl' data-tip='NDCG@5 ∈ [0,1], higher is better. Rewards getting a gene&#39;s top-5 most-stressful conditions right. Gate = 0.485; replicate ceiling = 0.66.'>NDCG@5</span>; within-gene Spearman secondary.</p>
""" + TBL_WARM + """
<div class='callout'><b>Read this table as the whole story.</b> Every learned global model plateaus at ~0.43; a non-parametric lookup reaches 0.485; and even the best possible predictor (replicates agreeing with themselves) only reaches 0.66 — an intrinsically noisy target with a low ceiling.</div>
<h2>The three questions this dashboard answers</h2>
<div class='grid c3'>
  <div class='panel'><h3 style='margin-top:0'>What have we done?</h3><p class='muted'>A pre-registered elimination across encoder, objective, capacity, data-volume, and hybridization. → <a href='#timeline'>Timeline</a></p></div>
  <div class='panel'><h3 style='margin-top:0'>Why can't we beat the kNN?</h3><p class='muted'>The lookup reads each gene's own measured history; the cross-gene signal a model could learn is ≈0. → <a href='#result'>Key result</a></p></div>
  <div class='panel'><h3 style='margin-top:0'>Where next?</h3><p class='muted'>Stop trying to beat the lookup; ask a question it structurally can't answer. → <a href='#directions'>Directions</a></p></div>
</div>
"""

KEYRESULT = """
<h1>The key result: why a lookup beats every model</h1>
<p class='lede'>The standard ML intuition — "a flexible model with rich features should beat a dumb lookup" — fails here for one specific, code-verifiable reason.</p>
<div class='callout warn'><b>Framing update (2026-07-06):</b> this "beat the kNN" contest is now understood to be a <b>rigged benchmark (H1)</b>, not the scientific question. The numbers below are correct; their <i>significance</i> is re-scoped. See <b><a href='#reframe'>First-principles reframe</a></b> for the corrected conclusion.</div>
<h2>1 · The kNN uses information the model structurally cannot</h2>
<div class='grid c2'>
  <div class='panel'><h3 style='margin-top:0'>chem-kNN <span class='badge b-code'>code-verified</span></h3><p>Reads <b>gene g's own measured fitness</b> at the training conditions chemically nearest to <i>c</i>, and averages them. A per-gene local regression using that gene's own answer key.</p><p class='muted'><code>chemistry_knn_predict</code> builds a gene×condition fit matrix and indexes <i>g</i>'s row — harness.py:337.</p></div>
  <div class='panel'><h3 style='margin-top:0'>global model</h3><p>Takes <code>(g's embedding, c's chemistry)</code> → predicts, using weights shared across ~54,000 genes. At inference it <b>never reads g's own labels.</b></p><p class='muted'>It compressed g's history into shared weights during training and can't recover g's idiosyncrasies at test time.</p></div>
</div>
<div class='callout good'>The kNN is not a weaker competitor — it has <b>strictly more relevant information at prediction time</b>: the target gene's own phenotypic history.</div>
<h2>2 · That own-history is the entire game here</h2>
<ul class='tight'>
  <li><b>The population signal is ≈0 within-gene.</b> <span class='badge b-code'>code</span> <span class='badge b-art'>artifact</span> The gene-blind population profile (chem-NULL) scores <b>0.023</b> within-gene Spearman. Knowing which conditions hurt genes <i>on average</i> tells you almost nothing about which conditions hurt <i>this</i> gene.</li>
  <li><b>A learned fitness-aware representation doesn't rescue it.</b> <span class='badge b-code'>code</span> <span class='badge b-art'>artifact</span> linear-MF learns a free per-gene latent <code>U[g]</code> purely from fitness and still scores 0.429 &lt; 0.485. The bottleneck is not representation quality; it's the global-vs-local structure.</li>
</ul>
<div class='callout'>A global model is a doctor predicting a never-seen patient from their demographic type. The kNN is a doctor holding <i>that patient's own chart</i>. When the patient's own history exists and is idiosyncratic — our exact setup — the second doctor wins.</div>
<h2>3 · All results, grouped by comparison</h2>
<p class='muted'>Each table compares only methods scored on the same footing. Full methodology + explained numbers for each are in the <a href='#timeline'>timeline</a>.</p>
<h3>A · Cross-organism regression (T-regime) — metric was RMSE, not NDCG</h3>""" + TBL_CROSSORG + """
<h3>B · Within-organism warm ranking — the headline</h3>""" + TBL_WARM + """
<h3>C · Chemistry encoder sweep (R1)</h3>""" + TBL_ENCODER + """
<h3>D · Loss family (R-LOSS + R-TOPK)</h3>""" + TBL_LOSS + """
<h3>E · Model + kNN hybrids (R-HYBRID)</h3>""" + TBL_HYBRID + """
<h3>F · Training-organism augmentation (R-AUG)</h3>""" + TBL_AUG + """
<h3>G · Confidence stratification (R-CONF) — NDCG@5 by quartile</h3>""" + TBL_CONF + """
<h3>H · Cold-gene diagnostic (R-COLD) — the one positive</h3>""" + TBL_COLD + """
<div class='callout warn'><b>The one-breath rebuttal to "we should be able to beat it":</b> the baseline reads each gene's own history at prediction time (verified in code), the cross-gene signal a model could learn is ≈0 (chem-NULL Spearman 0.023), and a learned fitness-aware rep still loses on the full panel. The gap isn't architecture — it's per-gene memorization. A 2025 <i>Nature Methods</i> paper reports the identical result for perturbation prediction.</div>
"""

REFRAME = """
<h1>First-principles reframe — the benchmark was rigged</h1>
<p class='lede'>A ground-up re-derivation (taking no prior conclusion at face value) that <b>amends the framing</b> of the whole warm-task program. The measurements stand; what they <i>mean</i> changes. Full version: <code>research_log/SCIENTIFIC_SYNTHESIS.md §9</code>.</p>

<h2>1 · The one equation everything hangs on</h2>
<div class='panel'><p>Any single fitness measurement decomposes as:</p>
<pre>fit(g,c) = &mu; + a_g + b_c + I(g,c) + noise</pre>
<ul class='tight'>
<li><b>a_g</b> — gene effect: how important gene <i>g</i> is <i>on average</i> (housekeeping genes are big here).</li>
<li><b>b_c</b> — condition effect: how harsh condition <i>c</i> is <i>on average</i>.</li>
<li><b>I(g,c)</b> — <b>the interaction</b>: is gene <i>g</i> <i>specifically</i> needed for condition <i>c</i>, beyond a_g and b_c? <b>This is the biology.</b></li>
<li><b>noise</b> — measurement error (the <code>t</code>-stat measures its size).</li>
</ul>
<p>Ranking one gene's conditions fixes <i>g</i> (so a_g drops out) and eligibility selects genes where b_c is tiny — so <b>the task is, purely, predicting I(g,c)</b>. That target was <i>chosen</i> by the metric + eligibility, not discovered. <b>We were always trying to predict I(g,c) — that has not changed.</b></p></div>

<h2>2 · Two hypotheses got conflated</h2>
<div class='grid c2'>
<div class='panel'><h3 style='margin-top:0'>H1 — a benchmark</h3><p>"Can a learned model predict <b>I(g,c)</b> <i>better than chem-kNN</i>?" The kNN predicts I(g,c) by reading <b>gene g's own measured I(g,c&prime;)</b> — its answer key — at other conditions.</p><p class='muted'>This is the objective the project has been chasing.</p></div>
<div class='panel'><h3 style='margin-top:0'>H2 — the science</h3><p>"Do the embedding (proxy for <i>what g does</i>) + chemistry (proxy for <i>what c demands</i>) actually <b>contain</b> generalizable I(g,c) signal, <i>where nobody has the answer key</i>?"</p><p class='muted'>This is the original foundational bet — and the real prize.</p></div>
</div>

<h2>3 · The warm contest is RIGGED three ways</h2>
<div class='callout bad'>The chem-kNN is handed: <b>(a)</b> the target gene's own labels at inference; <b>(b)</b> genes pre-selected (by eligibility) to have <i>rich histories</i>, where lookups thrive; <b>(c)</b> a random split that leaves each held-out condition a chemically-<i>near</i> measured neighbor. Given a high-rank, idiosyncratic target, a <b>lossless lookup</b> beating a <b>lossy model</b> is <i>expected</i> (Feldman long-tail) — a property of the <b>formulation</b>, not a verdict on deep learning.</div>
<div class='callout good'><b>The tell:</b> R-COLD removes the crutch (no own-history) and the model <b>wins</b> vs chem-NULL. So <b>H2 is a partial YES</b> — the embedding carries real I(g,c) signal; it's only outgunned where memorization is available.</div>

<h2>4 · What was never actually tested</h2>
<p>Every failed model was <b>global / compressing</b> (MLP; linear-MF rank-32; naive retrieval-concat; global-MLP-predicts-residual). The family designed for this regime was <b>never built</b> (verified: no code): <b>learned non-parametric</b> (EASE / learned-metric kNN / TabR / canonical ResMem), a <b>de-meaned</b> target (model only I, not a_g/b_c), a <b>denoised</b> target.</p>
<div class='callout warn'><b>The trap:</b> a learned-<i>local</i> method that wins is a <b>better memorizer</b> — it <i>confirms</i> the memorization finding, it doesn't refute it. So this sweep is worth running only to make the H1 negative <b>airtight</b>, not as a path to a modeling "win."</div>

<h2>5 · The 0.66 ceiling, corrected</h2>
<div class='panel'><p><b>Where it comes from:</b> many conditions were run twice (replicates). Use replicate A's fitness as the "prediction" and replicate B's as the "truth" → NDCG@5 &asymp; <b>0.66</b>. That's how well the measurement predicts <i>itself</i>; no model can beat it, so ~0.34 of the error is pure noise.</p>
<p><b>The caveat:</b> 0.66 is the ceiling of the <b>single-noisy-measurement</b> task — computed on the same noisy cells the model is graded on. <b>Denoising raises it.</b> The kNN&rarr;ceiling gap (0.175) is far bigger than the model&rarr;kNN gap (0.055), and is an <b>unquantified mix of irreducible noise vs. structure the fixed kNN misses.</b></p></div>

<h2>6 · Denoising — what it would look like</h2>
<p>Every <code>fit(g,c)</code> is one noisy shot. The kNN reads single noisy neighbor values and <i>inherits</i> their noise; a model that denoises can <i>average it out</i> — so denoising is one of the few moves that could shift the balance, and it was never applied to the target.</p>
<ul class='tight'>
<li><b>Replicate-average</b> — average repeated (gene, condition) measurements; the ceiling rises.</li>
<li><b>Precision-shrink by <code>t</code></b> — pull low-confidence cells toward the gene/condition mean (empirical Bayes).</li>
<li><b>Low-rank denoise</b> — the matrix is ~ low-rank (few stress programs) + sparse; fit the low-rank part as a denoised target.</li>
</ul>

<h2>7 · A non-rigged benchmark</h2>
<div class='callout'>Demote <b>chem-kNN</b> from "the gate you must beat" to "a <i>reference</i> for what pure memorization achieves on warm data." The honest benchmark removes the near-neighbor gift so <i>every</i> method must generalize:
<ul class='tight' style='margin-top:8px'>
<li><b>Primary split &rarr; leave-compound-out (scaffold / leave-chemical-class-out)</b>, plus cold-gene.</li>
<li><b>Bar &rarr; chem-NULL</b> (population profile); <b>scored as fraction of the (denoised) ceiling.</b></li>
<li>This measures <b>H2</b> — is the biology in our features — instead of H1's rigged contest.</li>
</ul></div>

<h2>8 · Other ways to formulate the question</h2>
<p class='muted'><b>MoA = Mechanism of Action</b> — for a drug, <i>how</i> it harms the cell (which target/pathway it hits; e.g. a &beta;-lactam blocks cell-wall synthesis). Deconvolving MoA = inferring an unknown compound's mechanism from its genome-wide fitness fingerprint.</p>
<table>
<thead><tr><th>Formulation</th><th>Target</th><th>Why it's different</th></tr></thead>
<tbody>
<tr><td class='hl'>Cold-condition (leave-compound-out)</td><td>I(g,c) for an <i>unmeasured</i> compound</td><td>Honest split; the <i>useful</i> prediction (untested drugs)</td></tr>
<tr><td class='hl'>Interaction-as-target</td><td>I(g,c) directly (strip a_g, b_c; low-rank+sparse)</td><td>Model only the biology, not the rank-irrelevant main effects</td></tr>
<tr><td class='hl'>Fitness-as-feature &rarr; function</td><td>the gene's <i>function</i>, from its I(g,&middot;) profile</td><td>Predict <i>from</i> fitness, not fitness itself</td></tr>
<tr><td class='hl'>Fitness-as-feature &rarr; MoA</td><td>a compound's <i>mechanism</i>, from its genome-wide response</td><td>Deconvolve unknown drugs — translational</td></tr>
<tr><td class='hl'>Matrix-structure / co-essentiality</td><td>which genes have <i>correlated</i> I-profiles (&rarr; same pathway)</td><td>Predict relationships, not cells; evaluate on recovered biology</td></tr>
<tr><td class='hl'>Cross-organism rewiring</td><td>is I(g,c) conserved or rewired across orthologs?</td><td>A comparison a within-org lookup can't compute</td></tr>
<tr><td class='hl'>Causal / dose</td><td>direct-target vs downstream, from Hill-curve shape</td><td>Uses the concentration series as a causal signal</td></tr>
<tr><td class='hl'>Active learning</td><td>which unmeasured (g,c) to test next</td><td>The decision-theoretic, actionable version</td></tr>
</tbody></table>

<h2>9 · The corrected conclusion</h2>
<div class='callout good'>"No model beats the kNN" is re-scoped to: <b>"no <i>global</i> model beats a lookup that's been handed the gene's own answer key, on a task selected to reward that."</b> Expected, low-significance, and <b>not the scientific question.</b> <b>Retire H1.</b> The live science is <b>H2 on honest splits</b> and the <b>reframed questions</b> above — now materially strengthened by the new 62-org dataset (M. tuberculosis + a TB-drug panel; in-vivo <i>mouse</i> / <i>in planta</i> conditions).</div>
"""

DATA = """
<h1>Data — analysis &amp; description</h1>
<div class='callout good'><b>Raw data restored.</b> The Fitness Browser database was re-downloaded from the live source (<code>fit.genomics.lbl.gov/cgi_data/feba.db</code>, integrity-checked OK). Note it is a <b>newer, larger release</b> than the project's original — <b>62 organisms / 9,532 experiments / 33.8M rows</b> vs the original 48 / 7,552 / 27.4M — with a different sha256, so it will not byte-reproduce the pinned numbers. <b>Still pending:</b> rebuild the canonical parquet + regenerate the ProteomeLM embeddings, then re-pin the regression gate.</div>
<h2>The dataset (Fitness Browser / RB-TnSeq)</h2>
<p>Genome-wide random-barcode transposon sequencing: knock out ~every gene, grow the pooled mutants under a condition, and read each gene's <code>fit</code> (log2 abundance change; strongly negative ⇒ important under that condition) with a significance <code>t</code>.</p>
<div class='grid c3'>
  <div class='stat'><div class='n'>62</div><div class='l'>organisms (current DB; 23 have a replicate noise floor)</div></div>
  <div class='stat'><div class='n'>9,532</div><div class='l'>experiments (stress · carbon · nitrogen · pH · temperature …)</div></div>
  <div class='stat'><div class='n'>2.84M</div><div class='l'>cross-organism ortholog links</div></div>
  <div class='stat'><div class='n'>48</div><div class='l'>stress compounds tested in ≥10 organisms (26 in ≥20)</div></div>
  <div class='stat'><div class='n'>38,525</div><div class='l'>curated specific-phenotype triples</div></div>
  <div class='stat'><div class='n'>7</div><div class='l'>(org,compound) cells with a ≥5-point dose series</div></div>
</div>
<h3>Planned analyses (to render once the parquet is rebuilt)</h3>
<ul class='tight muted'>
  <li>Distribution of <code>fit</code> and <code>t</code>; the replicate noise model (signal vs noise by <code>t</code>-bin).</li>
  <li>The shared-perturbation grid: which compounds span which organisms (the backbone for cross-organism ideas).</li>
  <li>Conditional-essentiality breadth per gene; dark-genome fraction and its phenotype rate.</li>
  <li>Ortholog coverage and matched-condition counts per organism pair.</li>
</ul>
"""

DIRECTIONS = """
<h1>Future directions</h1>
<p class='lede'>The escape from the wall isn't a better model — it's a different question the lookup <b>structurally can't answer.</b> Generated via a blind-hypothesis-generator ↔ Socratic-critic dialogue; ranked by paper-worthiness × feasibility. Full slate: <code>research_log/DIRECTIONS_adversarial_slate_2026-06-29.md</code>.</p>
<div class='panel'><div style='display:flex;justify-content:space-between'><h3 style='margin-top:0'>1 · Conserved vs. rewired conditional essentiality across orthologs</h3><span class='verdict pos'>recommended</span></div>
<p>Does a gene keep the same conditional-essentiality profile across two organisms, or get rewired? The target is a <b>cross-organism comparison</b> — the organism-local lookup can't compute it (cross-org transfer ≈0 becomes the <i>signal generator</i>). N is ample (328 hi-confidence orthologs for a distant pair; 48 compounds in ≥10 orgs). Precedents: Roguev 2008 &amp; Dixon 2008 (yeast); the ortholog conjecture.</p>
<p class='muted'><b>Good:</b> a replicate-gated population of sequence-conserved/phenotype-rewired genes, enriched for regulators, predictable beyond sequence identity. <b>Bad:</b> divergence tracks sequence divergence → clean null. <b>Guard:</b> orthology from raw <code>Ortholog</code> only; correlations train-only; gate above the replicate floor.</p></div>
<div class='panel'><h3 style='margin-top:0'>2 · Fitness fingerprint as a mechanism-of-action sensor</h3><p>For a compound whose nearest chemical neighbour has a different mechanism, does its genome-wide fitness fingerprint recover the true MoA where structure fails? Fitness as input, MoA as external label. Escape is real only on the structurally-distant slice; the live risk is N after a leave-compound-AND-organism-out split.</p></div>
<div class='panel'><h3 style='margin-top:0'>3 · The local-memorization characterization (weaponize the negative)</h3><p>Don't beat the wall — name and measure it. A rigorous benchmark: noise ceiling pinned, applicability-domain-aware evaluation, the cold-gene generalization result. ~80% already done. The methods backbone that frames a reframe paper.</p></div>
<div class='panel'><h3 style='margin-top:0'>4 · Genome → substrate-utilization (on probation)</h3><p>Predict whether a never-assayed organism uses a carbon/nitrogen source from genome content. Escapes the wall (held-out genome). <b>Blocker:</b> needs an independently measured growth phenotype to validate — confirm it exists first.</p></div>
<div class='panel'><h3 style='margin-top:0'>5 · De-leaked condition-resolved genetic-interaction networks</h3><p>Which gene pairs are co-essential only under specific stress? Biologically rich, but the obvious baseline (<code>Cofit</code>) IS the leak — advances only with a strict train-only, per-condition, noise-gated rebuild.</p></div>
<div class='callout'><b>Honorable mentions / retracted:</b> dose-response curve-shape (too few multi-point series — 7 cells); specialist↔generalist breadth law (needs an independent target); dark-genome-by-condition (premise falsified).</div>
"""

STATUS = """
<h1>Status &amp; next steps</h1>
<div class='callout good'><b>✓ Data restored.</b> <code>feba.db</code> re-downloaded from the live Fitness Browser (9.08 GB, integrity OK) — a newer release (62 orgs). <b>Remaining to fully re-arm the pipeline:</b> (1) locate/recreate <code>media_composition.xlsx</code>, (2) rebuild the canonical parquet (<code>build_canonical_v0.py</code>), (3) regenerate ProteomeLM embeddings, (4) re-pin the R-EVAL gate to the new release (old numbers won't byte-reproduce).</div>
<h2>Where we left off</h2>
<ul class='tight'>
  <li><b>Branch:</b> <code>ranking</code> (trunk). Clean modular <code>src/ranking/</code> core + shared runner + bit-exact <code>R-EVAL</code> regression gate.</li>
  <li><b>Warm baseline (23-org/3-seed):</b> model NDCG@5 0.432 / Spearman 0.152; chem-kNN gate 0.485 / 0.240. <span class='badge b-art'>artifact-verified</span></li>
  <li><b>Last result:</b> R-COLD cold-gene <b>first positive</b> — embedding beats the population null on unseen genes.</li>
  <li><b>All warm-task axes negative:</b> encoder, objective, capacity, hybrid, augmentation, confidence.</li>
  <li><b>Direction under review:</b> (i) publish the characterization, (ii) pivot to a lookup-proof question, or (iii) wind down. <b>Do not invest in new global-model architectures for the warm task.</b></li>
</ul>
<h2>The decision on the table</h2>
<table>
  <thead><tr><th>Option</th><th>What it is</th><th>Honest upside</th></tr></thead>
  <tbody>
    <tr><td class='hl'>A · Publish the negative</td><td>Rigorous memorization-dominated benchmark + cold-gene result</td><td>Real, defensible, modest; ~80% done</td></tr>
    <tr><td class='hl'>B · Pivot the question</td><td>Ortholog rewiring / MoA sensor — lookup-proof targets</td><td>Novel, higher-impact; reuses the data</td></tr>
    <tr><td class='hl'>C · Wind down</td><td>Bank the learnings, stop</td><td>Frees effort; leaves negative unpublished</td></tr>
  </tbody>
</table>
<p class='muted'>Recommendation (updated 2026-07-06, see <a href='#reframe'>reframe</a>): <b>retire H1</b> ("beat chem-kNN") — it's a rigged benchmark. Run a short, bounded diagnostic sweep (de-mean + denoise + re-run; similarity-stratified NDCG; a learned condition metric; flip the primary task to leave-compound-out) <i>only</i> to make the negative airtight for <b>A</b>, then commit the real effort to <b>B</b> — a reframed question the lookup can't answer (top pick: cross-organism rewiring, now boosted by the new M. tuberculosis + in-vivo data).</p>
<h2>Concrete next step</h2>
<p>A 1-day go/no-go spike on Direction 1: for one close (DvH↔Miya) and one distant (Caulo↔MR1) organism pair, compute the profile-correlation over matched compounds, apply the replicate-noise gate, and count the sequence-conserved/phenotype-rewired cell. If it's populated above noise, the reframe paper is real. <span class='muted'>(Now unblocked — data restored; parquet rebuild first.)</span></p>
"""

LIMITS = """
<h1>Limitations &amp; verification status</h1>
<p class='lede'>Honesty page. What was independently verified vs. what rests on the decision ledger — so no claim here is taken on faith.</p>
<div class='legend'>
  <span><span class='badge b-code'>code-verified</span> <b>strongest</b> — confirmed in source</span>
  <span><span class='badge b-art'>artifact-verified</span> reproduced from a committed run file</span>
  <span><span class='badge b-web'>web-verified</span> citation checked online</span>
  <span><span class='badge b-ledger'>ledger</span> from the decision log; <b>not</b> re-run this session</span>
</div>
<h2>Verified from primary sources</h2>
<ul class='tight'>
  <li>kNN reads the target gene's own history — <span class='badge b-code'>code</span> harness.py:337.</li>
  <li>chem-NULL is the gene-blind population profile — <span class='badge b-code'>code</span> harness.py:312.</li>
  <li>Population signal ≈0 within-gene (chem-NULL Spearman 0.023) — <span class='badge b-art'>artifact</span>.</li>
  <li>Model loses to kNN on the headline (0.432 vs 0.485) — <span class='badge b-art'>artifact</span>.</li>
  <li>linear-MF is a learned fitness-aware rep and loses on the full panel (0.429) — <span class='badge b-code'>code</span> + <span class='badge b-art'>artifact</span>.</li>
  <li>Static hybrid gain +0.008, below gate — <span class='badge b-art'>artifact</span>.</li>
  <li>Cold-gene: model beats chem-NULL, NDCG CIs overlap — <span class='badge b-art'>artifact</span>.</li>
  <li>Ahlmann-Eltze <i>Nature Methods</i> 2025 — <span class='badge b-web'>web</span>.</li>
</ul>
<h2>Corrections the verification loop caught</h2>
<div class='callout warn'><b>chem-NULL NDCG@5 is 0.343, not ~0.31.</b> The README rounded it low. The within-gene Spearman (0.02) — the number the "population signal ≈0" argument rests on — was correct.</div>
<div class='callout warn'><b>On the small 3-org "fast" set, linear-MF (0.512) slightly beats chem-kNN (0.509).</b> It only clearly loses on the full 23-org panel (0.429 vs 0.485). A quick small-scale test can look like the model "wins"; lead with the 23-org numbers.</div>
<h2>Reproduction caveat (new)</h2>
<div class='callout warn'><b>The re-downloaded DB is a newer release (62 orgs vs the original 48).</b> Every headline number in this dashboard was computed on the original 48-org DB. Re-running on the new DB will shift them; the pinned regression gate must be re-established before any number here is treated as current.</div>
<h2>Most numbers here are ledger, not re-run</h2>
<ul class='tight muted'>
  <li>The per-experiment tables (R1, R-LOSS, R-AUG, R-CONF, R-HYBRID-B) are quoted from the decision ledger. The headline warm table, the hybrid static gain, and the cold-gene result are artifact-verified; the rest await a fresh run on restored data.</li>
  <li>Replicate ceiling 0.66 / 0.39 — method code-verified; exact value from prior runs.</li>
</ul>
<div class='callout'><b>Caveat on the key citation:</b> the Nature Methods 2025 title says "not <i>yet</i>," and it studies <i>transcriptomic</i> perturbation response — cite it as a strong <b>analog</b>, not the identical task.</div>
"""

REFS = """
<h1>References</h1>
<p class='lede'>Verified citations underpinning the "why the lookup wins" argument and the reframe directions.</p>
<h2>The memorization-beats-global regime</h2>
<ul class='tight'>
  <li>Ahlmann-Eltze, Huber &amp; Anders. <b>Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines.</b> <i>Nature Methods</i> 2025. <a href='https://www.nature.com/articles/s41592-025-02772-6' target='_blank'>10.1038/s41592-025-02772-6</a> <span class='badge b-web'>web-verified</span></li>
  <li>Khandelwal et al. <b>Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM).</b> ICLR 2020. <a href='https://arxiv.org/abs/1911.00172' target='_blank'>arXiv:1911.00172</a> <span class='badge b-ledger'>ledger</span></li>
  <li>Feldman. <b>Does Learning Require Memorization? A Short Tale about a Long Tail.</b> STOC 2020. <a href='https://arxiv.org/abs/1906.05271' target='_blank'>arXiv:1906.05271</a> <span class='badge b-ledger'>ledger</span></li>
  <li>Grinsztajn et al. <b>Why do tree-based models still outperform deep learning on tabular data?</b> NeurIPS 2022. <a href='https://arxiv.org/abs/2207.08815' target='_blank'>arXiv:2207.08815</a> <span class='badge b-ledger'>ledger</span></li>
  <li>Rendle et al. <b>Neural Collaborative Filtering vs. Matrix Factorization Revisited.</b> RecSys 2020. <a href='https://arxiv.org/abs/2005.09683' target='_blank'>arXiv:2005.09683</a> <span class='badge b-ledger'>ledger</span></li>
</ul>
<h2>Conserved-vs-rewired (Direction 1)</h2>
<ul class='tight'>
  <li>Roguev et al. <b>Conservation and Rewiring of Functional Modules Revealed by an Epistasis Map in Fission Yeast.</b> <i>Science</i> 2008. <a href='https://www.science.org/doi/10.1126/science.1162609' target='_blank'>10.1126/science.1162609</a> <span class='badge b-web'>web-verified</span></li>
  <li>Dixon et al. <b>Significant conservation of synthetic lethal genetic interaction networks between distantly related eukaryotes.</b> <i>PNAS</i> 2008. <a href='https://www.pnas.org/doi/10.1073/pnas.0806261105' target='_blank'>10.1073/pnas.0806261105</a> <span class='badge b-web'>web-verified</span></li>
  <li>Altenhoff et al. <b>Resolving the Ortholog Conjecture.</b> <i>PLoS Comp Biol</i> 2012. <a href='https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002514' target='_blank'>10.1371/journal.pcbi.1002514</a> <span class='badge b-web'>web-verified</span></li>
  <li>Rousset et al. <b>The impact of genetic diversity on gene essentiality within the E. coli species.</b> Nat. Microbiology 2021. <a href='https://www.biorxiv.org/content/10.1101/2020.05.25.114553v1.full' target='_blank'>bioRxiv</a> <span class='badge b-web'>web-verified</span></li>
</ul>
<h2>The dataset</h2>
<ul class='tight'>
  <li>Price et al. <b>Mutant phenotypes for thousands of bacterial genes of unknown function.</b> <i>Nature</i> 2018. (The Fitness Browser / RB-TnSeq source dataset.) Data: <a href='https://fit.genomics.lbl.gov/' target='_blank'>fit.genomics.lbl.gov</a> <span class='badge b-ledger'>ledger</span></li>
</ul>
"""

TIMELINE = (
  "<h1>Experiment timeline</h1>"
  "<p class='lede'>Every experiment and tweak, in order. Each card states its <b>hypothesis</b>; click to expand <b>how we tested it</b>, the <b>results</b> (with every number explained), an <b>analysis</b>, and <b>what it means</b>. Badges mark how each result was verified.</p>"
  "<div class='legend'>"
  "<span><span class='badge b-art'>artifact</span> reproduced from a committed run file</span>"
  "<span><span class='badge b-code'>code</span> confirmed in source</span>"
  "<span><span class='badge b-web'>web</span> checked online</span>"
  "<span><span class='badge b-ledger'>ledger</span> from the decision log; not re-run this session</span>"
  "</div>"
  "<div class='tlbar'>"
  "<div class='filters' id='filters'>"
  "<span class='muted'>filter:</span>"
  "<button class='chip active' data-f='all'>all</button>"
  "<button class='chip' data-f='pos'>✓ worked</button>"
  "<button class='chip' data-f='neg'>✗ didn't</button>"
  "<button class='chip' data-f='neu'>infra/context</button>"
  "</div>"
  "<div class='tlbtns'><button class='tbtn' id='expandAll'>expand all</button>"
  "<button class='tbtn' id='collapseAll'>collapse all</button></div>"
  "</div>"
  "<div class='tl' id='tl'>" + "".join(render_experiment(e) for e in EX_ORDER) + "</div>"
)

LEARN_SEC = (
  "<h1>What we've learned</h1>"
  "<p class='lede'>The durable findings — what would survive even if we deleted all the code. Each has its supporting evidence and how we know it.</p>"
  "<div class='callout warn'>These are accurate <b>measurements</b>. Their <b>interpretation</b> was re-scoped on 2026-07-06: the \"chem-kNN beats every model\" finding is now understood as a property of a <b>rigged benchmark (H1)</b>, not a deep verdict on modeling — see <b><a href='#reframe'>First-principles reframe</a></b>.</div>"
  + "".join(render_learning(l, i+1) for i, l in enumerate(LEARNINGS))
)

SECTIONS = [
  ("home", HOME), ("result", KEYRESULT), ("reframe", REFRAME), ("data", DATA),
  ("timeline", TIMELINE), ("learnings", LEARN_SEC), ("directions", DIRECTIONS),
  ("status", STATUS), ("limits", LIMITS), ("refs", REFS),
]

NAV = """
<div class='sep'>Orientation</div>
<a href='#home'><span class='ic'>◆</span> Overview</a>
<a href='#result'><span class='ic'>★</span> The key result</a>
<a href='#reframe'><span class='ic'>⟳</span> First-principles reframe</a>
<a href='#data'><span class='ic'>▤</span> Data</a>
<div class='sep'>The work</div>
<a href='#timeline'><span class='ic'>↧</span> Experiment timeline</a>
<a href='#learnings'><span class='ic'>✦</span> What we've learned</a>
<a href='#directions'><span class='ic'>➤</span> Future directions</a>
<div class='sep'>Meta</div>
<a href='#status'><span class='ic'>◈</span> Status &amp; next steps</a>
<a href='#limits'><span class='ic'>⚠</span> Limitations &amp; verification</a>
<a href='#refs'><span class='ic'>§</span> References</a>
"""

# ----------------------------------------------------------------------------- CSS + JS (raw, no .format)
CSS = open(ROOT / "assets" / "style.css").read() if (ROOT/"assets"/"style.css").exists() else ""
JS  = open(ROOT / "assets" / "app.js").read() if (ROOT/"assets"/"app.js").exists() else ""

PAGE = (
  "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
  "<meta name='viewport' content='width=device-width, initial-scale=1'>"
  "<title>Conditional Gene Essentiality — Project Dashboard</title>"
  "<style>" + CSS + "</style></head><body>"
  "<div class='mobilebar'><button id='navToggle'>☰ menu</button><span>Conditional Gene Essentiality</span></div>"
  "<div class='layout'>"
  "<aside class='side' id='side'>"
  "<div class='brand'>Conditional Gene Essentiality<small>Project dashboard · working reference</small></div>"
  "<nav class='nav' id='nav'>" + NAV + "</nav>"
  "<div class='toolrow'>"
  "<button class='tool' id='themeToggle' title='light / dark'>◐</button>"
  "<button class='tool' id='presentToggle' title='presentation mode'>▶ present</button>"
  "<button class='tool' onclick='window.print()' title='print / PDF'>⎙ print</button>"
  "</div>"
  "<div class='sidefoot'><div><span class='dot good'></span>branch <code>ranking</code></div>"
  "<div style='margin-top:6px'><span class='dot good'></span>data restored (62-org release)</div>"
  "<div style='margin-top:10px' class='muted'>Every claim carries a verification badge — see <a href='#limits'>Limitations</a>.</div></div>"
  "</aside>"
  "<main id='main'>" + "".join(section(i, c) for i, c in SECTIONS) + "</main>"
  "</div>"
  "<div class='present-nav' id='presentNav'><button id='pPrev'>‹</button><span id='pLabel'></span><button id='pNext'>›</button><button id='pExit'>✕ exit</button></div>"
  "<script>" + JS + "</script></body></html>"
)

(ROOT / "index.html").write_text(PAGE, encoding="utf-8")
print("wrote index.html  (%d bytes, %d experiments, %d learnings)" % (
    len(PAGE), len(EX_ORDER), len(LEARNINGS)))
