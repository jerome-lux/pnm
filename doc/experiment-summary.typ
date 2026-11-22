// PNM Generation Studies — Experiment Summary (GM1)
// Compile with: typst compile experiment-summary.typ   (PDF written alongside)

#set document(title: "PNM Generation Studies — Experiment Summary (GM1)")
#set page(paper: "a4", margin: 1.6cm)
#set text(size: 10pt, lang: "en")
#set par(justify: true)

= PNM Generation Studies — Experiment Summary (GM1)

== Scope

Recap of every parameter study run on the GM1 mercury porosimetry dataset
(`examples/data/GM1.csv`), all with #strong[seed 7], scored by the Jensen–Shannon divergence (JSD)
between the generated throat volume distribution and the experimental target. Companion theory:
`doc/pore-distribution-inversion.md` (kernel model) and `doc/capillary-length-phase2.md`
(derived capillary length).

== 0. Conventions and metrics

- #strong[JSD] (`pnm.Stats.jensen_shannon_divergence`): symmetric, finite on disjoint supports; the study ranking metric, computed on per-bin throat volume fractions over the target-window bins.
- "pt" = percentage points of intruded volume in a band.
- #strong[Input windows] (throat radius). The window is a *practical constraint on the input distribution*, not an optimization variable: the untruncated GM1 curve has extreme tails that are finite-size-limited at $N <= 2 times 10^5$ (only ~13–21 top-bin-eligible pores), so two tractable input windows were used. All parameter effects below are measured *for a fixed given input distribution*; JSD values across different windows are not directly comparable as "improvements".
  + #strong[W1]: explicit interval $[0.0162, 1.594]$ µm; experimental large band ($> 0.5$ µm) = #strong[33.1%].
  + #strong[keep75]: symmetric cumulative-volume truncation keeping the central 75% of base volume → $[~ 0.0069, ~ 1.0485]$ µm; target bands: b1 $< 0.012$ µm = #strong[25.7%], …, large $>= 0.5$ µm = #strong[19.7%].
  + *untruncated*: full base curve (95% cumulative filter) — used only in the early studies (rows 1–4 and 6; row 7 uses a near-full "wide" window).
- Pore bins are always `throat_radii * PRTRR` with PRTRR $= 2.0$; target porosity $0.35$.

== 1. Pipeline components under test

=== Inversion models (experimental throat distribution → pore target)

#table(
  columns: (20%, 34%, 20%, 26%),
  align: left,
  table.header(strong[Model], strong[Function], strong[Free parameters], strong[Status]),
  "Independent (i.i.d.) pairing", `invert_pore_distribution_independent`, $e$ + " (+ continuous shift gauge in the early studies)",  "Classical model of " + `doc/theory.typ` + "; known pairing bias (under-produces large–large throats in the actual generator)",
  "Exclusion kernel", `invert_pore_distribution_kernel`, $e$ + " (+ porosity, sratio inputs)",  strong[Production default] + " — models the generator's surface-budget pairing; no ad-hoc gauge",
  "Kernel + derived capillary length (gamma)", `invert_pore_distribution_kernel_gamma`,  "none (" + `length_stat="mean"` + ")", "Parameter-free variant; structural b1 deficit remains",
  "Gamma, global packing reach",  "same, " + `length_stat="packing"`, "—",  strong[Rejected] + " (§4.5)",
  "Gamma hybrid power-law tail", `invert_pore_distribution_kernel_gamma_hybrid`, $beta$ + " (tail exponent)",  strong[Rejected as reference recipe] + ", kept for research (§4.6)",
  "Class-dependent degree",  `invert_pore_distribution_kernel_experimental` + " (" + `zcap` + ")", "—", "Research only; unstable without a degree cap, pushes mass away from large classes",
  "Kernel water-filling (multi-capillary)", `invert_pore_distribution_kernel_waterfill`, $e$ + " (+ porosity, sratio inputs; optional " + `zcap` + ")", { "Joint per-class (count, composition) from the greedy budget scan (theory doc §5.6); under evaluation — rows 21–23" },
  "Kernel budget-coupled degree", `invert_pore_distribution_kernel_budget`, $e$ + " (+ porosity, sratio inputs; optional " + `zcap` + ")", { "Production kernel composition unchanged, count coupled through the budget-saturated degree (theory doc §5.6); under evaluation — rows 21–23" },
)

=== Throat generation methods

#table(
  columns: (19%, 38%, 43%),
  align: left,
  table.header(strong[Method], strong[Description], strong[Notes]),
    "v1 " + `surf` + " (two-phase)", { "phase 1 (optional, " + `phase1_z > 0` + ") seeds each pore with its largest partners in a `(factor+1)*r_i` ball; phase 2 fills the remaining surface budget nearest-surface-distance-first (" + `order` + " = increasing/decreasing/random) over a `p2f*r_i + R_max` ball, then enforces a " + `phase2_zmin` + " connectivity floor; optional " + `maxc` }, { strong[Production default] + " for W1 recipes (old single-pass form); rewritten 2026-11-19 — the old knobs (" + `sort_by_radius` + ", " + `minc_search_factor` + ") are now warn-and-ignore under \"surf\"; basis of rows 21–23" },
   "v2 " + `surf_v2`,  "phase-based with surface-gap caps (" + `factor1/factor2` + "), per-pore large-pair floor", { "Too coarse at " + $N <= 10^5$ + " (K=4 already overshoots the " + $> 0.5$ + " µm band ~3×)" },
   "v3 " + `surf_v3` + " two-pass",  "pass 2 = minimal i.i.d.-style baseline; optional " + strong[pass 1] + " = independent large-pore sub-network (" + `large_radius_cutoff` + ", per-large-pore floor " + `minc` + ")", { "Pass 1 is required to fill the large band (§4.3)" },
  `add_throats_by_distribution`, "direct pair assignment from a prescribed volumetric throat distribution",  "Implemented + validated at N=80k (placement 95%, JSD 0.1768); " + strong[not wired anywhere] + " yet",
)

== 2. Study recap table (chronological)

// Landscape page group: the full study table, all columns visible at once per row
#set page(width: 29.7cm, height: 21cm)

{
  #set text(size: 8pt)
  #let rowfill(r, c) = { if calc.rem(r, 2) == 0 { luma(94%) } else { white } }
  #table(
    columns: (0.5cm, 1.3cm, 2.7cm, 3.4cm, 1.5cm, 1.4cm, 1.9cm, 1.8cm, 3.6cm, 2.5cm, 1fr),
    align: left,
    fill: rowfill,
    table.header(
      strong[#"#"], strong[Date], strong[Study], strong[Results dir], strong[N], strong[Runs],
      strong[Inversion], strong[Throat gen.], strong[Best config], strong[Best JSD],
      strong[Status / key finding],
    ),
    "1", "pre-11-11", "v1 study (untruncated)", "—", "80k", "—", "indep.", "v1 surf", "—", "—",  strong[INVALID] + " — corrupted by the " + `autothroats=True` + " bug + pore-sampling off-by-one; do not use",
    "2", "~11-11/12", "v2: untruncated, sbr=True, srf=3", `param_study_results_v2/`, "80k", "42", "indep.", "v1 surf", "e=1.8 / sr=0.8", strong[0.0704], "First valid study; large band structurally missed",
    "3", "~11-11/12", "v3: + continuous shift gauge", `param_study_results_v3/`, "80k", "75", "indep. + shift", "v1 surf",  "e=1.5 / sr=0.9 / shift=" + $+0.5$, strong[0.0533],  "Optimal shift coupled to " + $e$ + "; still no large-band match",
    "4", "~11-12/13", "surf_v2 405-run (untruncated) + N-scan", `param_study_results_surfv2_{80k,150k,20k}/`, "80k (+150k, 20k)", "405 (+8, +24)", "indep.", "v2", "e=1.75 / sr=0.8 / f1=None / decreasing",  strong[0.0543] + " (80k); 0.0598 (150k); 0.1297 (20k)",  "No config matches both large and intermediate bands; sratio composition-invariant; f1/order are coarse upward-only levers; " + $> 1.97$ + " µm band empty even at N=150k",
    "5", "~11-13/14", "Truncated W1 window (first tractable-input study)", `param_study_results_trunc_80k/`, "80k", "42", "indep. + shift", "v2",  "e=1.75 / shift=" + $-0.5$ + " / f1=None / decreasing", strong[0.0088],  "On a finite-size-feasible input, parameter tuning reaches JSD ~0.01; f1=1 buys large-pore connectivity at " + $+0.006$ + " JSD cost",
    "6", "~11-13/14", "Large-pair floor (per-pore connection floor)", `param_study_results_largepair_100k/`, "100k", "6", "indep.", "v2 (+ large-pair floor)", "—", strong[0.0873],  "Per-pore floor too coarse; connectivity of largest pores " + $!=$ + " large-band volume fraction (finite-size artifact on untruncated GM1: only ~13–21 top-bin-eligible pores)",
    "7", "~11-14", "v1 vs v2 A/B, N=200k wide window", "(ad hoc)", "200k", "2", "indep.", "v1 vs v2", "v1 surf",  strong[0.112] + " vs 0.158 (v2)",  "v1 wins clearly — kills a spurious " + $r_t approx 2$ + "–2.5 µm bump; notebook default switched to \"surf\"",
    "8", "11-14", "Kernel inversion A/B (W1 + WIDE windows)", `examples/networks/20261114-18*_kernelAB-*`, "80k", "—", "kernel", { "v1 surf (sbr=True, srf=2)" }, "e=1.75 / sr=0.8",  strong[0.0063] + " (W1); 0.0828 (WIDE)",  "Best W1 JSD ever at the time, no ad-hoc gauge; large band 27.6% vs exp 33.1%; beats previous best recipe that needed an ad-hoc " + `pdist_shift`,
    "9", "11-15", "v3 pass 1 + kernel, keep75 (valley cutoff)", `param_study_results_v3kernel_keep75_100k/`, "1e5", "60", "kernel", { "v3 two-pass (cut=valley)" }, "e=1.25 / sr 0.7–0.8 / minc=3", strong[0.0341],  "JSD(" + $e$ + ") valley flat over 1.1–1.5, minimum at 1.25; large band only 8.8% vs target 19.7% (" + $e approx 0.9$ + " hits it: 19.9% at JSD 0.0637); minc=3 " + $>=$ + " minc=2 in most cells",
    "10", "11-16", "Gamma inversion, keep75 (valley cutoff)", `param_study_results_kernelgamma_keep75_100k/`, "1e5", "11",  "gamma (no " + $e$ + ")", { "v3 two-pass (cut=valley)" }, "sr=0.7 / minc=2", strong[0.0284],  "Parameter-free; large band near-exact at sr=0.6 (" + $+0.05$ + " pt); minc=2 wins everywhere (opposite of #9); residual = b1 deficit + b3/b4 excess",
    "11", "11-16",  "v3 pass-1 cutoff study, keep75 (no-pass-1 control + cut " + $in \{0.5, 1.0\}$ + " µm)",  `..._v3nopass1_{kernel,gamma}_keep75_100k/` + linebreak() + `..._v3cut{0.5,1.0}_keep75_100k/`, "1e5", "9 + 12", { "kernel (control), gamma (sweep)" }, { "v3 two-pass (no pass 1 / cut=0.5 / cut=1.0)" }, { "gamma: cut=0.5 / sr=0.8 / minc=2" },  strong[0.0234] + " (gamma); no-pass-1: 0.0996–0.17",  "Without pass 1 the large band is structurally empty (" + $<=$ + "1.3%); restricting pass 1 to pores " + $>= 0.5$ + " µm removes forced connections among ~2500 intermediate pores that overproduced b4/b5; at cut=1.0 minc=1 required (minc=2 → " + $+18 ... +27$ + " pt large-band excess); " + strong[b1 deficit persists across ALL pass-1 variants — structural, not a cutoff artifact],
    "12", "11-16", { "Kernel e=1.25 + v3 cut=0.5, keep75" }, `..._v3cut0.5_kernel_keep75_100k/`, "1e5", "6", "kernel", { "v3 two-pass (cut=0.5)" }, "sr=0.7 / minc=2", strong[0.0195],  "Combination works as hypothesized: b1 improves to " + $-8 ... -10$ + " pt (vs gamma's " + $-14$ + "), large band near-exact at sr " + $<= 0.7$ + "; minc=2 wins at every sr",
    "13", "11-16",  "Gamma " + `length_stat="packing"` + ", keep75", `..._v3cut0.5_kernelgamma_pack_keep75_100k/`, "1e5", "3", { "gamma (global packing reach)" }, { "v3 two-pass (cut=0.5)" }, "—", strong[0.185–0.202],  strong[Rejected] + ": global physical reach collapses large-class step-1 lengths; b1 collapses to 0.7–0.9% (target 25.7); the class dependence of the budget reach is load-bearing (§4.5)",
    "14", "11-16",  "Gamma hybrid power-law tail, keep75 (" + $beta$ + " scan)",  `..._v3cut0.5_kernelgamma_hybrid_keep75_100k/` + linebreak() + `..._hybrid_bscan_keep75_100k/`, "1e5", "8",  "gamma hybrid (" + $beta$ + ")", { "v3 two-pass (cut=0.5)" },  $beta$ + "=0.5 / sr=0.7 / minc=2 (" + $beta$ + " scan minimum at/below " + $beta$ + "=0.3: 0.0234)", strong[0.0243],  "Ties gamma-mean, does NOT beat kernel e=1.25; structural b1-deficit + b3-excess pair persists at every " + $beta$ + "; matching measured gap lengths is not sufficient for reproducing the throat distribution (§4.6); kept for research only",
    "15", "11-16", { "N=2e5 scale test, keep75 (kernel + gamma)" },  `..._v3cut0.5_kernel_200k_keep75/` + linebreak() + `..._kernelgamma_200k_keep75/`, "2e5", "8", { "kernel + gamma" }, { "v3 two-pass (cut=0.5)" }, { "kernel e=1.25 / sr=0.7" },  strong[0.0188] + " (vs 0.0195 at N=1e5)",  "Only " + $-0.0007$ + " gain — " + strong[the structural residual is a model+generator effect, NOT a finite-size artifact] + "; optimal recipe does not shift with N; gamma still trails kernel by ~0.011 JSD (0.0298)",
    "16", "11-16", { "Kernel fine-tune around best recipe, keep75" }, `..._v3cut0.5_kernel_finetune_keep75_100k/`, "1e5", "9", "kernel", { "v3 two-pass (cut=0.5)" }, { "e=1.25 / sr=0.75" }, strong[0.0189],  $e$ + " valley narrow and asymmetric (JSD 0.041/0.019/0.029 at " + $e$ + "=1.0/1.25/1.5); " + $e$ + " trades b1 against the mid bands, not against the large band — no (" + $e$ + ", sr) fixes both ends; residual = b1 " + $-9.3$ + " pt + b3 " + $+7.3$ + " pt",
    "17", "11-17", { "Independent inversion + v3 pass 1, keep75 (cut=0.5 and valley)" },  `..._v3cut0.5_independent_keep75_100k/` + linebreak() + `..._v3valley_independent_keep75_100k/`, "1e5", "30", "indep.", { "v3 two-pass (cut=0.5 / valley)" }, { "e=1.25 / sr=0.7 (cut=0.5)" },  strong[0.0232] + " (cut=0.5); 0.0614 (valley)",  $+19%$ + " vs kernel e=1.25 — confirms the pairing bias: i.i.d. targets carry less large-pore mass than the kernel's; at valley cutoff the large band collapses to 3.0% and JSD is nearly flat in sratio",
    "18", "11-17", { "Diameter-dependent sratio recalibration, keep75 (gamma)" }, `..._v3cut0.5_kernelgamma_srprof_keep75_150k/`, "1.5e5", "21", { "gamma (+ sratio profile)" }, { "v3 two-pass (cut=0.5)" },  "flat sr=1.0 → " + strong[0.0169] + " (idealized bound); conservative " + $<= 0.9$ + ": p1-0.9@0.02-0.5 → " + strong[0.0179] + ", flat 0.9 → 0.0194", { "see left" },  "Decreasing profiles with large-pore floor " + $<= 0.4$ + " all FAIL (JSD 0.030–0.075): sratio is an input that determines the pore target itself, not a post-hoc scaling knob; " + strong[sr $> 1$ is unphysical] + " (more throat cross-section than pore surface area) — apparent further gains (flat 1.4 → 0.0149, p1.4-1 → 0.0137) are over-connectivity artifacts, invalid for production",
    "19", "11-17", { "W1 kernel recipe: maxc calibration + N=2e5 scale test" },  `..._kernelAB_W1_{80k_calib_maxc1024,200k}/` + " (+ ablation dirs)", "80k / 2e5", "4 + 1", "kernel", { "v1 surf (sbr=True, srf=2)" }, { "e=1.75 / sr=0.8 / maxc=1024" },  strong[0.0063] + " (80k) → " + strong[0.0042] + " (2e5, " + $-33%$ + ")",  `maxc=1024` + " is REQUIRED (default 512 cuts off the giant-pore candidate scan → JSD ~0.024); calibrated 80k run matches the original A/B network within MC noise (E=747,587 vs 746,131); large band improves 27.6% → 30.3% vs exp 33.1% — " + strong[the W1 recipe scales favorably with N],
     "20", "11-17", { "W1 maxc/mnf/sratio factorial, N=150k" },  `param_study_results_w1_maxc_mnf_150k/` + linebreak() + "(analysis: " + `examples/analyze_w1_maxc_mnf_150k.py` + ")", "1.5e5", "80", "kernel", { "v1 surf (sbr=True, srf=2)" }, strong[sr=0.5 / maxc=512 / mnf=10],  strong[0.0026] + " ← global best",  "Every band within ~3 pt of exp; large band 31.9% vs 33.1% (runner-up sr=0.7/maxc=1024/mnf=10 → 0.0027, large band 34.2%). " + strong[maxc and sratio are COUPLED] + " (low sr ↔ low optimal cap; at sr=0.8 maxc=512 collapses the large band to ~19%); " + `inf` + " is never best (over-connects the giants, top-class mean degree ~2160); mnf monotone higher-is-better with diminishing returns beyond ~6",
     "21", "11-19", { "v1 two-phase rewrite: e/sr/p1z grid (order=increasing fixed), asym50_95 window [0.068, 26.53] µm" }, `param_study_v2_asym50_95_e_sr_p1z_150k/`, "1.5e5", "72", "kernel", { "v1 two-phase (" + `p2f=3` + ", maxc=1024)" }, "e=2 / sr=0.3 / p1z=2 (increasing)", strong[0.0594],  "First study of the rewritten two-phase v1 on a wide window (large-band target 53.2%); with order fixed to increasing, JSD floors at ~0.06 — the processing-order dimension turns out decisive (row 22)",
     "22", "11-19", { "v1 two-phase full grid (+ p2f, order), asym50_95" }, `param_study_v2_asym50_95_e_sr_p1z_p2f_order_150k/`, "1.5e5", "432", "kernel", { "v1 two-phase (maxc=1024)" }, strong[e=1.75 / sr=0.7 / decreasing / p1z=0],  strong[0.0093],  "order is the dominant lever: decreasing/random reach ~0.01 vs increasing's 0.06; " + `phase1_z=0` + " (pass 1 off) wins at every e/sr; large band 48.0% vs target 53.2% (" + $-5.2$ + " pt); in decreasing order the small pores near a giant connect to it because giants claim partners first while all budgets are fresh",
     "23", "11-19/20", { "Multi-capillary inversions (waterfill + budget), asym50_95 — same grid as row 22" }, `param_study_v2_asym50_95_wf_bud_e_sr_p1z_p2f_order_150k/`, "1.5e5", "864", { "kernel_waterfill + kernel_budget" }, { "v1 two-phase (maxc=1024)" }, { "budget: e=1.25 / sr=0.4 / random / p1z=0; waterfill: e=1.25 / sr=0.6 / random / p1z=0" }, strong[0.0099] + " (budget); 0.0129 (waterfill)", { "Both beat the plain kernel in 63–67% of grid cells (mean ΔJSD " + $-0.07 ... -0.085$ + ") but do not beat its single best point (0.0093, within MC noise). Optimal " + $e$ + " shifts from 1.75 to 1.25 and the optimal order flips decreasing → random; budget (degree-only) " + $>=$ + " waterfill everywhere — fixing only the count is sufficient on both windows tested. Mechanism: relative to the plain kernel at equal (" + $e$ + ", sr), both corrections shift pore mass toward smaller classes — on keep75 that degrades the large band, but here it corrects the low-" + $e$ + " flooding (kernel @ e=1.25/sr=0.6: large 91.8% → ~54–55% ≈ target 53.2%). Budget's best has a near-exact large band (" + $+1.8$ + " pt vs kernel's " + $-5.2$ + ") at lower connectivity (9.3 vs 18.9)" },
  )
}

Notebook validation of the current default recipe (row 22, asym50_95 kernel), executed 2026-11-20 at N=2.5e5 via `generate_best_networks.py`:
#strong[JSD = 0.0118], l2 = 3.37e-2, E = 2,374,650 throats, mean connectivity 19.00, large band ($>= 0.5$ µm) 47.4% vs exp 53.2% — consistent with the study value (0.0093 at N=1.5e5); network saved in `examples/networks/20261120_asym5095_250k_kernel_e1.75_sr0.7_dec.gml` (three sibling networks from the other top recipes alongside).

// Back to portrait
#set page(paper: "a4")

== 3. Current best recipes by objective

#table(
  columns: (26%, 38%, 36%),
  align: left,
  table.header(strong[Objective], strong[Recipe], strong[Result]),
    strong[Overall shape, W1 window] + " (former notebook default — NOT reproducible with the rewritten two-phase v1: its knobs " + `sort_by_radius` + "/" + `minc_search_factor` + " are now warn-and-ignore under \"surf\")",  "kernel e=1.75 + " + strong[sratio=0.5 / maxc=512 / min_neighbors_factor=10] + ", OLD single-pass v1 surf (" + `sort_by_radius=True` + ", " + `minc_search_factor=None` + ", " + `search_radius_factor=2` + "), W1 " + $[0.0162, 1.594]$ + " µm",  strong[JSD 0.0026] + " @ N=1.5e5 (row 20); JSD 2.42e-3 @ N=1e5 in the notebook; large band 31.9% vs exp 33.1% — historical value only, recipe must be re-expressed in two-phase parameters before reuse",
  { "Parameter-free, keep75 window" },  "gamma inversion (no " + $e$ + ") + flat " + strong[sratio=0.9] + ", v3 cut=" + $0.5$ + " µm / minc=2",  "JSD " + strong[0.0194] + " @ N=1.5e5; b1 " + $-12.9$ + " pt; large band " + $+3.1$ + " pt — essentially tied with the one-parameter kernel recipe within MC noise (profile p1-0.9@0.02-0.5 → 0.0179 but anchored at sr_small=1.0 = full surface coverage, borderline)",
    "One free parameter, keep75 window (former notebook alternative)",  "kernel e=1.25 + sratio=0.75, v3 cut=" + $0.5$ + " µm / minc=2",  "JSD " + strong[0.0189] + " @ N=1e5; b1 16.5% vs target 25.7 (" + $-9.3$ + " pt); large band " + $-1.6$ + " pt",
    strong[Current notebook default] + ", overall shape, wide asym50_95 window [0.068, 26.53] µm", { "kernel e=1.75 + sratio=0.7 / decreasing / p1z=0, v1 two-phase (" + `p2f=2` + ", maxc=1024) — or kernel_budget e=1.25 + sratio=0.4 / random / p1z=0 (tied JSD at N=1.5e5, better large band)" }, { "JSD 0.0093 (kernel) / 0.0099 (budget) @ N=1.5e5; budget's large band 55.1% vs target 53.2 (" + $+1.8$ + " pt) vs kernel's 48.0 (" + $-5.2$ + "); budget at lower connectivity (9.3 vs 18.9). Kernel validated at N=2.5e5 on 2026-11-20: JSD 0.0118, large band 47.4% (" + $-5.8$ + " pt), E = 2.37M — the first recipe that matches the full experimental shape including the large band" },
)

== 4. Key structural insights

#enum(
   strong[The input window is a practical constraint, not part of the optimization.] + " The studies measure parameter effects " + emph[for a fixed given input distribution] + "; choosing a narrower window is not an \"improvement\" and is never counted as one. Untruncated GM1 is simply infeasible at " + $N <= 2 times 10^5$ + ": its extreme tails are finite-size-limited (~13–21 top-bin-eligible pores), so W1 and keep75 serve as tractable input distributions, and all comparisons below are made within one fixed window.",
   strong[maxc and sratio are coupled] + " (row 20): low sr ↔ low optimal connectivity cap, because the budget is spent on small throats before distant giant partners are reached; unbounded caps over-connect the giants and hurt shape. " + `min_neighbors_factor` + " raises small-pore connectivity (~" + $+15%$ + " from 2→10) without moving the large classes — a nearly free JSD improvement with diminishing returns beyond ~6.",
   strong[v3 pass 1 (forced large–large pairing) is required] + " to fill the " + $> 0.5$ + " µm band: without it the band is structurally empty (" + $<=$ + "1.3%) at N=1e5; restricting pass 1 to pores " + $>=$ + " cutoff removes overproduction of intermediate bands, and the optimal minc depends on how many giant pores survive the cutoff (cut=0.5 → minc=2 wins for sr " + $>= 0.7$ + "; cut=1.0 → minc=1 required).",
   strong[The keep75 structural residual is a model+generator effect, not finite size] + " (row 15): doubling N moves JSD by " + $-0.0007$ + " while the b1-deficit + b3-excess pair persists unchanged. The next model fix is the transition-region length correction of " + `doc/capillary-length-phase2.md` + " §7 — it must act on " + strong[both ends] + " of the transition region simultaneously (b3 excess at 0.03–0.05 µm AND the b1 deficit in the smallest classes).",
   strong[The class dependence of the budget reach is load-bearing] + " (row 13): replacing it by a global physical number-density reach collapses the small-pore mass (JSD ×6–7 worse) — an implicit large-throat penalty that keeps small-pore mass in the target cannot be replaced by a global length scale.",
   strong[Matching measured gap lengths is not sufficient] + " for reproducing the throat distribution (row 14): below m_switch=" + $0.03$ + " µm the " + emph[unvalidated] + " power-law extrapolation beats the " + emph[validated] + " derived profile on b1 — in the smallest classes, as in the largest (§6 of the phase-2 doc), the generator's pairing statistics deviate from the Poisson-gas survival model.",
   strong[sratio is an input to the pore target, not a post-hoc scaling knob] + " (row 18): lowering large-pore sratio shortens the inversion's model gaps → more predicted large throats → larger pores → generator overshoots the large band by " + $+15 ... +27$ + " pt. And " + strong[sr $> 1$ is unphysical] + " (fraction of pore surface covered by throats); values beyond 1 act as an over-connectivity lever and are invalid for production.",
    strong[The W1 recipe scales favorably with N] + " (row 19: 0.0063 @ 80k → 0.0042 @ 2e5), unlike the keep75 residual — because the W1 window's large band is not finite-size-limited at these N (" + $n_{"top, eligible"}$ + " 39 → 99).",
    strong[The multi-capillary degree correction shifts pore mass toward smaller classes] + " relative to the plain kernel (rows 22–23): on keep75 that degrades the large band, but on wide windows it corrects the low-" + $e$ + " large-band flooding — which is why the optimal step-1 factor moves from " + $e = 1.75$ + " (kernel) to " + $e = 1.25$ + " (budget/waterfill). The degree-only correction (budget) matches or beats the joint count+composition one (waterfill) on both windows tested: fixing only the count is sufficient so far.",
    strong[The phase-2 processing order interacts with the inversion] + " (rows 22–23): for the plain kernel, decreasing wins (giants claim partners first while budgets are fresh); for the budget/waterfill targets, random wins and increasing is worst under all inversions. " + `phase1_z = 0` + " (pass 1 off) is best at every inversion on this window.",
)

== 5. Open items

- #strong[Transition-region length correction] (`doc/capillary-length-phase2.md` §7) for the gamma inversion: empirical tabulation or NN-selection kernel refinement; both alternatives tested so far were rejected (rows 13, 14). Within the physical range (sr $<= 1$) the sratio level only shrinks the b1 deficit from $-14.3$ to ~$-12$ pt, so this remains the main model gap for keep75-style windows.
- #strong[`add_throats_by_distribution`]: implemented and validated but not wired anywhere; its remaining work (forced-mode large-pair pairing) is deprioritized in favor of the kernel inversion.
- #strong[Adopt kernel_budget for wide windows?] + " (row 23): tied with the plain kernel on JSD (0.0099 vs 0.0093, within MC noise) but with a near-exact large band (" + $+1.8$ + " pt vs " + $-5.2$ + ") and lower connectivity (9.3 vs 18.9); keep75 still favors the plain kernel — the choice is window-dependent, so no production switch yet.",
- #strong[RESOLVED 2026-11-20: notebook rewired to the asym50_95 recipe] + " (row 22). The old " + `RECIPE` + " dispatch and W1 knobs (" + `sort_by_radius` + ", " + `minc_search_factor` + ") are gone from the notebook; it now carries the two-phase parameters directly — window [50%, 95%], kernel e=1.75/sr=0.7, order=decreasing, phase1_z=0, p2f=2, maxc=1024 (N=250k). The W1 recipe of row 20 is no longer reproducible under \"surf\" and would need re-expression in two-phase parameters before reuse.",
