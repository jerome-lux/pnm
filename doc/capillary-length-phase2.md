# Capillary Length Model — Phase 2 Correction of Step 1

**Status.** Theory + numerical validation complete; implementation in `_corrected_throat_counts` pending. This document explains the "Phase 2" correction proposed for the pore-distribution inversion: replacing the empirical corrective exponent `e` of step 1 by a **per-class mean capillary length derived from the same geometry as the exclusion kernel** (no free parameter). It also answers two specific questions raised during the study:

1. What exactly does "incomplete gamma" (*gamma incomplète*) mean in this context, and why does it appear?
2. Justification of the formula for $\bar L_k$, the mean capillary length per class — including the **median**, which is a more robust statistic here because large-pore throats pile up at zero gap (contact).

Companion data: `examples/param_study_results_v3kernel_keep75_100k/Lbar_phase2/` (`lbar_per_class.csv`, `lbar_mean.png`, `lbar_median.png`). Full kernel derivation: [`pore-distribution-inversion.md`](pore-distribution-inversion.md).

## 1. What step 1 assumes today

Step 1 of both inversions (`_corrected_throat_counts` in `pnm/Stats.py`) converts the experimental per-bin throat **volume** fractions $v_k$ into a throat **count** distribution:

$$
\mathrm{cdist}_k = \frac{v_k}{\left(\pi\, \mathbb{E}[r^2]_k\right)^e}, \qquad \mathbb{E}[r^2]_k = m_k^2 + \tfrac{1}{12}\Delta r_k^2 ,
$$

where $m_k$ and $\Delta r_k$ are the bin center and width, and $e$ is the single empirical parameter of the pipeline (tuned values 1.25–1.75).

A throat in class $k$ has volume ≈ cross-section × length, so the true count fraction satisfies $n_k \propto v_k / \langle S\, L\rangle_k$. Step 1 is therefore **exact if and only if**

$$
\left(\pi\,\mathbb{E}[r^2]_k\right)^e = \langle S\,L\rangle_k \approx S_k\, \bar L_k
\quad\Longleftrightarrow\quad
\bar L_k \propto S_k^{\,e-1} \propto m_k^{\,2(e-1)} \qquad (S_k = \pi\,\mathbb{E}[r^2]_k \approx \pi m_k^2),
$$

where $\bar L_k$ is the mean capillary length of class $k$. Step 1 thus **silently assumes a power law for the mean length**: with $e=1.25$, $\bar L \propto m_k^{+0.5}$; with $e=1.75$, $\bar L \propto m_k^{+1.5}$. A single global exponent cannot represent a per-class function — and, as shown next, the assumed *sign* is wrong.

> **Size convention.** All size thresholds in this document refer to the throat bin center $m_k$ (a **radius**); diameters are twice that value.

## 2. What "Phase 2" is

The structural diagnostic of the keep75 study (flat JSD valley over $e\in[1.1,1.5]$, simultaneous deficits at both ends of the distribution for every $(e,\mathrm{sratio})$) led to a two-phase plan:

- **Phase 1** (control experiments): extend the grid to $\mathrm{sratio}\in\{0.25, 0.3\}$ with the *current* model and add damping/Anderson acceleration to the fixed point (which plateaus at low sratio). Expected outcome: no improvement — a negative result that would confirm the length hypothesis is the bottleneck.
- **Phase 2** (the actual correction): derive $\bar L_k$ from the kernel geometry itself — the same Poisson-gas model that produces $W_{ij}$ and $D_i$ — so that step 1 becomes parameter-free:

$$
\mathrm{cdist}_k = \frac{v_k}{S_k\, \bar L_k^{\,\mathrm{model}}},
$$

with the porosity entering naturally through the number density $\rho = 9\varphi/(4\pi\langle r^3\rangle)$ and the reach $D_i$ (exactly the intuition that motivated this work). This document covers Phase 2: the formula, why it involves an incomplete gamma function, and its validation against measured per-class lengths.

## 3. Measured per-class lengths — mean **and** median

**Setup.** Same pore set for both networks (keep75 window $[0.0069,\,1.0485]~\mu$m throat radius, $N=10^5$, kernel inversion at $e=1.25$, $\mathrm{sratio}=0.8$, porosity 0.35, seed 7 — the best keep75 configuration):

- **Network A** — `add_throats_by_surf_v3` without pass 1 (`large_radius_cutoff=None`): 773 234 throats.
- **Network B** — with pass 1 at the valley cutoff $0.15873~\mu$m (pore radius), $\mathrm{minc}=3$: 778 796 throats, of which 6 582 are large–large connections added by pass 1.

The class of a throat is the bin of its **smaller endpoint** (throat radius = $\min(r_1,r_2)/\mathrm{PRTRR}$). The "length" is the `length` edge attribute of `pore_network`, i.e. the **surface-to-surface gap**: center distance minus $r_1+r_2$, clamped at 0 (`PNM.py::_compute_auto_throat_length`).

**Power-law exponents** $\bar L \sim m_k^{\alpha}$ (log–log fit on per-class values):

| network | quantity | $\alpha$ for $m_k<0.03~\mu$m | $\alpha$ for $m_k>0.05~\mu$m |
|---|---|---|---|
| A (no pass 1) | mean gap | **+0.29** | **−0.99** |
| A (no pass 1) | median gap | +0.32 | −1.09 |
| B (pass 1 on) | mean gap | **+0.29** | **−0.82** |
| B (pass 1 on) | median gap | +0.32 | −1.02 |

Two regimes, and a sign flip:

- $m_k \lesssim 0.03~\mu$m: $\bar L$ is nearly flat / slightly *increasing* ($\alpha \approx +0.3$), at $\bar L \approx 0.6$–$0.85~\mu$m — small pores sit in interstices, so their throats are long regardless of throat size.
- $m_k \gtrsim 0.05~\mu$m: $\bar L$ **decreases** as $m_k^{-0.8\dots-1.1}$ (mean) and $m_k^{-1.0\dots-1.1}$ (median). The larger the throat, the *shorter* its capillary — large pores that end up connected are nearly in contact.

Step 1 assumes $\bar L \propto m_k^{+2(e-1)}$ with a **positive** exponent (+0.5 at $e=1.25$, +1.5 at $e=1.75$). The measured sign is the opposite, for both mean and median: **no value of $e$ can fix this**, which explains why no $(e,\mathrm{sratio})$ combination matches both ends of the distribution simultaneously.

**Effect of pass 1.** Pass 1 leaves small-throat lengths untouched (ratio B/A = 1.00 for $m_k<0.05~\mu$m) and multiplies large-class lengths by ~1.1–1.25 while multiplying their *counts* by 2–3 (e.g. class $m_k=0.16~\mu$m: count ×3.3, mean gap ×1.25). The decay exponent is therefore slightly weaker in B than in A; the qualitative picture is identical.

## 4. "Incomplete gamma" — what it means exactly

The **upper incomplete gamma function** is defined by

$$
\Gamma(s, x) = \int_x^{\infty} t^{s-1}\, e^{-t}\, dt ,
$$

i.e. the same integral as the (complete) gamma function $\Gamma(s) = \int_0^\infty t^{s-1}e^{-t}dt$, but integrated only from $x$ to $\infty$. It is called *incomplete* precisely because its lower limit is a positive number rather than 0: it measures the "remaining tail" of the gamma integral beyond $x$. (The complementary **lower** incomplete gamma is $\gamma(s,x) = \int_0^x t^{s-1}e^{-t}dt$, with $\Gamma(s,x) + \gamma(s,x) = \Gamma(s)$.) In `scipy.special`: `gammaincc(s, x)` returns the regularized upper form $\Gamma(s,x)/\Gamma(s)$.

**Why it appears here.** The exclusion kernel models the partner of a focal pore in class $i$ as drawn from a Poisson gas with number density $\rho$, survival probability $\exp(-(s/D_i)^3)$ (budget exhaustion, §5.2–5.3 of the main document), and **hard-core exclusion** $s \ge a_{ij} = r_i + r_j$. The partner-distance density is therefore

$$
f(s) \propto s^2\, e^{-(s/D_i)^3}, \qquad s \in [a_{ij},\, \infty).
$$

The *count* of partners (which gives $W_{ij}$) needs only $\int_a^\infty s^2 e^{-(s/D_i)^3}ds$, which is elementary after $u=(s/D_i)^3$ because the integrand becomes a pure exponential. The *length moment*, however, carries one extra power of $s$:

$$
\int_{a}^{\infty} s^3\, e^{-(s/D_i)^3}\, ds .
$$

With $u = (s/D_i)^3$ one has $s^3\,ds = \tfrac{D_i^4}{3}\, u^{1/3}\, du$, so the integral becomes $\tfrac{D_i^4}{3}\int_{x_{ij}}^{\infty} u^{1/3} e^{-u}\, du$ — and **that remaining integral is exactly the upper incomplete gamma function** $\Gamma(\tfrac43, x_{ij})$, with

$$
x_{ij} = \left(\frac{r_i + r_j}{D_i}\right)^3 > 0 .
$$

The lower limit $x_{ij}$ is positive *because of hard-core exclusion*: two non-overlapping pores cannot have their centers closer than $r_i+r_j$. If partners could sit at any distance from the focal pore ($a=0$, no exclusion), then $x_{ij}=0$ and the moment would collapse to the **complete** gamma $\Gamma(4/3)$ — which is why the incomplete version is needed: it is the gamma integral with its lower part cut off by the exclusion distance.

## 5. Derivation of $\bar L_k$ (mean length per class)

All quantities below are those of the kernel model at the fixed point ($x$, $D_i$, $K(i\to j)$ as in `invert_pore_distribution_kernel`; pore bin mid-radii $r_k = \mathrm{PRTRR}\cdot m_k$).

### 5.1 Partner count and length moment for a directed pair $(i \to j)$

For a focal pore in class $i$ (reach $D_i$) and partner class $j$, with $a_{ij} = r_i + r_j$:

**Count.** $\displaystyle N_{ij} = 4\pi\rho \int_a^\infty s^2 e^{-(s/D_i)^3}ds = \frac{4}{3}\pi\rho D_i^3\, e^{-x_{ij}}$ — this is the kernel weight (up to factors common to all $j$).

**Length moment.** $\displaystyle M_{ij} = 4\pi\rho \int_a^\infty s^3 e^{-(s/D_i)^3}ds$. Substituting $u=(s/D_i)^3$:

$$
M_{ij} = 4\pi\rho\, \frac{D_i^4}{3}\int_{x_{ij}}^{\infty} u^{1/3} e^{-u}\, du
= \frac{4}{3}\,\pi\rho\, D_i^4\; \Gamma\!\left(\tfrac{4}{3},\, x_{ij}\right).
$$

**Mean center distance.** Dividing the two moments (the $4\pi\rho$ factors cancel):

$$
\boxed{\;\langle s\rangle_{ij} = D_i\; e^{x_{ij}}\, \Gamma\!\left(\tfrac{4}{3},\, x_{ij}\right)\;}
\qquad\text{and the mean gap is } \bar L_{ij} = \langle s\rangle_{ij} - r_i - r_j .
$$

**Limits (sanity checks).**

- $x_{ij}\to 0$ (exclusion negligible, reach large): $\Gamma(4/3,0)=\Gamma(4/3)\approx 0.893$, so $\langle s\rangle \to 0.893\,D_i$. The partner distance is a pure Weibull-3 draw from the origin — mean $0.893\,D_i$.
- $x_{ij}\to\infty$ (exclusion dominates): using $\Gamma(s,x) \sim x^{s-1}e^{-x}(1 + (s-1)/x + \cdots)$, one gets $\langle s\rangle \approx a_{ij}(1 + 1/(3x_{ij})) \to a_{ij}$: **all partners sit at contact**, gap → 0. This is exactly the "giants nearly touching" behavior measured in §3.

The formula interpolates between these two regimes with crossover at $x_{ij}\sim O(1)$, and $\langle s\rangle_{ij} \ge a_{ij}$ holds identically (the integral's lower limit enforces it).

### 5.2 Median center distance (closed form)

The CDF of the partner distance is

$$
F(s) = \frac{\int_a^s t^2 e^{-(t/D_i)^3}dt}{\int_a^\infty t^2 e^{-(t/D_i)^3}dt}
= 1 - \exp\!\Big(-\big[(s/D_i)^3 - x_{ij}\big]\Big),
$$

so the median solves $F(s_{\mathrm{med}}) = 1/2$:

$$
\boxed{\;s_{\mathrm{med},ij} = \left(a_{ij}^3 + D_i^3 \ln 2\right)^{1/3}\;}
\qquad\text{(median gap } = s_{\mathrm{med},ij} - a_{ij}\text{).}
$$

No special functions needed: the median is the contact distance $a_{ij}$ "inflated" by an additive volume term $D_i^3\ln 2$. In the two limits it reduces to $0.856\,D_i$ (no exclusion; note $(\ln 2)^{1/3}\approx 0.856 < 0.893$, as expected for a right-skewed density) and to $a_{ij}(1 + \tfrac{\ln 2}{3x_{ij}})$ (contact). The median is the more robust statistic in the large classes, where the gap distribution has a pile-up at 0 plus a long tail.

### 5.3 Aggregation per throat class

A throat of class $k$ (smaller endpoint in bin $k$) can be produced by two directed mechanisms, exactly as in the throat identity (§5.4 of the main document): focal pore in class $k$ pointing to an equal-or-larger partner ($j \ge k$), or a larger focal pore ($i > k$) pointing down at class $k$. Weighting each directed connection by its production rate $x_i\,K(i\to j)$:

$$
\bar L_k = \frac{
x_k \sum_{j\ge k} K(k\to j)\,\bar L_{kj}
+ \sum_{i>k} x_i\, K(i\to k)\,\bar L_{ik}
}{
\underbrace{x_k \sum_{j\ge k} K(k\to j) + \sum_{i>k} x_i K(i\to k)}_{=\;\mathrm{cdist}_k\;\text{(throat identity)}}
},
$$

with $\bar L_{ij}$ the mean (or median, for the robust variant) gap of §5.1–5.2. The denominator is *identically* $\mathrm{cdist}_k$ — verified numerically to $1.4\times 10^{-13}$ on the keep75 fixed point — so this is a genuine average over the throats that step 1 is trying to count, not an additional assumption.

**Note on the moment scaling.** The length moment scales as $D_i^4\,\Gamma(4/3, x_{ij})$ (one power of $s$ beyond the count integral). An earlier proportionality sketch in the study notes wrote "$\propto D_i^6\,\Gamma(5/3,x)$"; that was a loose statement — the exact derivation above gives $D_i^4$ and shape $4/3$. The scaling conclusions are unchanged.

## 6. Validation against the measured networks

Model intermediates at the keep75 fixed point ($e=1.25$, $\mathrm{sratio}=0.8$): $\rho = 10.52~\mu$m$^{-3}$, reach $D_i \in [0.66,\, 3.81]~\mu$m (grows with pore size).

| class | $m_k$ (throat radius, µm) | $n_B$ | measured mean gap | measured median gap | model mean gap | model median gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0077 | 189 439 | 0.610 | 0.616 | **0.605** | 0.599 |
| 2 | 0.0119 | 198 024 | 0.649 | 0.662 | **0.648** | 0.641 |
| 4 | 0.0182 | 37 504 | 0.780 | 0.803 | **0.771** | 0.760 |
| 6 | 0.0283 | 16 668 | 0.851 | 0.898 | **0.875** | 0.862 |
| 8 | 0.0432 | 7 819 | 0.808 | 0.861 | 0.980 | 0.962 |
| 10 | 0.0680 | 2 747 | 0.688 | 0.740 | 1.095 | 1.068 |
| 13 | 0.1280 | 1 150 | 0.521 | 0.472 | 1.210 | 1.160 |
| 16 | 0.2475 | 424 | 0.356 | 0.306 | 1.203 | 1.120 |
| 19 | 0.4741 | 148 | 0.177 | 0.103 | 1.082 | 0.959 |
| 22 | 0.9318 | 39 | 0.074 | 0.046 | 0.913 | 0.758 |

(Full per-class table: `Lbar_phase2/lbar_per_class.csv`; plots: `lbar_mean.png`, `lbar_median.png`.)

**Where the model works.** For $m_k \lesssim 0.03~\mu$m (classes 0–6, ~95 % of throat *count*), the derived mean gap matches the measurement within ≤3 % and the median within ~5 %. The budget-reach mechanism ($D_i$ from porosity + sratio) captures the interstitial spacing that sets small-throat lengths — with no fitted parameter. The agreement degrades smoothly through a transition region at $m_k \approx 0.03$–$0.05~\mu$m (+10 % at class 7, +21 % at class 8) before breaking down in the large classes: there the measured curve has already turned over (it peaks near $m_k=0.028~\mu$m), while the model keeps rising toward its slow decay.

**Where it fails, and why.** For $m_k \gtrsim 0.1~\mu$m the model *overpredicts* the gap by a factor of ~2 (class 13) up to ~12–16 (top class). The origin is structural: pass 2 scans each pore's candidates in **increasing distance order** and accepts the first affordable one, and pass 1 explicitly connects large pores to their *nearest* large neighbors. Large pores therefore connect at near-contact distances — a **nearest-neighbor selection effect**. The Poisson-gas survival weight $e^{-(s/D_i)^3}$ is a mean-field hazard that washes out this ordering: it assigns non-negligible probability to partners far beyond the nearest one, inflating $\langle s\rangle_{ij}$. (A secondary contributor: the real packing is hard-sphere — centers cannot approach closer than contact — while the gas model allows arbitrarily close points; for small pores the budget cutoff dominates and this matters little.)

**Volume caveat.** The well-validated classes ($m_k \lesssim 0.03~\mu$m) carry ~95 % of throat *count* but only ~36 % of generated throat *volume*; the transition and large classes (where the length is overpredicted) carry most of the remaining volume, including the >0.5 µm band that drives the JSD. A production Phase 2 must therefore handle the large-class regime explicitly — see §7.

## 7. Consequences for step 1 and next steps

With $\bar L_k$ derived, step 1 becomes parameter-free in principle:

$$
\mathrm{cdist}_k = \frac{v_k}{S_k\,\bar L_k^{\,\mathrm{model}}},
\qquad S_k = \pi\left(m_k^2 + \tfrac{1}{12}\Delta r_k^2\right),
$$

and the porosity enters through $\rho$ and $D_i$ as intended. The empirical exponent $e$ disappears (or survives only as a residual correction on top of a correct baseline).

Known limitation to close before re-running the keep75 study:

1. **Large-class length bias.** Options, in increasing order of ambition:
    - *Hybrid*: use the derived moment where validated ($m_k \lesssim 0.03~\mu$m) and a contact-limit correction $\bar L_{ij} \approx$ measured NN gap for larger classes (tested as a power-law tail — §7.2, rejected);
   - *Empirical tabulation*: measure $\bar L_k$ once on a reference network of the same recipe (pore set + sratio + cutoff) and use it as a lookup — parameter-free given the recipe, but no longer purely predictive;
   - *Refined kernel*: replace the survival weight by a nearest-neighbor selection model for large focal pores (partner = first affordable candidate in distance order), which is what the generator actually does.
2. **Pass 1 forced connections** are not part of the Poisson-gas model at all; their contribution to $\bar L_k$ is measurable now that both networks exist (§3) and can be added as a separate correction term.

### 7.1 Global physical-density reach (tested 2026-11-16 — rejected)

A natural "more physical" alternative to the budget reach was tested: replace $D_i$ in step 1's
gap matrix by a single global reach from the **physical** number density of the packing,
$\rho_{\mathrm{phys}} = 3\varphi/(4\pi\langle r^3\rangle)$, i.e.

$$
D \;=\; \left(\frac{\langle r^3\rangle_x}{\varphi}\right)^{1/3}
\qquad (\text{radius of a sphere holding on average one pore}),
$$

recomputed at every iterate through $\langle r^3\rangle_x$ (implemented as `length_stat="packing"`;
the kernel geometry $W$, $K$ keeps the budget reach, so only step 1 changes). The fixed point still
converges cleanly, but the variant is **decisively worse** on keep75 (v3 pass 1, cutoff 0.5 µm,
N=1e5, sratio 0.6–0.8):

| sratio | JSD ("mean") | JSD ("packing") | b1 % (target 25.7) | large band % (target 19.7) |
|---:|---:|---:|---:|---:|
| 0.6 | 0.0393 | **0.2024** | 8.3 → **0.7** | 32.2 → **62.0** |
| 0.7 | 0.0302 | **0.1925** | 9.7 → **0.8** | 28.1 → **59.0** |
| 0.8 | 0.0234 | **0.1852** | 11.3 → **0.9** | 23.5 → **57.2** |

Mechanism: the global $D$ (~0.56 µm at the "mean" fixed point) is far below the budget reach of
the largest classes ($D_i$ grows to ~3.8 µm), so large-class step-1 lengths collapse, their count
fractions $v_k/(S_k\bar L_k)$ balloon, and the pore target shifts strongly toward larger pores —
the fixed-point $\langle r^3\rangle_x$ grows ~5× (0.06 → 0.32 µm³), which in turn inflates its own
$D$ to ~0.97 µm. The generator then amplifies the shift: pass-1 giant–giant connections plus pass 2
empty the smallest band almost completely and flood the >0.5 µm band (+30…+34 pt).

**Conclusion.** The class dependence of the budget reach is *load-bearing*, not a calibration
artifact: it acts as an implicit large-throat penalty that keeps small-pore mass in the target, and
no global physical length scale can replace it. What fails is specifically the replacement by a
class-independent reach; the per-class structure itself — including the effective density prefactor
$\rho = 9\varphi/(4\pi\langle r^3\rangle)$ of §5, which fits measured small-class gaps within ≤3 % —
must be preserved. The next model fix remains item 1 above: a large-class length correction that
keeps the per-class structure.

### 7.2 Hybrid step-1 length (tested 2026-11-16 — rejected)

The first concrete attempt at item 1 was implemented as `pnm.invert_pore_distribution_kernel_gamma_hybrid`
(wired into `examples/param_study.py` as `--inversion kernel_gamma_hybrid --gamma-tail-exp β`): keep the
derived incomplete-gamma profile where it is validated ($m_k \le m_{\mathrm{switch}} = 0.03~\mu$m) and, beyond
the anchor class, replace the turnover by a power-law continuation

$$
\bar L_k \;=\; \max\!\Big(\bar L_k^{\,\mathrm{derived}},\;\bar L_{\mathrm{anchor}}\,(m_k/m_{\mathrm{anchor}})^{\beta}\Big),
\qquad k > \mathrm{anchor},
$$

continuous and monotone beyond the anchor (the derived values are flat/declining there, so the tail takes
over). The motivation: the pure derived profile turns over at ~0.13 µm and declines toward contact, which
makes step 1 demand too many large throats — the gamma-mean pore target carries ~4 pt more volume in pores
≥ 1 µm (93.8 % vs 90.2 %) than the JSD-optimal kernel recipe $e=1.25$, and the generated network overshoots
the >0.5 µm band by +3.8 pt at sratio 0.8. The default $\beta = 0.5$ reproduces the effective steepness of
that recipe, whose step-1 length grows as $m^{2(e-1)} = m^{0.5}$.

**Results** (keep75 window, v3 pass 1 cutoff 0.5 µm, N=1e5, seed 7; fixed point converges cleanly at every
point, residual < 1e-3):

| variant | sratio / minc | JSD | b1 % (target 25.7) | b3 % (target 10.5) | large band % (target 19.7) |
|---|---|---:|---:|---:|---:|
| hybrid β=0.3 | 0.7 / 2 | **0.0234** | 12.5 (−13.2) | 18.7 (+8.2) | 19.0 (**−0.7**) |
| hybrid β=0.5 | 0.7 / 2 | 0.0243 | 13.8 (−11.9) | 19.2 (+8.7) | 17.6 (−2.1) |
| hybrid β=0.7 | 0.7 / 2 | 0.0266 | 15.5 (−10.2) | 19.9 (+9.4) | 16.8 (−2.9) |
| gamma-mean (ref.) | 0.8 / 2 | 0.0234 | 11.3 (−14.4) | 16.4 (+5.9) | 23.5 (+3.8) |
| kernel $e=1.25$ (ref.) | 0.7 / 2 | **0.0195** | 15.9 (−9.8) | 17.8 (+7.3) | 18.2 (−1.5) |

Full β=0.5 grid: sratio ∈ {0.6, 0.7, 0.8} × minc ∈ {1, 2} — best JSD 0.0243 at sratio 0.7/minc 2;
minc = 2 wins at every sratio (as for the other inversions). The β scan is monotone: JSD(β) decreases as
the tail flattens, with its minimum at or below β = 0.3 — i.e. toward the gamma-mean limit, not away from it.

**Reading.** Two facts stand out:

1. **The structural residual survives unchanged.** Every hybrid variant shows the same b1-deficit +
   b3-excess pair (+8…+9 pt in b3) that sets the JSD floor of kernel $e=1.25$ (−9.8 / +7.3). β trades b1
   against the large band exactly like $e$ does in the kernel inversion — it is a rebalancing lever, not a
   fix: at β = 0.3 the large band is near-exact (−0.7 pt) but b1 worsens to −13.2 pt.
2. **The "validated" small-class profile is not optimal.** The only difference between hybrid and kernel
   $e=1.25$ below $m_{\mathrm{switch}}$ is the step-1 length: the derived profile (which matches measured
   per-class gaps within ≤3 %, §6) vs the unvalidated $m^{0.5}$ extrapolation of $e=1.25$. The latter puts
   more mass in small pores (count fraction < 0.05 µm: 86.4 % vs 84.9 %) and yields a better b1 (−9.8 vs
   −11.9…−13.2 pt). Matching the *measured* gap lengths is therefore not sufficient for reproducing the
   throat distribution: in the smallest classes, as in the largest (§6), the generator's actual pairing
   statistics deviate from the Poisson-gas survival model (nearest-neighbor/contact selection effects).

**Conclusion.** The hybrid does not beat kernel $e=1.25$ at any tested β (best 0.0234 vs 0.0195) and only
ties gamma-mean; it is rejected as a reference recipe, and `invert_pore_distribution_kernel_gamma_hybrid`
is kept for research only (same status as `invert_pore_distribution_kernel_experimental`). The diagnosis of
the remaining lever sharpens: the b3 excess co-locates with the §6 $\bar L_k$ overprediction at 0.03–0.05 µm,
and the b1 deficit tracks small-class length behaviour — a transition-region correction must act on both ends
simultaneously (item 1, "empirical tabulation" or "refined kernel"), not only on the large classes.

## 8. Reproducibility

- Measurement + prediction script: `/tmp/kilo/measure_Lbar_phase2.py` (re-runnable; per-class stats cached in `Lbar_phase2/stats_cache.npz`).
- Pore set: `examples/param_study_results_v3kernel_keep75_100k/pores/pores_e1.25_shift+0_n100000_phi0.35_seed7_wkeep75_kern_sr0.8.pnm` (keep75 window, kernel inversion $e=1.25$, $\mathrm{sratio}=0.8$, $N=10^5$, seed 7).
- Throat generation: `pnm.add_throats_by_surf_v3(ratio=0.8, PRTRR=2, minc=3, large_radius_cutoff=None | 0.15873, border_correction=True, seed=7)` — identical to the keep75 study runs (JSD 0.0341 for network B).
- Model: `pnm.invert_pore_distribution_kernel(volumes, midradii, deltaR, 1.25, PRTRR=2.0, porosity=0.35, sratio=0.8)` on the keep75-windowed GM1 data; moments as in §5 (upper incomplete gamma via `scipy.special.gammaincc`, asymptotic fallback for large $x$).
