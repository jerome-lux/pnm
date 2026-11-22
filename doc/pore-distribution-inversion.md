# From Capillary to Pore Size Distribution: Inversion Theory

This document derives the two pore-distribution inversions implemented in `pnm/Stats.py`:

- **`invert_pore_distribution_independent`** — the classical model of `examples/theory.typ`. It assumes that the two pores connected by a throat are drawn *independently* from the pore distribution.
- **`invert_pore_distribution_kernel`** — the corrected model. The partner of a focal pore is drawn from a size-conditioned *exclusion kernel* derived from the surface-budget mechanics of the generator itself (`pnm.Transformation.add_throats_by_surf`). It introduces no free parameter.

Both functions take the same inputs (experimental MIP throat distribution + corrective factor $e$) and return the same output contract: `(cdist, pdist)`, where `pdist` is a pore **count** distribution (one probability per bin, summing to 1). To use it, feed `pdist` — together with the full pore bin edges (`throat_radii * PRTRR`, i.e. n_bins + 1 values) — into `pnm.inverse_transform_sampling_from_raw_data`.

## Notation

All radii are in µm. A "class" is a bin index; we use $i, j, k$ interchangeably as class indices over $\{0, \dots, n-1\}$, with classes ordered by increasing radius (class 0 = smallest). Every symbol used below is defined here or at its first appearance.

| Symbol | Meaning |
|--------|---------|
| $n$ | number of throat bins (= number of pore bins) |
| $N$ | total number of pores in the network (distinct from $n$, the number of bins) |
| ${r^c_k}$ | center radius of throat bin $k$ — `midradii[k]` |
| $\Delta r_k$ | width of throat bin $k$: the bin spans $[{r^c_k} - \tfrac12\Delta r_k,\; {r^c_k} + \tfrac12\Delta r_k]$ — `deltaR[k]` |
| $v_k$ | experimental intruded volume fraction in throat bin $k$, with $\sum_k v_k = 1$ — `volumes[k]` |
| $e$ | empirical corrective exponent, the **only free parameter** of the pipeline (tuned values ≈ 1.5–2) |
| $\mathrm{PRTRR}$ | pore-to-throat radius ratio: a throat between pores of radii $r_i, r_j$ has radius $\min(r_i, r_j)/\mathrm{PRTRR}$. The pipeline uses 2. |
| $x_k$ | **unknown** of the inversion: probability that a randomly drawn pore falls in pore bin $k$, with $\sum_k x_k = 1$ (pore *count* distribution) |
| $r_k$ | center radius of pore bin $k$: $r_k = \mathrm{PRTRR}\, {r^c_k}$ (pores are PRTRR times larger than throats) |
| $\varphi$ | target pore-body porosity of the packing that will be generated (pipeline: 0.35) |
| $\rho$ | number density of pores in the box ($\mu\mathrm{m}^{-3}$), derived from $x$ and $\varphi$ (§5.1) |
| $\langle r^3\rangle$ | mean cubed pore radius under $x$: $\sum_k x_k r_k^3$ |
| $s_{\mathrm{ratio}}$ | surface coverage ratio: fraction of a pore's surface that may be covered by throats (pipeline: 0.8) |
| $B_i$ | surface budget of a focal pore in class $i$: $B_i = s_{\mathrm{ratio}}\, 4\pi r_i^2$ ($\mu\mathrm{m}^2$) |
| $\langle S_t\rangle_i$ | mean cross-section of a throat from a class-$i$ pore to a partner drawn from $x$ (§5.2) |
| $\ell$ | center-to-center distance between two pores ($\mu\mathrm{m}$); the running length variable in §5.2–5.3 |
| $\ell_i$ | effective connection reach of a class-$i$ pore ($\mu\mathrm{m}$): the value of $\ell$ at which its budget runs out, derived from its budget (§5.2) |
| $W_{ij}$ | unnormalized pair weight between classes $i$ and $j$ (§5.3) |
| $K(i \to j)$ | probability that the partner of a focal pore in class $i$ falls in class $j$; each row sums to 1 (§5.3) |
| $\mathrm{cdist}_k$ | corrected throat *count* distribution — the target of the inversion, with $\sum_k \mathrm{cdist}_k = 1$ (§2) |

## 1. Problem setup

An experimental mercury porosimetry (MIP) curve is binned into $n$ throat-radius classes: bin centers ${r^c_k}$, widths $\Delta r_k$, and normalized intruded volume fractions $v_k$.

The generator builds a network of pores whose radii are drawn from an unknown count distribution $x$. A throat connecting two pores of radii $r_i, r_j$ has radius

$$
r_t = \frac{\min(r_i, r_j)}{\mathrm{PRTRR}},
$$

so **the class of a throat is set by its smaller endpoint**: a large pore connected to a small one produces a *small* throat.

**Inversion problem.** Find the pore count distribution $x$ such that when the generator connects the pores, the resulting throat volume distribution matches the experimental $(v_k)$.

## 2. Step 1 — from volumes to counts (corrective factor $e$)

A MIP bin reports how much *volume* was intruded through throats of a given size range; the inversion needs how many *throats* there are in each class. A throat in bin $k$ has radius $\approx {r^c_k}$, hence volume ≈ cross-section × length, so dividing the per-bin volume fraction by a characteristic cross-section converts it into a count fraction.

For a radius drawn uniformly from bin $k$ (spanning $[{r^c_k} - \tfrac12\Delta r_k,\; {r^c_k} + \tfrac12\Delta r_k]$), the mean squared radius is

$$
\mathbb{E}[r^2]_k = {r^c_k}^2 + \frac{\Delta {r^c_k}^2}{12},
$$

because a uniform distribution on an interval of width $\Delta r_k$ has variance $\Delta {r^c_k}^2/12$, and $\mathbb{E}[r^2] = \mathrm{Var}(r) + (\mathbb{E}[r])^2$.  
If $\mathbb{E}[L_k]$ is the mean length in the bin, the distribution of throats by number is:

$$
\mathrm{cdist}_k \propto \frac{v_k}{\left(\pi \left({r^c_k}^2 + \Delta {r^c_k}^2/12\right)\right) \mathbb{E}[L_k]},
\qquad \text{then } \mathrm{cdist} = \sum_p \mathrm{cdist}_p .
\tag{1}
$$

The term $\mathbb{E}[L_k]$ is approximated as $r_k^\alpha$, giving the final expression:
$$
\mathrm{cdist}_k \propto \frac{v_k}{\left(\pi \left({r^c_k}^2 + \Delta {r^c_k}^2/12\right)\right)^e},
\tag{2}
$$

The exponent $e$ is the **only empirical parameter** of the whole pipeline: it absorbs the residual size dependence of throat length and other second-order effects ($e = 1$ means pure cross-section correction; tuned values are ≈ 1.5–2). Equation (1) is shared verbatim by both inversions (private helper `_corrected_throat_counts`).

## 3. The independent-pairing model (classical inversion)

**Assumption.** The two pores connected by a throat are drawn *independently* from $x$ — knowing the size of one endpoint tells you nothing about the other.

A throat falls in class $k$ iff its smaller endpoint is in class $k$. There are two mutually exclusive ways this can happen:

- both endpoints are in $k$: probability $x_k \cdot x_k$;
- one endpoint is in $k$ and the other in a larger class $j > k$: probability $2\, x_k x_j$ (factor 2: either of the two drawn pores may be the one that falls in class $k$; it is then automatically the smaller one since $j > k$).

Summing over all larger classes gives

$$
\mathrm{cdist}_k = x_k^2 + 2\, x_k \sum_{j>k} x_j .
\tag{2}
$$

Writing $T_k := \sum_{j>k} x_j$ (the tail of $x$ above class $k$), equation (2) is a quadratic in the unknown $x_k$ whose coefficients involve only *larger* classes:

$$
x_k^2 + 2 T_k x_k - \mathrm{cdist}_k = 0 .
$$

The system is therefore solved **exactly by backward substitution**, from the largest class down, each step being a single quadratic with known coefficients (positive root):

$$
x_{n-1} = \sqrt{\mathrm{cdist}_{n-1}},
\qquad
x_k = -T_k + \sqrt{T_k^2 + \mathrm{cdist}_k}.
\tag{3}
$$

No iteration is needed: (3) is the exact solution of (2). This is `invert_pore_distribution_independent`.

## 4. Why independence fails: the spatial pairing mechanics

The actual generator does **not** pair pores randomly. In pass 1 of `add_throats_by_surf`:

1. each pore $i$ (the *focal* pore) has a surface budget $B_i = s_{\mathrm{ratio}}\, 4\pi r_i^2$;
2. its candidates are scanned in **increasing** surface-to-surface distance;
3. a candidate pair $(i, j)$ is accepted only if both pores' budgets can still absorb the throat cross-section $\pi (\min(r_i, r_j)/\mathrm{PRTRR})^2$, after which that amount is deducted from both budgets.

Two consequences follow:

1. **Disassortative pairing.** A large focal pore spends its budget on cheap nearby small throats long before a distant large partner is reached in the distance ordering (a throat to a small partner costs little cross-section, so many of them fit within $B_i$). The partner-size distribution of a class-$i$ pore is therefore *not* $x$: it is size-conditioned by proximity and cost, strongly depleted at the top for every $i$.
2. **Large-throat deficit.** A large–large throat requires two large pores to be close — rare in a random placement (the expected number of close large-large pairs scales linearly with $N$). The generated network systematically under-produces the largest throat band (~26 % experimental vs < 3 % generated at N = 80k), whatever $e$ is tuned to.

The independent inversion returns the $x$ that would reproduce the experiment **only if pairing were random**. Fed into a spatial generator, the bias survives. The kernel below removes the assumption instead of patching its consequence.

## 5. Derivation of the exclusion kernel

The goal: replace the partner distribution "$x$" in (2) by a partner distribution $K(i \to j)$ that reflects what the generator actually does, using only quantities already fixed by the model ($\varphi$, $s_{\mathrm{ratio}}$, $\mathrm{PRTRR}$).

### 5.1 Number density from porosity

The kdtree packers rescale the box so that total pore volume / box volume = target porosity $\varphi$. With $N$ pores of mean cubed radius $\langle r^3\rangle = \sum_k x_k r_k^3$, the box volume is $V_{\mathrm{box}} = N (4/3)\pi \langle r^3\rangle / \varphi$, hence the number density

$$
\rho := \frac{N}{V_{\mathrm{box}}} = \frac{9\, \varphi}{4\pi\, \langle r^3\rangle}.
\tag{4}
$$

Note that $\rho$ depends on $x$, one reason the inversion is a fixed point (§6).

### 5.2 Connection reach from the surface budget

For a focal pore in class $i$ (radius $r_i$, budget $B_i = s_{\mathrm{ratio}}\, 4\pi r_i^2$), let $\langle S_t\rangle_i$ be the mean cross-section of a throat to a partner drawn from $x$: averaging over the partner's class $j$,

$$
\langle S_t\rangle_i = \frac{\pi}{\mathrm{PRTRR}^2}\sum_j x_j\, \min(r_i, r_j)^2 .
\tag{5}
$$

(The $\min$ encodes the throat-radius rule: the smaller endpoint sets the size.)

In a homogeneous packing, the expected number of neighbors within center distance $\ell$ is $(4/3)\pi \ell^3 \rho$. If pore $i$ accepted every such neighbor, the cumulative cross-section it would have paid up to distance $\ell$ is

$$
C(\ell) \approx (4/3)\, \pi\, \ell^3\, \rho\, \langle S_t\rangle_i .
\tag{6}
$$

The budget runs out when $C(\ell_i) = B_i$, which defines the **effective connection reach** of class $i$:

$$
\ell_i^3 = \frac{B_i}{(4/3)\, \pi\, \rho\, \langle S_t\rangle_i}.
\tag{7}
$$

A partner at center distance $\ell$ is therefore connected with survival probability

$$
P(\ell) \approx \exp\!\left(-\left(\frac{\ell}{\ell_i}\right)^3\right),
\qquad \text{i.e. } P(\ell) \approx \exp\big(-C(\ell)/B_i\big).
\tag{8}
$$

Equation (8) is a mean-field idealization of the sequential scan: $C(\ell)/B_i$ plays the role of an accumulated hazard, and a candidate at distance $\ell$ survives only if the budget spent on all closer candidates left room for it.

### 5.3 Partner counting with hard-core exclusion

The number density of class-$j$ pores is $\rho x_j$ (a fraction $x_j$ of all pores). The expected number of *connected* class-$j$ partners around a focal pore in class $i$ is the shell integral

$$
\int_{r_i + r_j}^{\infty} \underbrace{4\pi \ell^2\, \rho x_j}_{\text{class-}j\text{ pores in }[\ell,\, \ell+d\ell]}
\exp\!\left(-\left(\frac{\ell}{\ell_i}\right)^3\right) d\ell .
\tag{9}
$$

The lower limit $r_i + r_j$ is **hard-core exclusion**: two non-overlapping spheres of radii $r_i, r_j$ have center distance at least $r_i + r_j$ (surface gap $\ge 0$).

With the substitution $u = (\ell/\ell_i)^3$, one has $\ell^2\, d\ell = \ell_i^3 du / 3$ and the lower limit becomes $a_{ij}^3$ with $a_{ij} := (r_i + r_j)/\ell_i$, so that $\int_{a_{ij}^3}^{\infty} e^{-u}\, du = e^{-a_{ij}^3}$ and (9) evaluates in closed form to

$$
\rho x_j\, \tfrac{4}{3}\pi \ell_i^3\, \exp\!\left(-\left(\frac{r_i + r_j}{\ell_i}\right)^3\right).
\tag{10}
$$

The factors common to all $j$ (the $\rho$ and $(4/3)\pi \ell_i^3$) do not affect the *relative* weights of partner classes, so we define

$$
W_{ij} := \exp\!\left(-\left(\frac{r_i + r_j}{\ell_i}\right)^3\right)
$$

and normalize each row to a probability. The **partner-size distribution of a focal pore in class $i$** is

$$
K(i \to j) = \frac{x_j\, W_{ij}}{\sum_p x_p\, W_{ip}},
\qquad \text{so that } \sum_j K(i \to j) = 1 .
\tag{11}
$$

$K$ depends on $x$ through its rows ($\ell_i$ via (5)–(7)), their normalization, and $\rho(x)$ — the inversion is a fixed point (§6).

### 5.4 The throat identity

We now count throats from **directed** connections. Let $N_i \approx N x_i$ be the number of pores in class $i$, and let $z_i$ be the mean degree (number of throats) of a class-$i$ pore. Focal pores in class $i$ make $N_i z_i$ directed connections, distributed over partner classes according to row $K(i \to \cdot)$; hence the number of directed connections from class $i$ to class $j$ is

$$
\nu_{ij} = N_i\, z_i\, K(i \to j).
$$

Every undirected edge contributes exactly two directed connections (one per endpoint), and **both** have the same smaller-endpoint class. Therefore the fraction of directed connections whose smaller endpoint lies in class $k$ equals the fraction of throats in class $k$. A directed connection has its smaller endpoint in class $k$ in exactly two cases:

- the focal pore is in class $k$ and points to an equal or larger partner ($j \ge k$);
- the focal pore is in a larger class $j > k$ and points down at partner class $k$.

The total number of directed connections is $\sum_{i,j} \nu_{ij} = Nz \sum_i x_i = Nz$ when all classes share one mean degree $z$ (the assumption discussed below). Dividing the two cases by this total gives

$$
\mathrm{cdist}_k = x_k \left[ K(k \to k) + \sum_{j>k} K(k \to j) \right] + \sum_{j>k} x_j\, K(j \to k).
\tag{12}
$$

Two properties worth noting:

- **(12) sums to 1 identically** in $x$: the two cases above partition all directed connections (diagonal, focal-smaller, focal-larger), so no choice of $x$ can make the target inconsistent.
- When $K(i \to j) = x_j$ (independence), (12) reduces exactly to (2).

*Where the equal-degree assumption enters.* The factor $z_i$ cancels in (12) only if it is class-independent. It is **exactly** constant when a typical partner of pore $i$ is larger than it: then $\sum_j x_j\, \min(r_i, r_j)^2 \approx r_i^2$ and

$$
z_i = \frac{B_i}{\langle S_t\rangle_i}
= \frac{s_{\mathrm{ratio}}\, 4\pi r_i^2}{\pi r_i^2 / \mathrm{PRTRR}^2}
= 4\, s_{\mathrm{ratio}}\, \mathrm{PRTRR}^2,
$$

independent of $r_i$. For the largest classes (whose partners are mostly smaller), $\sum_j x_j\min(r_i,r_j)^2 \approx \langle r^2\rangle_x := \textstyle\sum_j x_j r_j^2 < r_i^2$, so $z_i$ **rises** above this floor as $r_i^2/\langle r^2\rangle_x$: a large pore spends its (larger) budget on cheap small throats and therefore forms more of them. Treating $z_i$ as the constant floor thus mildly under-weights their outgoing connections in (12). This is one of the controlled idealizations of the model, alongside §8 — and the main candidate for refinement (§5.5).

### 5.5 Size-dependent degree (analyzed, not used in production)

Equation (12) assumes a class-independent mean degree $z$ so that it cancels against the total $\sum_{i,j}\nu_{ij}$. In reality the coordination number does depend on pore size: from $z_i = B_i/\langle S_t\rangle_i$, small pores sit at the floor $4\,s_{\mathrm{ratio}}\,\mathrm{PRTRR}^2$ (budget saturated) while large pores rise as $r_i^2/\langle r^2\rangle_x$ — a large pore spends its larger budget on cheap small throats and forms more of them. The class-dependent form weights each term of (12) by $z_i$ and normalizes by the mean degree $\bar z = \sum_i x_i z_i$:

$$
\mathrm{cdist}_k = \frac{x_k z_k\left[K(k\to k)+\textstyle\sum_{j>k}K(k\to j)\right] + \textstyle\sum_{j>k}x_j z_j K(j\to k)}{\bar z}.
\tag{13}
$$

**Why it is not used in production.** Measured on generated networks, the size-dependence of connectivity is a *rare-giant heavy-tail* effect: ~90 % of pores sit at a nearly flat degree (~10–12) and only a handful of giant pores reach high degrees (hundreds). The mean-field $z_i = B_i/\langle S_t\rangle_i$ overshoots that tail by orders of magnitude — it assumes an infinite Poisson gas, so a giant pore surrounded by tiny ones is assigned a degree $\sim 10^7$ — which makes the fixed point stiff: left unclamped it oscillates and does not converge. Even when stabilized with a cap on $z_i$, inverting (13) pushes mass *away* from the large classes, because a high-degree large pore "explains" many small throats so fewer are needed to match $\mathrm{cdist}$; that works against the >0.5 µm band goal that motivated the kernel. The production model therefore keeps equal connectivity (12). The class-dependent variant is available as `pnm.invert_pore_distribution_kernel_experimental` (with an optional `zcap`) for research only. §5.6 revisits the size-dependent degree in two stable, budget-saturated forms and evaluates them against the production kernel.

### 5.6 Multi-capillary corrections: water-filling and budget-coupled identity

Equations (12)–(13) count throats from a mean degree that is either assumed constant (§5.4) or imposed as a mean field (§5.5). Both idealize the **single-link limit**: in reality each pore forms *several* capillaries sharing ONE surface budget — it scans candidates in increasing surface-to-surface distance and accepts each only while its cross-section still fits the remaining budget (exactly what `add_throats_by_surf` does). Two consequences are missed by a class-independent or mean-field degree:

1. the number of capillaries per class is not an independent factor but is **set by** the budget;
2. the partner composition seen by each class is biased toward cheap (small) partners, because an expensive throat must fit in what remains after cheaper, closer ones are paid for.

Two functions address this: `pnm.invert_pore_distribution_kernel_waterfill` models the greedy process directly and derives a **joint** (count, composition) per focal class; `pnm.invert_pore_distribution_kernel_budget` keeps the production kernel's partner composition exactly unchanged and fixes only the count coupling. They form a clean A/B pair: if they agree on a window, the degree was the whole story; if they differ, water-filling's per-class composition correction is what matters.

**Water-filling count.** For a focal pore in class $i$ with budget $B_i = s_{\mathrm{ratio}}\,4\pi r_i^2$, let $S_t[i,j] = \pi\min(r_i,r_j)^2/\mathrm{PRTRR}^2$ be the cross-section of a throat to a class-$j$ partner and $\langle S_t\rangle_i = \sum_j x_j\, S_t[i,j]$ its mean under the pore distribution. Sweeping candidates in distance order, the cumulative cost grows as $(4/3)\,\pi\rho\,\langle S_t\rangle_i\, s^3$ (Poisson gas) and runs out at the kernel reach; a class-$j$ candidate at that boundary is accepted only if it still fits in the *remaining* budget. Integrating over allowed distances gives the per-class count

$$
n_{ij} = x_j\, \frac{B_i - S_t[i,j]}{\langle S_t\rangle_i},
\qquad
z_i = \sum_j n_{ij} = \frac{B_i}{\langle S_t\rangle_i} - 1.
\tag{14}
$$

Two properties distinguish (14) from §5.5: the degree is size-dependent **by construction** — a large pore spends its budget on cheap small throats and forms more of them, with no free parameter (it follows from the budget); and the composition is corrected *per class*, not just rescaled — expensive partners ($S_t[i,j]$ close to $B_i$) are suppressed by $(B_i - S_t[i,j])$. Count and composition come from a single object. The throat identity then counts undirected throats from these directed counts:

$$
\mathrm{cdist}_k = \frac{x_k \sum_{j\ge k} n_{kj} + \sum_{j>k} x_j\, n_{jk}}{\bar z},
\qquad
\bar z = \sum_i x_i z_i.
\tag{15}
$$

**Budget-coupled identity.** The conservative variant plugs the budget-saturated degree (14) into (13), leaving $K(i\to j)$ untouched:

$$
\mathrm{cdist}_k = \frac{x_k z_k\left[K(k{\to}k)+\textstyle\sum_{j>k}K(k{\to}j)\right] + \textstyle\sum_{j>k}x_j z_j K(j{\to}k)}{\bar z},
\qquad
z_i = \frac{B_i}{\langle S_t\rangle_i} - 1.
\tag{16}
$$

The extra $-1$ relative to the §5.5 mean field is the water-filling boundary correction (the focal pore's own first throat must also fit in the budget). Both variants share this degree, so they differ **only** in composition.

**Why the pore-distribution cost, not the kernel mix.** The degree must use the *pore-distribution* mean cost $\langle S_t\rangle_i = \sum_j x_j\,S_t[i,j]$, not the kernel partner mix $\sum_j K(i\to j)\,S_t[i,j]$: the latter double-counts the budget (the reach $D_i$ already encodes it) and collapses to ~0 for large pores whose kernel mix concentrates on tiny cheap partners, blowing up $z_i$ ($\sim 10^3$–$10^4$) and destabilizing the fixed point.

**Numerics.** Both use the §6 scheme (independent-recurrence warm start, backward substitution with the diagonal self-term coupled so each equation stays quadratic in $x_k$, renormalization, forward-residual stopping). The budget variant additionally requires **adaptive damping** of the update ($\lambda = 1$, halved to a floor of 0.05 on stagnation): with a class-dependent degree the undamped map exhibits a period-2 cycle (the top classes oscillate) — the same pathology that forced damping in `invert_pore_distribution_kernel_gamma`. An optional `zcap` clamps $z_i$ from above; it stabilizes wide windows with extreme giants at the cost of an ad-hoc parameter.

**Status.** Both are candidate refinements of the production kernel, **not yet part of the pipeline**. On the keep75 window (N = 1e5) they converge cleanly but shift pore mass toward *smaller* classes and degrade the large band (JSD ≈ 0.12 water-filling / 0.10 budget vs ≈ 0.02 for the production kernel); capping $z_i$ at ~13 restores JSD to ≈ 0.022 — i.e. on that window the whole effect is carried by the size-dependent degree, in line with §5.5's finding that a high-degree large pore "explains" many small throats and pushes mass away from the large classes. The full grid study on the asym50_95 cumulative-volume window (e × sratio × two-phase generator settings, N = 150k; `param_study_v2`, 864 runs, completed 2026-11-20) settles this in a **window-dependent** way: both variants beat the production kernel at 63–67 % of grid cells (mean ΔJSD ≈ −0.07…−0.085), and the budget variant's best point (e = 1.25 / sratio = 0.4 / random order / pass 1 off, JSD 0.0099) ties the kernel's best (0.0093) within MC noise while matching the large band near-exactly (+1.8 pt vs −5.2). Mechanism: relative to the plain kernel at equal $(e, s_{\mathrm{ratio}})$, both corrections shift pore mass toward smaller classes — on keep75 that degrades the large band, but here it corrects the low-$e$ flooding (kernel @ e = 1.25 / sratio = 0.6: large band 91.8 % → ~54–55 % ≈ target 53.2 %). The optimal step-1 factor shifts from $e = 1.75$ to $e = 1.25$, and the budget variant (degree-only) matches or beats water-filling everywhere — on both windows tested, fixing only the count is sufficient.

## 6. Fixed-point scheme

The kernel depends on $x$, so solve $x = F(x)$ by iteration:

1. **Initialize** with the independent-pairing solution (3) — a good warm start; convergence is typically reached in a few dozen sweeps.
2. **Sweep (backward substitution, largest class first).** Freeze $K$ at the current iterate *except for the diagonal self-term*. For class $k$, equation (12) becomes

   $$
   \mathrm{cdist}_k = a_k x_k^2 + b_k x_k + C_k,
   $$

   with

   $$
    a_k = \frac{W_{kk}}{\sum_p x_p W_{kp}},
   \qquad
   b_k = \sum_{j>k} K(k \to j) \;\;(\text{frozen}),
   \qquad
   C_k = \sum_{j>k} x_j K(j \to k) \;\;(\text{already solved}).
   $$

    Keeping $K(k \to k) = x_k W_{kk}/\sum_p x_p W_{kp}$ **coupled** makes each equation quadratic in its own unknown (positive root taken); a fully frozen top class would instead oscillate with period 2. Negative right-hand sides are clamped to zero.
3. **Renormalize** after every sweep so the iterate stays on the simplex $\sum_k x_k = 1$.
4. **Stop** when $\max_k |\Delta x_k| < \mathrm{tol}$ (convergence threshold, default $10^{-12}$). Success is judged on the **forward-model residual**: let $P_c(x)_k$ denote the left-hand side of (12) evaluated at the current iterate — i.e. the throat distribution that the model *predicts* from $x$. The residual $\max_k |P_c(x)_k - \mathrm{cdist}_k|$ measures how well $x$ actually reproduces the target through the kernel, and a warning is issued if it exceeds $10^{-3}$ after `max_iter` sweeps. This matters because near-empty classes can stay pinned at zero — their right-hand side is a difference of ~$10^{-9}$ numbers — while $|\Delta x|$ stalls.

## 7. No free parameter

The kernel uses only quantities already fixed by the model and the pipeline: porosity $\varphi$, surface ratio $s_{\mathrm{ratio}}$, $\mathrm{PRTRR}$ — plus the experimental data. In particular it **replaces the ad-hoc `pdist_shift` gauge**, which existed to absorb exactly this pairing bias. The single empirical parameter of the whole pipeline remains $e$ (step 1, shared with the independent model).

## 8. Finite-size caveat

The kernel is an infinite-system (Poisson-gas) idealization: it assumes every drawn pore can be placed. At finite $N$ the largest bins of a wide window may have expected count $\ll 1$ and/or not fit in the box sized for $\varphi$; those pores are dropped by the packer and the corresponding throat tail stays unreachable **regardless of the inversion** (a shared artifact, not a kernel defect). On wide windows the self-consistent $\rho$ from $\langle r^3\rangle$ also overestimates dilution because realized draws undersample phantom tail bins, so the kernel's correction is weaker there than in the idealized infinite-system limit — but still positive on every metric.

## 9. Validation (A/B, GM1, N = 80k, seed 7, e = 1.75)

Both arms identical except the inversion method; networks in `examples/networks/20261114-18*_kernelAB-*`.

Metrics: **JSD** = Jensen-Shannon divergence between generated and experimental throat *volume* distributions (symmetric, bounded by $\ln 2$, finite even on disjoint supports — the study ranking metric); **L2** = Euclidean distance between the same two normalized per-bin distributions. Generator setup abbreviations: "v1 surf" = `add_throats_by_surf` (pass-based method, notebook default); "f1" = pass-0 candidate-pool widening factor (`minc_search_factor`, active with `sort_by_radius=True`; None = off); "srf" = pass-1 search radius factor (`search_radius_factor`).

| Window | Setup | Independent (JSD / L2) | Kernel (JSD / L2) | Large band > 0.5 µm (gen vs exp) |
|--------|-------|------------------------|-------------------|----------------------------------|
| W1 [0.0162, 1.594] µm | v1 surf, f1=None, srf=2 | 0.0203 / 0.0875 | **0.0063** / **0.0498** | 18.7 % → **27.6 %** (exp 33.1 %) |
| WIDE [midradii[5], midradii[-2]] | v1 surf, f1=1, srf=3 (notebook) | 0.0976 / 0.1339 | **0.0828** / 0.1251 | 3.1 % → 5.4 % (exp 28.8 %) |

- The kernel result on W1 is the **best JSD ever recorded at N = 80k** — it beats the previous best recipe (independent + tuned `pdist_shift = −0.5`, JSD 0.0088) *without any ad-hoc gauge*.
- On WIDE, the N = 80k kernel run already beats the notebook's N = 1e5 independent run (JSD 0.089).
- Mechanism confirmed numerically: the kernel shifts pore mass toward larger classes (W1 top quartile 0.42 % → 0.55 %), compensating the generator's disassortative pairing — e.g. at W1, $K(\text{small} \to \text{top-quartile})$ is depleted ~430× relative to $x$.

## 10. API

```python
# Classical model (theory.typ): independent pairing, exact backward recurrence.
cdist, pdist = pnm.invert_pore_distribution_independent(volumes, midradii, deltaR, e)

# Corrected model: exclusion-kernel fixed point (no free parameter beyond e).
cdist, pdist = pnm.invert_pore_distribution_kernel(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8
)

# Multi-capillary corrections of the kernel (doc §5.6): same signature as the kernel above;
# zcap=None leaves the budget-saturated degree unclamped.
cdist, pdist = pnm.invert_pore_distribution_kernel_waterfill(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8, zcap=None
)
cdist, pdist = pnm.invert_pore_distribution_kernel_budget(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8, zcap=None
)

# Both: feed the pore count distribution to the exact piecewise-uniform sampler.
rdist = pnm.inverse_transform_sampling_from_raw_data(pdist, bins=pore_bins, n_samples=N)
# with pore_bins = mip.throat_radii * PRTRR   (n_bins + 1 full edges).
```

`porosity` and `sratio` must match the arguments passed to `pnm.random_PNM` at generation time.

An experimental class-dependent-degree variant, `pnm.invert_pore_distribution_kernel_experimental(...)` (see §5.5), is provided for research but is **not** used in the pipeline: it does not converge without a degree cap and pushes mass away from the large classes. The multi-capillary variants `pnm.invert_pore_distribution_kernel_waterfill` / `..._kernel_budget` (§5.6) are stable production-oriented candidates — see the status note at the end of §5.6 for their current results (keep75 and asym50_95 windows).
