# From Capillary to Pore Size Distribution: Inversion Theory

This document derives the five pore-distribution inversions implemented in `pnm/Stats.py`:

- **`invert_pore_distribution_independent`** (§3) — It assumes that the two pores connected by a throat are drawn *independently* from the pore distribution.
- **`invert_pore_distribution_kernel`** (§5) — the corrected model. The partner of a focal pore is drawn from a size-conditioned *exclusion kernel* derived from the surface-budget mechanics of the generator itself (`pnm.Transformation.add_throats_by_surf`). It introduces no free parameter beyond $e$.
- **`invert_pore_distribution_kernel_waterfill`** (§5.5.1) and **`invert_pore_distribution_kernel_budget`** (§5.5.2) — multi-capillary corrections of the kernel that replace its class-independent mean degree by the budget-saturated, size-dependent one; water-filling additionally corrects the per-class partner composition, while budget keeps it unchanged.
- **`invert_pore_distribution_kernel_gamma`** (§5.6) — the parameter-free variant: it replaces step 1's empirical exponent $e$ by a per-class mean capillary length *derived* from the same exclusion-kernel geometry (incomplete-gamma length moments), so no free parameter remains at all.

All five functions take the experimental MIP throat distribution as input and return the same output contract: `(cdist, pdist)`, where `pdist` is a pore **count** distribution (one probability per bin, summing to 1). The four $e$-based variants additionally take the corrective factor $e$; the parameter-free variant takes only $\gamma$, porosity and sratio. To use it, feed `pdist` — together with the full pore bin edges (`throat_radii * PRTRR`, i.e. n_bins + 1 values) — into `pnm.inverse_transform_sampling_from_raw_data`.

## Notation

All radii are in µm. A "class" is a bin index; we use $i, j, k$ interchangeably as class indices over $\lbrace 0, \dots, n-1 \rbrace$, with classes ordered by increasing radius (class 0 = smallest). Every symbol used below is defined here or at its first appearance.

| Symbol | Meaning |
|--------|---------|
| $n$ | number of throat bins (= number of pore bins) |
| $N$ | total number of pores in the network (distinct from $n$, the number of bins) |
| ${r^c_k}$ | center radius of throat bin $k$ — `midradii[k]` |
| $\Delta r_k$ | width of throat bin $k$: the bin spans $[{r^c_k} - \tfrac12\Delta r_k,\ {r^c_k} + \tfrac12\Delta r_k]$ — `deltaR[k]` |
| $v_k$ | experimental intruded volume fraction in throat bin $k$, with $\sum_k v_k = 1$ — `volumes[k]` |
| $e$ | empirical corrective exponent, the only free parameter of the $e$-based inversions (tuned values ≈ 1.5–2); absent in the parameter-free variant (§5.6) |
| $\gamma$ | pore-to-throat radius ratio: a throat between pores of radii $r_i, r_j$ has radius $\min(r_i, r_j)/\gamma$. The pipeline uses 2. |
| $x_k$ | **unknown** of the inversion: probability that a randomly drawn pore falls in pore bin $k$, with $\sum_k x_k = 1$ (pore *count* distribution) |
| $r_k$ | center radius of pore bin $k$: $r_k = \gamma\ {r^c_k}$ (pores are PRTRR times larger than throats) |
| $\varphi$ | target pore-body porosity of the packing that will be generated (pipeline: 0.35) |
| $\rho$ | number density of pores in the box ($\mu\mathrm{m}^{-3}$), derived from $x$ and $\varphi$ (§5.1) |
| $\langle r^3\rangle$ | mean cubed pore radius under $x$: $\sum_k x_k r_k^3$ |
| $s_{\mathrm{ratio}}$ | surface coverage ratio: fraction of a pore's surface that may be covered by throats (pipeline: 0.8) |
| $B_i$ | surface budget of a focal pore in class $i$: $B_i = s_{\mathrm{ratio}}\ 4\pi r_i^2$ ($\mu\mathrm{m}^2$) |
| $\langle S_t\rangle_i$ | mean cross-section of a throat from a pore in class $i$ to a partner drawn from $x$ (§5.2) |
| $S_t[i,j]$ | cross-section of a throat between classes $i$ and $j$: $\pi\ \min(r_i, r_j)^2/\gamma^2$ (§5.5) |
| $z_i$, $\bar z$ | mean degree (number of throats) of a pore in class $i$ / network-mean degree; budget-saturated form in §5.5 |
| $n_{ij}$ | directed count of capillaries from a focal pore in class $i$ to partners in class $j$ (water-filling, §5.5.1) |
| $\ell$ | center-to-center distance between two pores ($\mu\mathrm{m}$); the running length variable in §5.2–5.3 |
| $\ell_i$ | effective connection reach of a pore in class $i$ ($\mu\mathrm{m}$): the value of $\ell$ at which its budget runs out, derived from its budget (§5.2) |
| $W_{ij}$ | unnormalized pair weight between classes $i$ and $j$ (§5.3) |
| $K(i \to j)$ | probability that the partner of a focal pore in class $i$ falls in class $j$; each row sums to 1 (§5.3) |
| $\mathrm{cdist}_k$ | corrected throat *count* distribution — the target of the inversion, with $\sum_k \mathrm{cdist}_k = 1$ (§2) |
| $g_{ij}$ | mean surface-to-surface gap between a focal pore in class $i$ and a partner in class $j$, from the kernel geometry (§5.6) |
| $\bar L_k$ | per-class mean capillary gap for throats in class $k$, aggregated from the pair gaps $g_{ij}$ (§5.6) |

## 1. Problem setup

An experimental mercury porosimetry (MIP) curve is binned into $n$ throat-radius classes: bin centers ${r^c_k}$, widths $\Delta r_k$, and normalized intruded volume fractions $v_k$.

The generator builds a network of pores whose radii are drawn from an unknown count distribution $x$. A throat connecting two pores of radii $r_i, r_j$ has radius

$$
r_t = \frac{\min(r_i, r_j)}{\gamma},
$$

so **the class of a throat is set by its smaller endpoint**: a large pore connected to a small one produces a *small* throat.

**Inversion problem.** Find the pore count distribution $x$ such that when the generator connects the pores, the resulting throat volume distribution matches the experimental $(v_k)$.

## 2. Step 1: from volumes to counts

A MIP bin reports how much *volume* was intruded through throats of a given size range; the inversion needs how many *throats* there are in each class. A throat in bin $k$ has radius $\approx {r^c_k}$, hence volume ≈ cross-section × length, so dividing the per-bin volume fraction by a characteristic cross-section converts it into a count fraction.

For a radius drawn uniformly from bin $k$ (spanning $[{r^c_k} - \tfrac12\Delta r_k,\ {r^c_k} + \tfrac12\Delta r_k]$), the mean squared radius is

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

The exponent $e$ is the **only empirical parameter** of the whole pipeline: it absorbs the residual size dependence of throat length and other second-order effects ($e = 1$ means pure cross-section correction; tuned values are ≈ 1.5–2). Equation (2) is shared verbatim by the four $e$-based inversions (private helper `_corrected_throat_counts`); the parameter-free variant (§5.6) instead uses equation (1) in its exact form with a derived $\bar L_k$.

## 3. The independent-pairing model

**Assumption.** The two pores connected by a throat are drawn *independently* from $x$. Knowing the size of one endpoint tells you nothing about the other.

A throat falls in class $k$ iff its smaller endpoint is in class $k$. There are two mutually exclusive ways this can happen:

- both endpoints are in $k$: probability $x_k \cdot x_k$;
- one endpoint is in $k$ and the other in a larger class $j > k$: probability $2\ x_k x_j$ (factor 2: either of the two drawn pores may be the one that falls in class $k$; it is then automatically the smaller one since $j > k$).

Summing over all larger classes gives

$$
\mathrm{cdist}_k = x_k^2 + 2\ x_k \sum_{j>k} x_j .
\tag{3}
$$

Writing $T_k := \sum_{j>k} x_j$ (the tail of $x$ above class $k$), equation (3) is a quadratic in the unknown $x_k$ whose coefficients involve only *larger* classes:

$$
x_k^2 + 2 T_k x_k - \mathrm{cdist}_k = 0 .
$$

The system is therefore solved **exactly by backward substitution**, from the largest class down, each step being a single quadratic with known coefficients (positive root):

$$
x_{n-1} = \sqrt{\mathrm{cdist}_{n-1}},
\qquad
x_k = -T_k + \sqrt{T_k^2 + \mathrm{cdist}_k}.
\tag{4}
$$

No iteration is needed: (4) is the exact solution of (3). This is `invert_pore_distribution_independent`.

## 4. Why independence fails: the spatial pairing mechanics

The actual generator does **not** pair pores randomly. In pass 1 of `add_throats_by_surf`:

1. each pore $i$ (the *focal* pore) has a surface budget $B_i = s_{\mathrm{ratio}}\ 4\pi r_i^2$;
2. its candidates are scanned in **increasing** surface-to-surface distance;
3. a candidate pair $(i, j)$ is accepted only if both pores' budgets can still absorb the throat cross-section $\pi (\min(r_i, r_j)/\gamma)^2$, after which that amount is deducted from both budgets.

Two consequences follow:

1. **Disassortative pairing.** A large focal pore spends its budget on cheap nearby small throats long before a distant large partner is reached in the distance ordering (a throat to a small partner costs little cross-section, so many of them fit within $B_i$). The partner-size distribution of a pore in class $i$ is therefore *not* $x$: it is size-conditioned by proximity and cost, strongly depleted at the top for every $i$.
2. **Large-throat deficit.** A large–large throat requires two large pores to be close — rare in a random placement (the expected number of close large-large pairs scales linearly with $N$). The generated network systematically under-produces the largest throat band, whatever $e$ is tuned to.

The independent inversion returns the $x$ that would reproduce the experiment **only if pairing were random**. Fed into a spatial generator, the bias survives. The kernel below removes the assumption instead of patching its consequence.

## 5. Derivation of the exclusion kernel

The goal: replace the partner distribution $x$ in (2) by a partner distribution $K(i \to j)$ that reflects what the generator actually does, using only quantities already fixed by the model ($\varphi$, $s_{\mathrm{ratio}}$, $\gamma$).

### 5.1 Number density from porosity

The kdtree packers rescale the box so that total pore volume / box volume = target porosity $\varphi$. With $N$ pores of mean cubed radius $\langle r^3\rangle = \sum_k x_k r_k^3$, the box volume is $V_{\mathrm{box}} = N (4/3)\pi \langle r^3\rangle / \varphi$, hence the number density

$$
\rho := \frac{N}{V_{\mathrm{box}}} = \frac{9\ \varphi}{4\pi\ \langle r^3\rangle}.
\tag{5}
$$

Note that $\rho$ depends on $x$, one reason the inversion is a fixed point (§6).

### 5.2 Connection reach from the surface budget

For a focal pore in class $i$ (radius $r_i$, budget $B_i = s_{\mathrm{ratio}}\ 4\pi r_i^2$), let $\langle S_t\rangle_i$ be the mean cross-section of a throat to a partner drawn from $x$: averaging over the partner's class $j$,

$$
\langle S_t\rangle_i = \frac{\pi}{\gamma^2}\sum_j x_j\ \min(r_i, r_j)^2 .
\tag{6}
$$

(The $\min$ encodes the throat-radius rule: the smaller endpoint sets the size.)

In a homogeneous packing, the expected number of neighbors within center distance $\ell$ is $(4/3)\pi \ell^3 \rho$. If pore $i$ accepted every such neighbor, the cumulative cross-section it would have paid up to distance $\ell$ is

$$
C(\ell) \approx (4/3)\ \pi\ \ell^3\ \rho\ \langle S_t\rangle_i .
\tag{7}
$$

The budget runs out when $C(\ell_i) = B_i$, which defines the **effective connection reach** of class $i$:

$$
\ell_i^3 = \frac{B_i}{(4/3)\ \pi\ \rho\ \langle S_t\rangle_i}.
\tag{8}
$$

A partner at center distance $\ell$ is therefore connected with survival probability

$$
P(\ell) \approx \exp\left(-\left(\frac{\ell}{\ell_i}\right)^3\right),
\qquad \text{i.e. } P(\ell) \approx \exp\big(-C(\ell)/B_i\big).
\tag{9}
$$

Equation (9) is a mean-field idealization of the sequential scan: $C(\ell)/B_i$ plays the role of an accumulated hazard, and a candidate at distance $\ell$ survives only if the budget spent on all closer candidates left room for it.

### 5.3 Partner counting with hard-core exclusion

The number density of pores in class $j$ is $\rho x_j$ (a fraction $x_j$ of all pores). The expected number of *connected* partners in class $j$ around a focal pore in class $i$ is the shell integral

$$
\int_{r_i + r_j}^{\infty} \underbrace{4\pi \ell^2\ \rho x_j}_{\text{class-}j\text{ pores in }[\ell,\ \ell+d\ell]}
\exp\left(-\left(\frac{\ell}{\ell_i}\right)^3\right) d\ell .
\tag{10}
$$

The lower limit $r_i + r_j$ is **hard-core exclusion**: two non-overlapping spheres of radii $r_i, r_j$ have center distance at least $r_i + r_j$ (surface gap $\ge 0$).

With the substitution $u = (\ell/\ell_i)^3$, one has $\ell^2\ d\ell = \ell_i^3 du / 3$ and the lower limit becomes $a_{ij}^3$ with $a_{ij} := (r_i + r_j)/\ell_i$, so that $\int_{a_{ij}^3}^{\infty} e^{-u}\ du = e^{-a_{ij}^3}$ and (10) evaluates in closed form to

$$
\rho x_j\ \tfrac{4}{3}\pi \ell_i^3\ \exp\left(-\left(\frac{r_i + r_j}{\ell_i}\right)^3\right).
\tag{11}
$$

The factors common to all $j$ (the $\rho$ and $(4/3)\pi \ell_i^3$) do not affect the *relative* weights of partner classes, so we define

$$
W_{ij} := \exp\left(-\left(\frac{r_i + r_j}{\ell_i}\right)^3\right)
$$

and normalize each row to a probability. The **partner-size distribution of a focal pore in class $i$** is

$$
K(i \to j) = \frac{x_j\ W_{ij}}{\sum_p x_p\ W_{ip}},
\qquad \text{so that } \sum_j K(i \to j) = 1 .
\tag{12}
$$

$K$ depends on $x$ through its rows ($\ell_i$ via (5)–(7)), their normalization, and $\rho(x)$ — the inversion is a fixed point (§6).

### 5.4 The throat identity

We now count throats from **directed** connections. Let $N_i \approx N x_i$ be the number of pores in class $i$, and let $z_i$ be the mean degree (number of throats) of a pore in class $i$. Focal pores in class $i$ make $N_i z_i$ directed connections, distributed over partner classes according to row $K(i \to \cdot)$; hence the number of directed connections from class $i$ to class $j$ is

$$
\nu_{ij} = N_i\ z_i\ K(i \to j).
$$

Every undirected edge contributes exactly two directed connections (one per endpoint), and **both** have the same smaller-endpoint class. Therefore the fraction of directed connections whose smaller endpoint lies in class $k$ equals the fraction of throats in class $k$. A directed connection has its smaller endpoint in class $k$ in exactly two cases:

- the focal pore is in class $k$ and points to an equal or larger partner ($j \ge k$);
- the focal pore is in a larger class $j > k$ and points down at partner class $k$.

The total number of directed connections is $\sum_{i,j} \nu_{ij} = Nz \sum_i x_i = Nz$ when all classes share one mean degree $z$ (the assumption discussed below). Dividing the two cases by this total gives

$$
\mathrm{cdist}_k = x_k \left[ K(k \to k) + \sum_{j>k} K(k \to j) \right] + \sum_{j>k} x_j\ K(j \to k).
\tag{13}
$$

Two properties worth noting:

- **(13) sums to 1 identically** in $x$: the two cases above partition all directed connections (diagonal, focal-smaller, focal-larger), so no choice of $x$ can make the target inconsistent.
- When $K(i \to j) = x_j$ (independence), (13) reduces exactly to (3).

*Where the equal-degree assumption enters.* The factor $z_i$ cancels in (13) only if it is class-independent. It is **exactly** constant when a typical partner of pore $i$ is larger than it: then $\sum_j x_j\ \min(r_i, r_j)^2 \approx r_i^2$ and

$$
z_i = \frac{B_i}{\langle S_t\rangle_i}
= \frac{s_{\mathrm{ratio}}\ 4\pi r_i^2}{\pi r_i^2 / \gamma^2}
= 4\ s_{\mathrm{ratio}}\ \gamma^2,
$$

independent of $r_i$. For the largest classes (whose partners are mostly smaller), $\sum_j x_j\min(r_i,r_j)^2 \approx \langle r^2\rangle_x := \textstyle\sum_j x_j r_j^2 < r_i^2$, so $z_i$ **rises** above this floor as $r_i^2/\langle r^2\rangle_x$: a large pore spends its (larger) budget on cheap small throats and therefore forms more of them. Treating $z_i$ as the constant floor thus mildly under-weights their outgoing connections in (13). This is one of the controlled idealizations of the model, alongside §8 — refined in §5.5 by the budget-saturated size-dependent degree.

### 5.5 Multi-capillary corrections: size-dependent degree

Equation (13) counts throats from a mean degree that is assumed class-independent (§5.4). That idealizes the **single-link limit**: in reality each pore forms *several* capillaries sharing ONE surface budget — it scans candidates in increasing surface-to-surface distance and accepts each only while its cross-section still fits the remaining budget (exactly what `add_throats_by_surf` does). Two consequences are missed by a class-independent degree:

1. the number of capillaries per class is not an independent factor but is **set by** the budget;
2. the partner composition seen by each class is biased toward cheap (small) partners, because an expensive throat must fit in what remains after cheaper, closer ones are paid for.

Two functions address this: `pnm.invert_pore_distribution_kernel_waterfill` models the greedy process directly and derives a **joint** (count, composition) per focal class; `pnm.invert_pore_distribution_kernel_budget` keeps the production kernel's partner composition exactly unchanged and fixes only the count coupling. They form a clean A/B pair: if they agree on a window, the degree was the whole story; if they differ, water-filling's per-class composition correction is what matters.

**Shared quantity — budget-saturated degree.** For a focal pore in class $i$ with budget $B_i = s_{\mathrm{ratio}}\ 4\pi r_i^2$, let $\langle S_t\rangle_i = \sum_j x_j\ S_t[i,j]$ be the mean cross-section of a throat to a partner drawn from the *pore* distribution. Sweeping candidates in distance order, the cumulative cost grows as $(4/3)\ \pi\rho\ \langle S_t\rangle_i\ s^3$ (Poisson gas) and runs out at the kernel reach; a candidate is accepted only if it still fits in the *remaining* budget. The **budget-saturated mean degree** of class $i$ is therefore

$$
z_i = \frac{B_i}{\langle S_t\rangle_i} - 1,
\qquad
\bar z = \sum_i x_i z_i .
\tag{14}
$$

The degree is size-dependent **by construction** — a large pore spends its budget on cheap small throats and forms more of them, with no free parameter (it follows from the budget). The $-1$ is a boundary correction: the focal pore's own first throat must also fit in the budget. Both variants below share this degree; they differ **only** in composition.

*Why the pore-distribution cost, not the kernel mix.* The degree must use the *pore-distribution* mean cost $\langle S_t\rangle_i = \sum_j x_j\ S_t[i,j]$, not the kernel partner mix $\sum_j K(i\to j)\ S_t[i,j]$: the latter double-counts the budget (the reach $\ell_i$ already encodes it) and collapses to ~0 for large pores whose kernel mix concentrates on tiny cheap partners, blowing up $z_i$ (between $\sim 10^3$ and $10^4$) and destabilizing the fixed point.

#### 5.5.1 Water-filling inversion (`invert_pore_distribution_kernel_waterfill`)

A candidate in class $j$ at the budget boundary is accepted only if it still fits in the *remaining* budget; integrating this acceptance condition over allowed distances gives the per-class directed count

$$
n_{ij} = x_j\ \frac{B_i - S_t[i,j]}{\langle S_t\rangle_i},
\qquad
z_i = \sum_j n_{ij}.
\tag{15}
$$

The composition is corrected *per class*, not just rescaled — expensive partners ($S_t[i,j]$ close to $B_i$) are suppressed by $(B_i - S_t[i,j])$. Count and composition come from a single object. The throat identity then counts undirected throats from these directed counts:

$$
\mathrm{cdist}_k = \frac{x_k \sum_{j\ge k} n_{kj} + \sum_{j>k} x_j\ n_{jk}}{\bar z}.
\tag{16}
$$

**Numerics.** Uses the §6 scheme (independent-recurrence warm start, backward substitution with the diagonal self-term coupled so each equation stays quadratic in $x_k$, renormalization, forward-residual stopping).

#### 5.5.2 Budget-coupled identity (`invert_pore_distribution_kernel_budget`)

The conservative variant extends the throat identity (13) with the class-dependent degree (14), leaving $K(i\to j)$ untouched:

$$
\mathrm{cdist}_k = \frac{x_k z_k\left[K(k{\to}k)+\textstyle\sum_{j>k}K(k{\to}j)\right] + \textstyle\sum_{j>k}x_j z_j K(j{\to}k)}{\bar z}.
\tag{17}
$$

**Numerics.** Same §6 scheme, but additionally requires **adaptive damping** of the update ($\lambda = 1$, halved to a floor of 0.05 on stagnation): with a class-dependent degree the undamped map exhibits a period-2 cycle (the top classes oscillate) — the same pathology that forced damping in `invert_pore_distribution_kernel_gamma`. An optional `zcap` clamps $z_i$ from above; it stabilizes wide windows with extreme giants at the cost of an ad-hoc parameter.

**Mechanism.** Relative to the plain kernel at equal $(e, s_{\mathrm{ratio}})$, the correction shifts pore mass toward smaller classes: a high-degree large pore "explains" many small throats, so fewer are needed and mass moves away from the large classes. The effect is carried entirely by the size-dependent degree — fixing only the count suffices, and water-filling's per-class composition correction adds nothing on top.

### 5.6 Parameter-free capillary length (`invert_pore_distribution_kernel_gamma`)

**Motivation.** Step 1 of every $e$-based inversion divides each bin's volume by $(\pi\ \mathbb{E}[r^2])^e$. A throat in class $k$ has volume ≈ cross-section × length, so step 1 is *exact if and only if*

$$
\left(\pi\ \mathbb{E}[r^2]_k\right)^e = \langle S\ L\rangle_k \approx S_k\ \bar L_k
\quad\Longleftrightarrow\quad
\bar L_k \propto {r^c_k}^{2(e-1)},
$$

i.e. it silently assumes a power law for the mean capillary length $\bar L_k$ with a *positive* exponent in the throat bin center ${r^c_k}$ ($+0.5$ at $e=1.25$, $+1.5$ at $e=1.75$). Measured per-class gaps on generated networks show the opposite behaviour: nearly flat for small throats and *decreasing* with size in the large classes — large pores that end up connected are nearly in contact (nearest-neighbor selection, §4) — so no value of $e$ can fix both ends at once. The variant below removes the assumption instead: it derives $\bar L_k$ from the *same* Poisson-gas geometry that produces $W_{ij}$ and $\ell_i$, making step 1 — and hence the whole inversion — parameter-free (only $\varphi$, $s_{\mathrm{ratio}}$ and $\gamma$ enter).

#### 5.6.1 Per-pair length moments

The **upper incomplete gamma function** is defined by $\Gamma(s, x) := \int_x^\infty t^{s-1} e^{-t}\ dt$: it measures the remaining tail of the gamma integral beyond $x$ (the complementary lower form is $\gamma(s,x) = \int_0^x t^{s-1}e^{-t}dt$, with $\Gamma(s,x)+\gamma(s,x)=\Gamma(s)$; in `scipy.special`, `gammaincc` returns the regularized upper form).

**Why it appears here.** The exclusion kernel models the partner of a focal pore in class $i$ (reach $\ell_i$) as drawn from a Poisson gas with survival weight (9) and hard-core exclusion $s \ge a_{ij} = r_i + r_j$, so the partner-distance density is

$$
f(s) \propto s^2\ e^{-(s/\ell_i)^3}, \qquad s \in [a_{ij},\ \infty).
$$

The *count* of partners (which gives $W_{ij}$) needs only $\int_{a_{ij}}^\infty s^2 e^{-(s/\ell_i)^3}ds$, elementary after $u = (s/\ell_i)^3$ because the integrand becomes a pure exponential. The *length moment*, however, carries one extra power of $s$:

$$
M_{ij} = 4\pi\rho \int_{a_{ij}}^{\infty} s^3\ e^{-(s/\ell_i)^3}\ ds .
$$

With $u = (s/\ell_i)^3$ one has $s^3\ ds = \tfrac{\ell_i^4}{3}\ u^{1/3}\ du$, so the integral becomes $\tfrac{\ell_i^4}{3}\int_{x_{ij}}^{\infty} u^{1/3} e^{-u}\ du$ — and **that remaining integral is exactly the upper incomplete gamma** $\Gamma(\tfrac43, x_{ij})$, with

$$
x_{ij} = \left(\frac{r_i + r_j}{\ell_i}\right)^3 > 0 .
$$

The lower limit $x_{ij}$ is positive *because of hard-core exclusion*: two non-overlapping pores cannot have their centers closer than $r_i+r_j$. If partners could sit at any distance ($a_{ij}=0$), then $x_{ij}=0$ and the moment would collapse to the **complete** gamma $\Gamma(4/3)$ — which is why the incomplete version is needed: it is the gamma integral with its lower part cut off by the exclusion distance.

Dividing the length moment by the count (the $4\pi\rho$ factors cancel) gives the mean center distance and, subtracting the exclusion distance, the mean gap:

$$
\langle s\rangle_{ij} = \ell_i\ e^{x_{ij}}\ \Gamma\left(\tfrac{4}{3},\ x_{ij}\right),
\qquad
g_{ij} = \langle s\rangle_{ij} - a_{ij}.
\tag{18}
$$

**Limits (sanity checks).** $x_{ij}\to 0$ (exclusion negligible): $\Gamma(4/3)\approx 0.893$, so $\langle s\rangle \to 0.893\ \ell_i$ — the partner distance is a pure Weibull-3 draw from the origin. $x_{ij}\to\infty$ (exclusion dominates): using $\Gamma(s,x) \sim x^{s-1}e^{-x}(1 + (s-1)/x + \cdots)$, one gets $\langle s\rangle \approx a_{ij}(1 + 1/(3x_{ij})) \to a_{ij}$ — all partners sit at contact, gap → 0. The formula interpolates between these regimes with crossover at $x_{ij}\sim O(1)$, and $\langle s\rangle_{ij} \ge a_{ij}$ holds identically (the integral's lower limit enforces it).

**Median center distance (closed form).** The CDF of the partner distance is

$$
F(s) = \frac{\int_{a_{ij}}^s t^2 e^{-(t/\ell_i)^3}dt}{\int_{a_{ij}}^\infty t^2 e^{-(t/\ell_i)^3}dt}
= 1 - \exp\Big(-\big[(s/\ell_i)^3 - x_{ij}\big]\Big),
$$

so the median solves $F(s_{\mathrm{med}}) = 1/2$:

$$
s_{{\mathrm{med}},ij} = \left(a_{ij}^3 + \ell_i^3 \ln 2\right)^{1/3}
\qquad\text{(median gap } = s_{{\mathrm{med}},ij} - a_{ij}\text{).}
$$

No special functions needed: the median is the contact distance "inflated" by an additive volume term $\ell_i^3\ln 2$. In the two limits it reduces to $0.856\ \ell_i$ (no exclusion; $(\ln 2)^{1/3}\approx 0.856 < 0.893$, as expected for a right-skewed density) and to $a_{ij}(1 + \tfrac{\ln 2}{3x_{ij}})$ (contact). The median is the more robust statistic in the large classes, where the gap distribution has a pile-up at 0 plus a long tail. `length_stat` selects "mean" (default) or "median".

#### 5.6.2 Aggregation per throat class

A throat of class $k$ (smaller endpoint in bin $k$) can be produced by two directed mechanisms, exactly as in the throat identity (§5.4): focal pore in class $k$ pointing to an equal-or-larger partner ($j \ge k$), or a larger focal pore ($i > k$) pointing down at class $k$. Weighting each directed connection by its production rate $x_i\ K(i\to j)$:

$$
\bar L_k = \frac{x_k \sum_{j \ge k} K(k \to j)\ g_{kj} + \textstyle\sum_{i > k} x_i\ K(i \to k)\ g_{ik}}{\mathrm{cdist}_k}.
\tag{19}
$$

The denominator is *identically* $\mathrm{cdist}_k$ (throat identity), so this is a genuine average over the throats that step 1 is trying to count, not an additional assumption.

Classes with zero mass at an intermediate iterate have an undefined $0/0$ average; they fall back to the $x$-independent geometric mean of the same pairs weighted by the spatial kernel $W_{ij}$ alone, which keeps $\bar L_k$ finite and $O(\ell_i)$ so step 1 stays well-scaled.

#### 5.6.3 Step 1 without $e$: the coupled fixed point

With $\bar L_k$ derived, step 1 becomes parameter-free:

$$
\mathrm{cdist}_k = \frac{v_k}{S_k\ \bar L_k},
\qquad
S_k = \pi\left({r^c_k}^2 + \frac{\Delta r_k^2}{12}\right).
\tag{20}
$$

The porosity enters through $\rho$ and $\ell_i$ as intended; the empirical exponent $e$ disappears. Because step 1 now depends on $x$ (through $\ell_i$, $K$ and hence $\bar L_k$), the inversion is a *coupled* fixed point: at each sweep, build the kernel geometry from the current iterate, recompute $\bar L_k$ and the step-1 $\mathrm{cdist}$, then solve the throat identity by backward substitution (§6) with the kernel frozen except for the diagonal self-term. The update is damped, $x \leftarrow (1-\lambda)\ x + \lambda\ x_{\mathrm{new}}$: the undamped map exhibits a period-2 cycle at low sratio (a kink induced by zero-mass classes, whose production-weighted $\bar L_k$ and the $W$-based fallback disagree) — the same pathology that forced damping in §5.5.2. $\lambda$ starts at 1 (fast path) and is halved automatically after 25 stalled iterations, down to a floor of 0.05; success is judged on the forward-model residual of the *coupled* map (how well $x$ reproduces its own step-1 cdist through $K$).

#### 5.6.4 Known limitations

The survival weight (9) is a mean-field hazard: it assigns non-negligible probability to partners far beyond the nearest one, whereas the generator scans candidates in increasing distance order and accepts the first affordable one (§4). In the large classes — where focal pores are sparse and connect at near-contact distances — this overpredicts the derived gap; `length_stat="median"` is the more robust statistic there. The small-throat regime, set by interstitial spacing, is captured by the budget-reach mechanism ($\ell_i$ from porosity + sratio) without any fitted parameter.

## 6. Fixed-point scheme

The kernel depends on $x$, so solve $x = F(x)$ by iteration:

1. **Initialize** with the independent-pairing solution (4) — a good warm start; convergence is typically reached in a few dozen sweeps.
2. **Sweep (backward substitution, largest class first).** Freeze $K$ at the current iterate *except for the diagonal self-term*. For class $k$, equation (13) becomes

$$
\mathrm{cdist}_k = a_k x_k^2 + b_k x_k + C_k,
$$

with

$$
a_k = \frac{W_{kk}}{\sum_p x_p W_{kp}},
\qquad
b_k = \sum_{j>k} K(k \to j)\quad(\text{frozen}),
\qquad
C_k = \sum_{j>k} x_j K(j \to k)\quad(\text{already solved}).
$$

Keeping $K(k \to k) = x_k W_{kk}/\sum_p x_p W_{kp}$ **coupled** makes each equation quadratic in its own unknown (positive root taken); a fully frozen top class would instead oscillate with period 2. Negative right-hand sides are clamped to zero.

3. **Renormalize** after every sweep so the iterate stays on the simplex $\sum_k x_k = 1$.
4. **Stop** when $\max_k |\Delta x_k| < \mathrm{tol}$ (convergence threshold, default $10^{-12}$). Success is judged on the **forward-model residual**: let $P_c(x)_k$ denote the left-hand side of (13) evaluated at the current iterate — i.e. the throat distribution that the model *predicts* from $x$. The residual $\max_k |P_c(x)_k - \mathrm{cdist}_k|$ measures how well $x$ actually reproduces the target through the kernel, and a warning is issued if it exceeds $10^{-3}$ after `max_iter` sweeps. This matters because near-empty classes can stay pinned at zero — their right-hand side is a difference on the scale of $10^{-9}$ — while $|\Delta x|$ stalls.


## 7. Finite-size caveat

The kernel is an infinite-system (Poisson-gas) idealization: it assumes every drawn pore can be placed. At finite $N$ the largest bins of a wide window may have expected count $\ll 1$ and/or not fit in the box sized for $\varphi$; those pores are dropped by the packer and the corresponding throat tail stays unreachable **regardless of the inversion** (a shared artifact, not a kernel defect). On wide windows the self-consistent $\rho$ from $\langle r^3\rangle$ also overestimates dilution because realized draws undersample phantom tail bins, so the kernel's correction is weaker there than in the idealized infinite-system limit — but still positive on every metric.

## 8. API

```python
# Classical model (theory.typ): independent pairing, exact backward recurrence.
cdist, pdist = pnm.invert_pore_distribution_independent(volumes, midradii, deltaR, e)

# Corrected model: exclusion-kernel fixed point (no free parameter beyond e).
cdist, pdist = pnm.invert_pore_distribution_kernel(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8
)

# Multi-capillary corrections of the kernel (doc §5.5): same signature as the kernel above;
# zcap=None leaves the budget-saturated degree unclamped.
cdist, pdist = pnm.invert_pore_distribution_kernel_waterfill(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8, zcap=None
)
cdist, pdist = pnm.invert_pore_distribution_kernel_budget(
    volumes, midradii, deltaR, e, PRTRR=2.0, porosity=0.35, sratio=0.8, zcap=None
)

# Parameter-free variant (doc §5.6): no step-1 exponent — the capillary length is derived from
# the kernel geometry; length_stat selects "mean" (default) or "median" pair gaps.
cdist, pdist = pnm.invert_pore_distribution_kernel_gamma(
    volumes, midradii, deltaR, PRTRR=2.0, porosity=0.35, sratio=0.8, length_stat="mean"
)

# All: feed the pore count distribution to the exact piecewise-uniform sampler.
rdist = pnm.inverse_transform_sampling_from_raw_data(pdist, bins=pore_bins, n_samples=N)
# with pore_bins = mip.throat_radii * PRTRR   (n_bins + 1 full edges).
```

`porosity` and `sratio` must match the arguments passed to `pnm.random_PNM` at generation time. All five inversions are selectable by name in `fitted_random_PNM` (`"independent"`, `"kernel"`, `"kernel_budget"`, `"kernel_waterfill"`, `"kernel_gamma"`); the parameter-free variant ignores the step-1 exponent $e$.
