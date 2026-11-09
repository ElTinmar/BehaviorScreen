# Derivations

Proofs of the identities the likelihood and the time rescaling rely on, each tied to the code that implements it.
The algebra of sections 1 to 3 and the limit of section 4 are also checked symbolically (`test/test_layer0_symbolic.py`);
section 4's argument is on paper only. Approximations (quadrature over gains, the time grid of renewal processes, the
optimizer) are not covered here: the tests check those against independent methods.

Notation: $\Phi$, $\varphi$ the standard normal CDF and density; $(a)_k = \Gamma(a+k)/\Gamma(a)$; a stream is one fish's trial.

## 1. Survival hazards: exact cumulative hazards

`survival_process.py`: each kernel's `integral_func` is `exact(F)`, $H(t) = F(t) - F(0)$, so it is the cumulative
hazard iff $F' = h$. The factor $H e^{\alpha m}$, the baseline $B$ and a trial-dependent width do not depend on $t$,
so it suffices to show each shape's antiderivative.

**Constant**, $h = B$: $F = Bt$.

**Gaussian bump**, $e^{-(t-\mu)^2/2\sigma^2}$, `gaussian_area`: $F = \sigma\sqrt{2\pi}\,\Phi(z)$, $z = (t-\mu)/\sigma$.
$F' = \sigma\sqrt{2\pi}\,\varphi(z)/\sigma = e^{-z^2/2}$. The variable-width kernel uses the same $F$ with $\sigma(m)$.

**Gamma pulse**, $(t/\tau)e^{-t/\tau}$: $F = -(\tau + t)e^{-t/\tau}$.
$F' = -e^{-t/\tau} + \frac{\tau + t}{\tau}e^{-t/\tau} = \frac{t}{\tau}e^{-t/\tau}$.

**Ex-Gaussian bump**, `exgaussian_shape`:
$f(t) = \frac{1}{\tau}\exp\!\big(\tfrac{\sigma^2}{2\tau^2} - \tfrac{t-\mu}{\tau}\big)\,\Phi\!\big(z - \tfrac{\sigma}{\tau}\big)$
(the code's exponent $\kappa(\kappa/2 - z) + \log\Phi(z-\kappa) - \log(\tau/\sigma) - \log\sigma$ with $\kappa = \sigma/\tau$ is the log of this).
`exgaussian_cdf`: $F = \Phi(z) - \tau f(t)$. Differentiating $\tau f$:

$$\frac{d}{dt}\,\tau f = -f + \frac{1}{\sigma}\,e^{\sigma^2/2\tau^2 - (t-\mu)/\tau}\,\varphi\!\big(z - \tfrac{\sigma}{\tau}\big) = -f + \frac{\varphi(z)}{\sigma},$$

because $\frac{\sigma^2}{2\tau^2} - \frac{t-\mu}{\tau} - \frac{1}{2}\big(z - \frac{\sigma}{\tau}\big)^2 = -\frac{z^2}{2}$ (expand the square; $z\sigma/\tau = (t-\mu)/\tau$).
So $F' = \varphi(z)/\sigma + f - \varphi(z)/\sigma = f$. Also $F(-\infty) = 0$ and $F(\infty) = 1$: $f$ is a density.

## 2. Gamma fish gain

`GammaGain`, $g \sim \Gamma(r, b)$ (rate $b$), a stream set with $N$ events and expected count $S$ at $g = 1$.

**Marginal** (`log_marginal`): $\int \frac{b^r}{\Gamma(r)} g^{r-1}e^{-bg}\, g^N e^{-gS}\,dg = \frac{b^r\,\Gamma(N+r)}{\Gamma(r)\,(S+b)^{N+r}}$, the Gamma integral.

**Rescaling** (`tau`): after $N$ events on exposure $S_\text{prev}$ the posterior is $\Gamma(r+N,\, b+S_\text{prev})$, so
$-\log E[e^{-g(S_\text{abs} - S_\text{prev})}] = (r+N)\log\frac{b + S_\text{abs}}{b + S_\text{prev}}$ (Laplace transform of a Gamma).

## 3. Trial gain: integrating $h$ exactly

A stream's intensity is $\lambda(t) = g\,h\,d(t) + b(t)$: $d$ the driven and $b$ the burst rate, both predictable
(recovery $\rho$ included); a gain on the whole rate is $d = \lambda_0$, $b = 0$. $h \sim \Gamma(a, a)$.

**Poisson-binomial expansion** (`poisson_binomial`, `_nested_drive_integrand`, `sequential_trial_drive_taus`). With $p_i = d_i/(d_i + b_i)$,

$$\prod_{i=1}^{n}(x\,d_i + b_i) = \prod_{i}(d_i + b_i)\prod_i\big(p_i x + 1 - p_i\big) = \prod_i(d_i + b_i)\sum_{k} P_k\, x^k,$$

since $p x + 1 - p$ is the generating function of a Bernoulli($p$) and a product of generating functions is that of the sum:
$P$ is the distribution of the number of driven events. Multiplying by one factor gives the code's recursion
$P_{j+1,k} = (1-q)P_{j,k} + q\,P_{j,k-1}$; a censored interval ends in no event ($q = 0$, no factor). $\prod_i(d_i + b_i)$ does
not depend on the gains and cancels from every posterior ratio below, which is why the code omits it.

**Gamma moments**: for $Y \ge 0$,
$E_h\big[h^k e^{-hY}\big] = \frac{a^a}{\Gamma(a)}\int h^{a+k-1}e^{-(a+Y)h}\,dh = a^a\,(a)_k\,(a+Y)^{-(a+k)}$.

Together, for the first $j$ events of a trial and driven exposure $X$:

$$\int p(h) \prod_{i \le j}(g h d_i + b_i)\, e^{-ghX}\,dh = a^a \prod_{i\le j}(d_i+b_i)\; N_j(g, X),\qquad
N_j(g, X) = \sum_k P_{jk}\,(a)_k\, g^k\,(a + gX)^{-(a+k)},$$

which is `trial_factor`: $\log P_{jk} + \log(a)_k + k(\log g - \log(a+gX)) - a\log(a+gX)$, summed over $k$ in log space.

## 4. Time rescaling with fish and trial gains

`sequential_trial_drive_taus` (and `sequential_drive_taus`, its $a \to \infty$ limit, where $h = 1$).

**Model.** Fish gain $g \sim p(g)$; trial gains $h_m$ independent $\Gamma(a,a)$, independent of $g$. Given the gains, trials are
independent point processes with intensity $g h_m d(t) + b(t)$, where $d$ and $b$ depend only on the stream's past events
and on the fish's recorded other bouts, taken as given (as in the fit).

**The marginal intensity.** Let $\mathcal F_t$ be what is observed of the fish by $t$: its earlier trials and the current
trial up to $t$. Integrating out the gains, the process has intensity $\bar\lambda(t) = E[g h_m \mid \mathcal F_{t^-}]\,d(t) + b(t)$.
On an interval $(t_{j-1}, t_j]$ with driven exposure $S_j = \int d$ and burst exposure $B_j = \int b$, let
$G(s) = P(\text{no event in } (t_{j-1}, s] \mid \mathcal F_{t_{j-1}}) = E\big[e^{-g h \int_{t_{j-1}}^s d - \int_{t_{j-1}}^s b} \mid \mathcal F_{t_{j-1}}\big]$. Then

$$-\frac{d}{ds}\log G(s) = \frac{E\big[g h\, e^{-gh\int d} \mid \mathcal F_{t_{j-1}}\big]}{E\big[e^{-gh\int d} \mid \mathcal F_{t_{j-1}}\big]}\,d(s) + b(s) = \bar\lambda(s),$$

because conditioning on no event up to $s$ is exactly $\mathcal F_{s^-}$. Hence the compensator increment is

$$\tau_j = \int_{t_{j-1}}^{t_j}\bar\lambda = -\log G(t_j) = B_j - \log E\big[e^{-g h S_j} \mid \mathcal F_{t_{j-1}}\big].$$

By the time-rescaling theorem (Brown et al. 2002, Neural Computation 14:325) applied to $\bar\lambda$, the $\tau_j$ of completed
intervals are i.i.d. Exp(1) and a trial's last interval is a right-censored Exp(1): exactly under the model, with no
approximation in the gains. A fish's trials are taken in order and fish are independent, so this holds for all intervals together.

**The posterior factorizes.** The likelihood of trial $m'$ involves $h$ only through $h_{m'}$, so

$$p(g, h_m \mid \mathcal F_{t_{j-1}}) \propto p(g)\prod_{m' < m} L_{m'}(g)\;\, p(h_m)\prod_{i<j}(g h_m d_i + b_i)\,e^{-g h_m X_{<j}},\qquad L_{m'}(g) = \int p(h)\,\ell_{m'}(g, h)\,dh,$$

with $X_{<j}$ the trial's driven exposure before interval $j$. By section 3, $L_{m'}(g) \propto N_{n_{m'}}(g, X_{m'})$
(all of the trial's events, its whole driven exposure), and the $h_m$ integrals with and without $e^{-g h S_j}$ are
$N_{j-1}(g, X_{<j} + S_j)$ and $N_{j-1}(g, X_{<j})$ up to the same constant. So, with $w(g) = p(g)\prod_{m'<m} N_{n_{m'}}(g, X_{m'})$,

$$\tau_j = B_j - \log \int w(g)\,N_{j-1}(g,\, X_{<j} + S_j)\,dg + \log\int w(g)\,N_{j-1}(g,\, X_{<j})\,dg .$$

In the code: `W` is $\log w$ at the nodes in $g$ (from `log_nodes`, the prior's log weights), updated after each trial by
`trial_factor(log_coef[L:], g, X[-1:])`, the full trial's $\log N$; `before` and `after` are the two integrals. The only
approximations are the quadrature in $g$ and dropping nodes whose weight is below $e^{-50}$ of the largest.

**Limits.** As $a \to \infty$, $\Gamma(a, a) \to \delta_1$: $a^a (a)_k (a + gX)^{-(a+k)} \to e^{-gX}$ (as $(a)_k \sim a^k$ and
$(1 + gX/a)^{-(a+k)} \to e^{-gX}$), so $a^a N_j(g, X) \to e^{-gX}\sum_k P_{jk}\, g^k = e^{-gX}\prod_{i \le j}\frac{g d_i + b_i}{d_i + b_i}$,
and the formula becomes `sequential_drive_taus`:
$\tau_j = B_j - \log\int w(g)\prod_{i<j}(g d_i + b_i)e^{-g(X_{<j}+S_j)}dg + \log\int w(g)\prod_{i<j}(g d_i + b_i)e^{-gX_{<j}}dg$.
With $b = 0$ as well it is the closed form of section 2.

**Assumption to keep in mind.** Other bouts are conditioned on as given, as in the likelihood. If a fish's gains also
change how many other bouts it makes, the rescaling is exact for the model as fitted, not for that richer one.
