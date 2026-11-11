"""
The identities of DERIVATIONS.md, proved symbolically (sympy): each test reduces the difference of the
two sides to zero, for all parameter values.
"""
import pytest

sp = pytest.importorskip("sympy")

t, s, mu, x, X, Y, g, h = sp.symbols("t s mu x X Y g h", real=True)
sigma, tau, a, r, b, S, D = sp.symbols("sigma tau a r b S Delta", positive=True)
N, k = sp.symbols("N k", positive=True, integer=True)
Phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2


def assert_zero(expression) -> None:
    assert sp.simplify(sp.expand(sp.unpolarify(expression))) == 0


class TestSurvivalAntiderivatives:
    """Section 1: each integral_func's antiderivative differentiates back to its hazard."""

    def test_gaussian_bump(self) -> None:
        F = sigma * sp.sqrt(2 * sp.pi) * Phi((t - mu) / sigma)
        assert_zero(sp.diff(F, t) - sp.exp(-(t - mu) ** 2 / (2 * sigma**2)))

    def test_gamma_pulse(self) -> None:
        assert_zero(sp.diff(-(tau + t) * sp.exp(-t / tau), t) - t / tau * sp.exp(-t / tau))

    def test_exgaussian(self) -> None:
        z, kappa = (t - mu) / sigma, sigma / tau
        f = sp.exp(sigma**2 / (2 * tau**2) - (t - mu) / tau) * Phi(z - kappa) / tau
        code = sp.exp(kappa * (kappa / 2 - z) + sp.log(Phi(z - kappa)) - sp.log(tau / sigma) - sp.log(sigma))
        F = Phi(z) - tau * f
        assert_zero(sp.expand_log(sp.log(code), force=True) - sp.expand_log(sp.log(f), force=True))  # exgaussian_shape is f
        assert_zero(sp.diff(F, t) - f)
        assert sp.limit(F, t, sp.oo) == 1 and sp.limit(F, t, -sp.oo) == 0


class TestGammaGain:
    """Section 2: GammaGain.log_marginal and GammaGain.tau."""

    def test_marginal(self) -> None:
        density = b**r / sp.gamma(r) * g ** (r - 1) * sp.exp(-b * g)
        marginal = sp.integrate(density * g**N * sp.exp(-g * S), (g, 0, sp.oo), conds="none")
        assert_zero(sp.gammasimp(marginal / (b**r * sp.gamma(N + r) / (sp.gamma(r) * (S + b) ** (N + r)))) - 1)

    def test_tau(self) -> None:
        shape, rate = r + N, b + S
        posterior = rate**shape / sp.gamma(shape) * g ** (shape - 1) * sp.exp(-rate * g)
        survival = sp.integrate(posterior * sp.exp(-g * D), (g, 0, sp.oo), conds="none")
        assert_zero(sp.simplify(-sp.log(survival)) - shape * sp.log((rate + D) / rate))


class TestTrialGain:
    """Section 3: the Gamma moments, the Poisson-binomial expansion and the trial factor N_j; section 4's a -> oo limit."""

    def test_gamma_moment(self) -> None:
        Yp = sp.Symbol("Y", positive=True)
        moment = sp.integrate(a**a / sp.gamma(a) * h ** (a + k - 1) * sp.exp(-(a + Yp) * h), (h, 0, sp.oo), conds="none")
        assert_zero(sp.gammasimp(moment / (a**a * sp.rf(a, k) * (a + Yp) ** (-(a + k))).rewrite(sp.gamma)) - 1)

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_poisson_binomial_expansion(self, n: int) -> None:
        d, c = sp.symbols(f"d1:{n + 1}", positive=True), sp.symbols(f"b1:{n + 1}", positive=True)
        P = [sp.Integer(1)]
        for di, bi in zip(d, c):  # the code's recursion
            q = di / (di + bi)
            P = [(1 - q) * (P[j] if j < len(P) else 0) + q * (P[j - 1] if j > 0 else 0) for j in range(len(P) + 1)]
        expansion = sp.prod([di + bi for di, bi in zip(d, c)]) * sum(P[j] * x**j for j in range(n + 1))
        assert_zero(sp.together(sp.prod([x * di + bi for di, bi in zip(d, c)]) - expansion))

    @pytest.mark.parametrize("n", [1, 2])
    def test_trial_factor(self, n: int) -> None:
        """Integral over h of p(h) prod (g h d_i + b_i) e^(-g h X) = a^a prod(d_i + b_i) N_n(g, X)."""
        gp, Xp = sp.symbols("g X", positive=True)
        d, c = sp.symbols(f"d1:{n + 1}", positive=True), sp.symbols(f"b1:{n + 1}", positive=True)
        p_h = a**a / sp.gamma(a) * h ** (a - 1) * sp.exp(-a * h)
        integrand = sp.expand(p_h * sp.prod([gp * h * di + bi for di, bi in zip(d, c)]) * sp.exp(-gp * h * Xp))
        integral = sum(sp.integrate(term, (h, 0, sp.oo), conds="none") for term in sp.Add.make_args(integrand))
        P = [sp.Integer(1)]
        for di, bi in zip(d, c):
            q = di / (di + bi)
            P = [(1 - q) * (P[j] if j < len(P) else 0) + q * (P[j - 1] if j > 0 else 0) for j in range(len(P) + 1)]
        N_n = sum(P[j] * sp.rf(a, j) * gp**j * (a + gp * Xp) ** (-(a + j)) for j in range(n + 1))
        expected = a**a * sp.prod([di + bi for di, bi in zip(d, c)]) * N_n
        assert_zero(sp.simplify(sp.gammasimp((integral - expected).rewrite(sp.gamma))))

    @pytest.mark.parametrize("j", [0, 1, 3])
    def test_infinite_a_limit(self, j: int) -> None:
        """a^a (a)_j (a + g X)^-(a + j) -> e^(-g X) as a -> oo: the trial gain disappears."""
        gp, Xp = sp.symbols("g X", positive=True)
        assert sp.simplify(sp.limit(a**a * sp.rf(a, j) * (a + gp * Xp) ** (-(a + j)), a, sp.oo) - sp.exp(-gp * Xp)) == 0
