import streamlit as st
import sympy as sp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de parseo y formato
# ─────────────────────────────────────────────────────────────────────────────

def _parse(func_str: str):
    """Devuelve (callable, latex_str, expr, x_sym) o lanza excepción."""
    x = sp.Symbol("x")
    ld = {
        "x":    x,      "e":    sp.E,    "E":    sp.E,
        "pi":   sp.pi,  "ln":   sp.log,  "log":  sp.log,
        "exp":  sp.exp, "sin":  sp.sin,  "cos":  sp.cos,
        "tan":  sp.tan, "sqrt": sp.sqrt, "Abs":  sp.Abs, "abs": sp.Abs,
    }
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f    = sp.lambdify(x, expr, modules=["numpy"])
    return f, sp.latex(expr), expr, x


class _SafeFunc:
    """
    Wrapper que usa sp.limit() (L'Hôpital) cuando f(x) produce NaN o ∞.
    Registra los nodos afectados en lhopital_points.
    """
    def __init__(self, f_callable, expr, x_sym):
        self._f            = f_callable
        self.expr          = expr      # acceso público para las secciones de error
        self.x_sym         = x_sym
        self.lhopital_points: list = []

    def __call__(self, x_val: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                val = float(self._f(x_val))
            except Exception:
                val = float("nan")

        if np.isnan(val) or np.isinf(val):
            try:
                lim_val = float(sp.limit(self.expr, self.x_sym, x_val))
                if not (np.isnan(lim_val) or np.isinf(lim_val)):
                    if not any(abs(p[0] - x_val) < 1e-14 for p in self.lhopital_points):
                        self.lhopital_points.append((float(x_val), lim_val))
                    return lim_val
            except Exception:
                pass
        return val


def _parse_val(s: str) -> float:
    ld = {
        "e": sp.E, "E": sp.E, "pi": sp.pi,
        "ln": sp.log, "log": sp.log, "exp": sp.exp,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
    }
    return float(sp.sympify(s.strip().replace("^", "**"), locals=ld))


def _val_latex(s: str) -> str:
    ld = {
        "e": sp.E, "E": sp.E, "pi": sp.pi,
        "ln": sp.log, "log": sp.log, "exp": sp.exp,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
    }
    try:
        return sp.latex(sp.sympify(s.strip().replace("^", "**"), locals=ld))
    except Exception:
        return s.strip()


def _fmt(v, d: int = 6) -> str:
    try:
        fv = round(float(v), 10)
        if fv == int(fv):
            return str(int(fv))
        return f"{fv:.{d}f}"
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Métodos de integración
# ─────────────────────────────────────────────────────────────────────────────

def _midpoint(f, a, b, n):
    h     = (b - a) / n
    xs    = [a + i * h for i in range(n)]
    xmids = [xi + h / 2 for xi in xs]
    fmids = [float(f(xm)) for xm in xmids]
    return h * sum(fmids), h, xs, xmids, fmids


def _trapezoid(f, a, b, n):
    h   = (b - a) / n
    xs  = [a + i * h for i in range(n + 1)]
    fxs = [float(f(xi)) for xi in xs]
    result = h / 2 * (fxs[0] + 2 * sum(fxs[1:-1]) + fxs[-1])
    return result, h, xs, fxs


def _simpson13(f, a, b, n):
    h      = (b - a) / n
    xs     = [a + i * h for i in range(n + 1)]
    fxs    = [float(f(xi)) for xi in xs]
    coeffs = [1] + [4 if i % 2 == 1 else 2 for i in range(1, n)] + [1]
    result = h / 3 * sum(c * fx for c, fx in zip(coeffs, fxs))
    return result, h, xs, fxs, coeffs


def _simpson38(f, a, b, n):
    h   = (b - a) / n
    xs  = [a + i * h for i in range(n + 1)]
    fxs = [float(f(xi)) for xi in xs]
    coeffs = []
    for i in range(n + 1):
        if i == 0 or i == n:    coeffs.append(1)
        elif i % 3 == 0:        coeffs.append(2)
        else:                   coeffs.append(3)
    result = 3 * h / 8 * sum(c * fx for c, fx in zip(coeffs, fxs))
    return result, h, xs, fxs, coeffs


# ─────────────────────────────────────────────────────────────────────────────
# Componentes UI reutilizables
# ─────────────────────────────────────────────────────────────────────────────

_MAX_EXP = 6


def _result_cards(result: float, n: int, h: float, label: str, d: int):
    st.markdown(f"""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid #2563eb;">
            <div class="rc-label">Integral aproximada</div>
            <div class="rc-value">{result:.{d}f}</div>
            <div class="rc-sub">{label}</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #16a34a;">
            <div class="rc-label">n</div>
            <div class="rc-value">{n}</div>
            <div class="rc-sub">subintervalos</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #d97706;">
            <div class="rc-label">h</div>
            <div class="rc-value">{_fmt(h, d)}</div>
            <div class="rc-sub">paso</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _latex_h(a, b, n, h, d, a_str, b_str):
    al, bl = _val_latex(a_str), _val_latex(b_str)
    st.latex(rf"h = \frac{{b - a}}{{n}} = \frac{{{bl} - {al}}}{{{n}}} = {_fmt(h, d)}")


def _sum_display(values, d, label=r"\Sigma"):
    S = sum(values)
    if not values:
        st.latex(rf"{label} = 0")
        return S
    if len(values) <= _MAX_EXP:
        terms = " + ".join(_fmt(v, d) for v in values)
        st.latex(rf"{label} = {terms} = {_fmt(S, d)}")
    else:
        head = " + ".join(_fmt(values[i], d) for i in range(3))
        tail = " + ".join(_fmt(values[i], d) for i in range(len(values) - 3, len(values)))
        st.latex(rf"{label} = {head} + \cdots + {tail} = {_fmt(S, d)}")
    return S


def _weighted_sum_display(coeffs, fxs, d):
    W = sum(c * fx for c, fx in zip(coeffs, fxs))
    n = len(coeffs) - 1
    if n <= _MAX_EXP:
        terms = " + ".join(f"{c} \\cdot {_fmt(fx, d)}" for c, fx in zip(coeffs, fxs))
        st.latex(rf"\sum c_i \cdot f(x_i) = {terms} = {_fmt(W, d)}")
    else:
        head = " + ".join(f"{coeffs[i]} \\cdot {_fmt(fxs[i], d)}" for i in range(3))
        tail = " + ".join(f"{coeffs[i]} \\cdot {_fmt(fxs[i], d)}" for i in range(n - 2, n + 1))
        st.latex(rf"\sum c_i \cdot f(x_i) = {head} + \cdots + {tail} = {_fmt(W, d)}")
    return W


def _show_lhopital_warning(f: _SafeFunc, d: int):
    if not f.lhopital_points:
        return
    filas = "\n".join(
        f"| ${_fmt(xv, d)}$ | NaN (0/0 ó ∞/∞) | ${_fmt(lv, d)}$ |"
        for xv, lv in f.lhopital_points
    )
    st.info(
        "**📐 Regla de L'Hôpital aplicada automáticamente**\n\n"
        "La función presentó una indeterminación (0/0 ó ∞/∞) en uno o más nodos "
        "de la malla. Se calculó el **límite simbólico** con SymPy:\n\n"
        "| Nodo $x$ | Evaluación directa | Límite (L'Hôpital) |\n"
        "|:---:|:---:|:---:|\n"
        + filas
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error de Truncamiento
# ─────────────────────────────────────────────────────────────────────────────

# Configuración por método: orden de derivada, fórmula, coeficiente
_TRUNC_CFG = {
    "Rectángulo Medio": {
        "order":       2,
        "formula":     r"E_M \approx \frac{(b-a)\,h^2}{24}\,f''(\xi)",
        "coeff_fn":    lambda ba, h: ba * h**2 / 24,
        "coeff_tex":   lambda bl, al, h, d: rf"\frac{{({bl}-{al})\cdot {_fmt(h,d)}^2}}{{24}}",
        "dx_name":     "f''(x)",
        "dxi_name":    "f''(\\xi)",
    },
    "Trapecios": {
        "order":       2,
        "formula":     r"E_T \approx -\frac{(b-a)\,h^2}{12}\,f''(\xi)",
        "coeff_fn":    lambda ba, h: -ba * h**2 / 12,
        "coeff_tex":   lambda bl, al, h, d: rf"-\frac{{({bl}-{al})\cdot {_fmt(h,d)}^2}}{{12}}",
        "dx_name":     "f''(x)",
        "dxi_name":    "f''(\\xi)",
    },
    "Simpson 1/3": {
        "order":       4,
        "formula":     r"E_{S_{1/3}} \approx -\frac{(b-a)\,h^4}{180}\,f^{(4)}(\xi)",
        "coeff_fn":    lambda ba, h: -ba * h**4 / 180,
        "coeff_tex":   lambda bl, al, h, d: rf"-\frac{{({bl}-{al})\cdot {_fmt(h,d)}^4}}{{180}}",
        "dx_name":     "f^{(4)}(x)",
        "dxi_name":    "f^{(4)}(\\xi)",
    },
    "Simpson 3/8": {
        "order":       4,
        "formula":     r"E_{S_{3/8}} \approx -\frac{(b-a)\,h^4}{80}\,f^{(4)}(\xi)",
        "coeff_fn":    lambda ba, h: -ba * h**4 / 80,
        "coeff_tex":   lambda bl, al, h, d: rf"-\frac{{({bl}-{al})\cdot {_fmt(h,d)}^4}}{{80}}",
        "dx_name":     "f^{(4)}(x)",
        "dxi_name":    "f^{(4)}(\\xi)",
    },
}


def _show_truncation_error(method, expr, x_sym, a, b, n, h, xi, xi_str, d, a_str, b_str):
    """
    Sección de Error de Truncamiento.
    Muestra la fórmula, la derivada simbólica, y — si xi fue ingresado —
    evalúa el error puntual en ξ.
    """
    al, bl = _val_latex(a_str), _val_latex(b_str)
    cfg    = _TRUNC_CFG[method]

    st.markdown("---")
    st.markdown("### 📐 Error de Truncamiento")

    # — Fórmula general
    st.markdown(
        f"**Fórmula** (ξ ∈ ({a_str}, {b_str}) garantizado por el Teorema del Valor Medio):"
    )
    st.latex(cfg["formula"])

    # — Coeficiente numérico con los valores del problema
    coeff     = cfg["coeff_fn"](b - a, h)
    coeff_tex = cfg["coeff_tex"](bl, al, h, d)
    st.markdown("**Sustituyendo** h y (b − a):")
    st.latex(
        rf"E \approx {coeff_tex} \cdot {cfg['dxi_name']}"
        rf"= {_fmt(coeff, d)} \cdot {cfg['dxi_name']}"
    )

    # — Derivada simbólica
    try:
        deriv_expr   = sp.diff(expr, x_sym, cfg["order"])
        deriv_latex  = sp.latex(deriv_expr)
    except Exception as exc:
        st.warning(f"No se pudo calcular la derivada simbólica: {exc}")
        return

    st.markdown("**Derivada necesaria:**")
    st.latex(rf"{cfg['dx_name']} = {deriv_latex}")

    # — Evaluación en ξ (opcional)
    if xi is not None:
        # Advertencia si ξ ∉ [a, b]
        if not (a <= xi <= b):
            st.warning(
                f"⚠️ ξ = {_fmt(xi, d)} está fuera de [{a_str}, {b_str}]. "
                "El TVM garantiza ξ ∈ (a, b), pero se evalúa igual."
            )
        try:
            deriv_at_xi = float(deriv_expr.subs(x_sym, xi).evalf())
            error_val   = coeff * deriv_at_xi
            xi_latex    = _val_latex(xi_str)

            st.markdown(f"**Evaluando en ξ = {xi_latex}:**")
            st.latex(
                rf"\left.{cfg['dxi_name']}\right|_{{\xi={xi_latex}}}"
                rf"= {_fmt(deriv_at_xi, d)}"
            )
            st.latex(
                rf"E \approx {_fmt(coeff, d)} \times {_fmt(deriv_at_xi, d)}"
                rf"= {_fmt(error_val, d)}"
            )
            st.latex(rf"\boxed{{\ E \approx {error_val:.{d}f}\ }}")
        except Exception as exc:
            st.warning(f"No se pudo evaluar la derivada en ξ: {exc}")
    else:
        st.info(
            "💡 Ingresá un valor de **ξ** en el campo de entrada para calcular "
            "el error de truncamiento puntual."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error Real (vs. integral exacta)
# ─────────────────────────────────────────────────────────────────────────────

def _show_real_error(expr, x_sym, a, b, result, d, a_str, b_str):
    """
    Calcula la integral exacta con SymPy y compara con el resultado numérico.
    """
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### 🎯 Error Real (vs. integral exacta)")

    try:
        with st.spinner("Calculando integral exacta con SymPy…"):
            exact_sym = sp.integrate(expr, (x_sym, a, b))
            exact_val = float(exact_sym.evalf(15))
        if np.isnan(exact_val) or np.isinf(exact_val):
            raise ValueError("La integral diverge o no es finita.")
    except Exception as exc:
        st.warning(f"⚠️ No se pudo obtener la integral exacta: {exc}")
        return

    abs_err = abs(result - exact_val)
    rel_err = abs_err / abs(exact_val) * 100 if abs(exact_val) > 1e-15 else float("inf")
    rel_str = f"{rel_err:.4f} %" if not np.isinf(rel_err) else "∞"

    # Mostrar integral exacta
    try:
        st.latex(
            rf"\int_{{{al}}}^{{{bl}}} \left({sp.latex(expr)}\right) dx"
            rf"= {sp.latex(exact_sym)} \approx {_fmt(exact_val, d)}"
        )
    except Exception:
        st.latex(rf"\text{{Valor exacto}} \approx {_fmt(exact_val, d)}")

    # Cards de error
    st.markdown(f"""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid #7c3aed;">
            <div class="rc-label">Valor exacto</div>
            <div class="rc-value">{exact_val:.{d}f}</div>
            <div class="rc-sub">integral simbólica</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #dc2626;">
            <div class="rc-label">Error absoluto</div>
            <div class="rc-value">{abs_err:.{d}f}</div>
            <div class="rc-sub">|aprox − exacto|</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #ea580c;">
            <div class="rc-label">Error relativo</div>
            <div class="rc-value">{rel_str}</div>
            <div class="rc-sub">|error / exacto| × 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(
        rf"|E_{{real}}| = \left|{_fmt(result, d)} - {_fmt(exact_val, d)}\right|"
        rf"= {_fmt(abs_err, d)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paso a paso — Rectángulo Medio
# ─────────────────────────────────────────────────────────────────────────────

def _show_midpoint(result, h, a, b, n, xs, xmids, fmids, d, a_str, b_str):
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### Paso 1 — Fórmula del rectángulo medio")
    st.latex(
        r"\int_a^b f(x)\,dx"
        r"\;\approx\; h \sum_{i=0}^{n-1} f\!\left(\bar{x}_i\right)"
        r"\,,\qquad \bar{x}_i = a + \left(i + \tfrac{1}{2}\right)h"
    )

    st.markdown("### Paso 2 — Cálculo de h")
    _latex_h(a, b, n, h, d, a_str, b_str)

    st.markdown("### Paso 3 — Tabla de nodos y puntos medios")
    rows = [
        r"| $i$ | $x_i$ | $\bar{x}_i$ | $f(\bar{x}_i)$ |",
        "|:---:|:---:|:---:|:---:|",
    ]
    for i in range(n):
        rows.append(f"| {i} | {xs[i]:.{d}f} | {xmids[i]:.{d}f} | {fmids[i]:.{d}f} |")
    st.markdown("\n".join(rows))

    st.markdown("### Paso 4 — Suma y resultado")
    S = _sum_display(fmids, d, label=rf"\sum_{{i=0}}^{{{n-1}}} f(\bar{{x}}_i)")
    st.latex(
        rf"\int_{{{al}}}^{{{bl}}} f(x)\,dx"
        rf"\;\approx\; h \cdot \Sigma"
        rf"= {_fmt(h, d)} \cdot {_fmt(S, d)} = {result:.{d}f}"
    )
    st.latex(rf"\boxed{{\ \int_{{{al}}}^{{{bl}}} f(x)\,dx \approx {result:.{d}f}\ }}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso a paso — Trapecios
# ─────────────────────────────────────────────────────────────────────────────

def _show_trapezoid(result, h, a, b, n, xs, fxs, d, a_str, b_str):
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### Paso 1 — Fórmula del trapecio compuesto")
    st.latex(
        r"\int_a^b f(x)\,dx"
        r"\;\approx\; \frac{h}{2}"
        r"\!\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]"
    )

    st.markdown("### Paso 2 — Cálculo de h")
    _latex_h(a, b, n, h, d, a_str, b_str)

    st.markdown("### Paso 3 — Tabla de nodos")
    rows = [r"| $i$ | $x_i$ | $f(x_i)$ |", "|:---:|:---:|:---:|"]
    for i in range(n + 1):
        rows.append(f"| {i} | {xs[i]:.{d}f} | {fxs[i]:.{d}f} |")
    st.markdown("\n".join(rows))

    st.markdown("### Paso 4 — Suma ponderada y resultado")
    inner = fxs[1:-1]
    S_in  = sum(inner)
    total = fxs[0] + 2 * S_in + fxs[-1]
    st.latex(rf"f(x_0) = {_fmt(fxs[0], d)}\,,\qquad f(x_n) = {_fmt(fxs[-1], d)}")
    if inner:
        _sum_display(inner, d, label=rf"\sum_{{i=1}}^{{{n-1}}} f(x_i)")
        st.latex(
            rf"f(x_0) + 2\cdot\Sigma_{{\text{{int}}}} + f(x_n)"
            rf"= {_fmt(fxs[0],d)} + 2\cdot{_fmt(S_in,d)} + {_fmt(fxs[-1],d)} = {_fmt(total,d)}"
        )
    else:
        st.latex(rf"f(x_0)+f(x_n)={_fmt(fxs[0],d)}+{_fmt(fxs[-1],d)}={_fmt(total,d)}")
    st.latex(
        rf"\int_{{{al}}}^{{{bl}}} f(x)\,dx"
        rf"\;\approx\; \frac{{{_fmt(h,d)}}}{{2}}\cdot{_fmt(total,d)} = {result:.{d}f}"
    )
    st.latex(rf"\boxed{{\ \int_{{{al}}}^{{{bl}}} f(x)\,dx \approx {result:.{d}f}\ }}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso a paso — Simpson 1/3
# ─────────────────────────────────────────────────────────────────────────────

def _show_simpson13(result, h, a, b, n, xs, fxs, coeffs, d, a_str, b_str):
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### Paso 1 — Fórmula de Simpson 1/3 compuesto")
    st.latex(
        r"\int_a^b f(x)\,dx"
        r"\;\approx\; \frac{h}{3}"
        r"\!\left[f(x_0)+4\,f(x_1)+2\,f(x_2)+\cdots+4\,f(x_{n-1})+f(x_n)\right]"
    )
    st.markdown("Patrón de coeficientes: **1 — 4 — 2 — 4 — 2 — ⋯ — 4 — 1**")

    st.markdown("### Paso 2 — Cálculo de h")
    _latex_h(a, b, n, h, d, a_str, b_str)

    st.markdown("### Paso 3 — Tabla de nodos y coeficientes")
    rows = [
        r"| $i$ | $x_i$ | $f(x_i)$ | $c_i$ | $c_i \cdot f(x_i)$ |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for i in range(n + 1):
        ci = coeffs[i]
        rows.append(f"| {i} | {xs[i]:.{d}f} | {fxs[i]:.{d}f} | {ci} | {ci*fxs[i]:.{d}f} |")
    st.markdown("\n".join(rows))

    st.markdown("### Paso 4 — Suma ponderada y resultado")
    W = _weighted_sum_display(coeffs, fxs, d)
    st.latex(
        rf"\int_{{{al}}}^{{{bl}}} f(x)\,dx"
        rf"\;\approx\; \frac{{{_fmt(h,d)}}}{{3}}\cdot{_fmt(W,d)} = {result:.{d}f}"
    )
    st.latex(rf"\boxed{{\ \int_{{{al}}}^{{{bl}}} f(x)\,dx \approx {result:.{d}f}\ }}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso a paso — Simpson 3/8
# ─────────────────────────────────────────────────────────────────────────────

def _show_simpson38(result, h, a, b, n, xs, fxs, coeffs, d, a_str, b_str):
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### Paso 1 — Fórmula de Simpson 3/8 compuesto")
    st.latex(
        r"\int_a^b f(x)\,dx"
        r"\;\approx\; \frac{3h}{8}"
        r"\!\left[f(x_0)+3\,f(x_1)+3\,f(x_2)+2\,f(x_3)+\cdots+f(x_n)\right]"
    )
    st.markdown("Patrón de coeficientes: **1 — 3 — 3 — 2 — 3 — 3 — 2 — ⋯ — 3 — 3 — 1**")

    st.markdown("### Paso 2 — Cálculo de h")
    _latex_h(a, b, n, h, d, a_str, b_str)

    st.markdown("### Paso 3 — Tabla de nodos y coeficientes")
    rows = [
        r"| $i$ | $x_i$ | $f(x_i)$ | $c_i$ | $c_i \cdot f(x_i)$ |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for i in range(n + 1):
        ci = coeffs[i]
        rows.append(f"| {i} | {xs[i]:.{d}f} | {fxs[i]:.{d}f} | {ci} | {ci*fxs[i]:.{d}f} |")
    st.markdown("\n".join(rows))

    st.markdown("### Paso 4 — Suma ponderada y resultado")
    W = _weighted_sum_display(coeffs, fxs, d)
    st.latex(
        rf"\int_{{{al}}}^{{{bl}}} f(x)\,dx"
        rf"\;\approx\; \frac{{3\cdot{_fmt(h,d)}}}{{8}}\cdot{_fmt(W,d)} = {result:.{d}f}"
    )
    st.latex(rf"\boxed{{\ \int_{{{al}}}^{{{bl}}} f(x)\,dx \approx {result:.{d}f}\ }}")


# ─────────────────────────────────────────────────────────────────────────────
# Página principal
# ─────────────────────────────────────────────────────────────────────────────

def run():
    st.markdown("""
    <style>
    html { font-size: 20px; }

    div.stButton { width: 100%; }
    div.stButton > button {
        background-color: #dc2626; color: #fff; border: none;
        border-radius: 6px; width: 100%; padding: .7rem 0;
        font-size: 1rem; font-weight: 600; cursor: pointer; transition: background .15s;
    }
    div.stButton > button:hover  { background-color: #b91c1c; color: #fff; }
    div.stButton > button:active { background-color: #991b1b; color: #fff; }

    table { margin: 0.5rem auto 0 auto; }
    table th, table td { text-align: center !important; }

    .result-cards { display: flex; gap: 1.2rem; margin: 0.6rem 0 1.6rem 0; flex-wrap: wrap; }
    .result-card {
        flex: 1; min-width: 140px; border-radius: 10px;
        padding: 1.3rem 1rem 1rem 1rem; text-align: center;
        border: 1px solid rgba(128,128,128,.18);
    }
    .result-card .rc-label {
        font-size: .72rem; font-weight: 700; letter-spacing: .09em;
        text-transform: uppercase; opacity: .5; margin-bottom: .55rem;
    }
    .result-card .rc-value {
        font-size: 1.8rem; font-weight: 700;
        font-family: 'Courier New', monospace; line-height: 1;
    }
    .result-card .rc-sub {
        font-size: .9rem; opacity: .35; margin-top: .35rem; font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Integración Numérica — Newton-Cotes")

    # ── Selector de método ────────────────────────────────────────────────────
    method = st.selectbox(
        "Método de integración",
        ["Rectángulo Medio", "Trapecios", "Simpson 1/3", "Simpson 3/8"],
    )

    # ── f(x) ─────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)", value="x**2",
        placeholder="Ej: sin(x),  exp(-x),  x**3 - 2*x + 1",
    )

    f = latex_f = None
    if func_str.strip():
        try:
            f_raw, latex_f, _expr, _x_sym = _parse(func_str)
            f = _SafeFunc(f_raw, _expr, _x_sym)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Intervalo [a, b] ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    a_str = col_a.text_input("a", value="0", placeholder="Ej: 0,  pi,  e")
    b_str = col_b.text_input("b", value="1", placeholder="Ej: 1,  pi/2,  2*pi")

    a = b = None
    try:
        a = _parse_val(a_str)
    except Exception:
        col_a.error("Valor inválido")
    try:
        b = _parse_val(b_str)
    except Exception:
        col_b.error("Valor inválido")

    # ── Render de la integral ─────────────────────────────────────────────────
    if f is not None and a is not None and b is not None and latex_f:
        al, bl = _val_latex(a_str), _val_latex(b_str)
        st.latex(rf"\int_{{{al}}}^{{{bl}}} \left({latex_f}\right) dx")

    # ── n, decimales y ξ ─────────────────────────────────────────────────────
    col_n, col_d = st.columns(2)
    n_str = col_n.text_input("n  (subintervalos)", value="4")
    d_str = col_d.text_input("Decimales en tabla", value="6")

    n = decimals = None
    try:
        n = int(n_str)
        if n <= 0:
            raise ValueError
    except Exception:
        col_n.error("Ingresá un entero positivo")
    try:
        decimals = int(d_str)
        if decimals < 0:
            raise ValueError
    except Exception:
        col_d.error("Valor inválido")

    # ── ξ (opcional) ─────────────────────────────────────────────────────────
    xi_str = st.text_input(
        "ξ (xi) — punto para el error de truncamiento  *(opcional)*",
        value="",
        placeholder="Ej: 0.5  |  pi/4  |  (a+b)/2  — debe pertenecer a [a, b]",
        help=(
            "La fórmula del error de truncamiento depende de la derivada de f "
            "evaluada en algún ξ ∈ (a, b) (Teorema del Valor Medio). "
            "Si ingresás un valor, se calcula el error puntual para ese ξ."
        ),
    )
    xi = None
    if xi_str.strip():
        try:
            xi = _parse_val(xi_str)
        except Exception:
            st.error("Valor de ξ inválido. Usá números o expresiones como pi/4, 0.5, e.")

    # ── Preview de h ──────────────────────────────────────────────────────────
    if a is not None and b is not None and n is not None:
        h_prev = (b - a) / n
        al, bl = _val_latex(a_str), _val_latex(b_str)
        st.latex(
            rf"h = \frac{{b - a}}{{n}} = \frac{{{bl} - {al}}}{{{n}}} = {_fmt(h_prev)}"
        )

    # ── Restricción de n según el método ─────────────────────────────────────
    if method == "Simpson 1/3":
        st.info("Simpson 1/3 requiere que **n sea par**  (2, 4, 6, …).")
    elif method == "Simpson 3/8":
        st.info("Simpson 3/8 requiere que **n sea múltiplo de 3**  (3, 6, 9, …).")

    # ── Botón ─────────────────────────────────────────────────────────────────
    if st.button("Calcular integral", use_container_width=True):

        errors = []
        if not f:
            errors.append("Ingresá una función válida.")
        if a is None or b is None:
            errors.append("Ingresá valores válidos para a y b.")
        elif a >= b:
            errors.append("Se requiere a < b.")
        if n is None:
            errors.append("Ingresá un valor válido para n.")
        if decimals is None:
            errors.append("Ingresá un valor válido para los decimales.")
        if errors:
            for msg in errors:
                st.error(msg)
            return

        if method == "Simpson 1/3" and n % 2 != 0:
            st.error(
                f"Simpson 1/3 requiere n par. "
                f"Probá con **n = {n + 1}** (o cualquier número par)."
            )
            return
        if method == "Simpson 3/8" and n % 3 != 0:
            st.error(
                f"Simpson 3/8 requiere n múltiplo de 3. "
                f"Probá con **n = {n + (3 - n % 3)}** (o cualquier múltiplo de 3)."
            )
            return

        d = decimals
        f.lhopital_points.clear()

        # ── Despacho por método ───────────────────────────────────────────────
        if method == "Rectángulo Medio":
            result, h, xs, xmids, fmids = _midpoint(f, a, b, n)
            _result_cards(result, n, h, "Rectángulo Medio", d)
            _show_lhopital_warning(f, d)
            _show_midpoint(result, h, a, b, n, xs, xmids, fmids, d, a_str, b_str)

        elif method == "Trapecios":
            result, h, xs, fxs = _trapezoid(f, a, b, n)
            _result_cards(result, n, h, "Trapecios", d)
            _show_lhopital_warning(f, d)
            _show_trapezoid(result, h, a, b, n, xs, fxs, d, a_str, b_str)

        elif method == "Simpson 1/3":
            result, h, xs, fxs, coeffs = _simpson13(f, a, b, n)
            _result_cards(result, n, h, "Simpson 1/3", d)
            _show_lhopital_warning(f, d)
            _show_simpson13(result, h, a, b, n, xs, fxs, coeffs, d, a_str, b_str)

        elif method == "Simpson 3/8":
            result, h, xs, fxs, coeffs = _simpson38(f, a, b, n)
            _result_cards(result, n, h, "Simpson 3/8", d)
            _show_lhopital_warning(f, d)
            _show_simpson38(result, h, a, b, n, xs, fxs, coeffs, d, a_str, b_str)

        # ── Secciones de error (comunes a todos los métodos) ──────────────────
        if np.isnan(result) or np.isinf(result):
            st.error(
                "El resultado es NaN o ∞. Verificá la función y el intervalo de integración."
            )
            return

        _show_truncation_error(
            method, f.expr, f.x_sym, a, b, n, h, xi, xi_str, d, a_str, b_str
        )
        _show_real_error(f.expr, f.x_sym, a, b, result, d, a_str, b_str)


if __name__ == "__main__":
    run()