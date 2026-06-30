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


def _max_abs_on_interval(expr, x_sym, a, b):
    """
    Calcula de forma EXACTA (simbólica) el máximo de |expr| en [a, b].

    Estrategia (estándar de cálculo / análisis numérico):
      1. Encontrar los puntos críticos de expr en (a, b): expr' = 0.
      2. Filtrar los que son reales y caen dentro de (a, b).
      3. Evaluar |expr| en los puntos críticos y en los extremos a, b.
      4. El máximo de esos valores es el máximo absoluto buscado.

    Devuelve (max_val: float, x_max: float, candidatos: list[(x, |expr(x)|)])
    """
    deriv = sp.diff(expr, x_sym)

    # Puntos críticos (raíces de la derivada) dentro de (a, b)
    crit_points = []
    try:
        sols = sp.solve(sp.Eq(deriv, 0), x_sym)
    except Exception:
        sols = []

    for s in sols:
        try:
            s_val = complex(s.evalf())
        except Exception:
            continue
        # descartar soluciones no reales
        if abs(s_val.imag) > 1e-9:
            continue
        s_real = s_val.real
        if a - 1e-9 <= s_real <= b + 1e-9:
            # clamp por seguridad numérica a los bordes del intervalo
            s_real = min(max(s_real, a), b)
            crit_points.append(s_real)

    # Candidatos: extremos del intervalo + puntos críticos
    candidatos_x = [a, b] + crit_points

    # Evaluar |expr| en cada candidato (numéricamente, vía SymPy para exactitud)
    candidatos = []
    for xv in candidatos_x:
        try:
            val = abs(float(expr.subs(x_sym, xv).evalf()))
            if np.isnan(val) or np.isinf(val):
                continue
            candidatos.append((xv, val))
        except Exception:
            continue

    if not candidatos:
        raise ValueError("No se pudo evaluar la derivada en el intervalo dado.")

    x_max, max_val = max(candidatos, key=lambda t: t[1])
    return max_val, x_max, candidatos


def _show_max_error(method, deriv_expr, deriv_latex, a, b, h, d, a_str, b_str):
    """
    Error Máximo Posible (cota de error): usa el máximo EXACTO de |derivada|
    en [a, b] en lugar de un ξ puntual. No requiere que el usuario ingrese ξ.
    """
    al, bl = _val_latex(a_str), _val_latex(b_str)
    cfg    = _TRUNC_CFG[method]
    coeff  = cfg["coeff_fn"](b - a, h)

    st.markdown("#### 🔺 Error Máximo Posible (cota de error)")
    st.markdown(
        "En lugar de evaluar en un ξ puntual, acotamos el error usando el "
        f"**máximo de $|{cfg['dx_name']}|$** en todo el intervalo "
        f"$[{a_str}, {b_str}]$. Esto da el **peor caso posible** del error, "
        "sin necesidad de conocer ξ."
    )

    x_sym = sp.Symbol("x")
    try:
        max_val, x_max, candidatos = _max_abs_on_interval(deriv_expr, x_sym, a, b)
    except Exception as exc:
        st.warning(f"No se pudo calcular el máximo exacto de la derivada: {exc}")
        return

    # Tabla de candidatos (extremos + puntos críticos)
    rows = [
        r"| Punto candidato | $\lvert {} \rvert$ |".format(cfg["dx_name"]),
        "|:---:|:---:|",
    ]
    for xv, val in sorted(candidatos, key=lambda t: t[0]):
        rows.append(f"| {_fmt(xv, d)} | {_fmt(val, d)} |")
    st.markdown(
        "**Candidatos a máximo** (extremos del intervalo + puntos críticos "
        f"donde $\\frac{{d}}{{dx}}{cfg['dx_name']} = 0$):\n\n"
        + "\n".join(rows)
    )

    deriv_func_name = {
        "f''(x)":     "f''",
        "f^{(4)}(x)": "f^{(4)}",
    }.get(cfg["dx_name"], cfg["dx_name"])
    st.latex(
        rf"\max_{{x\,\in\,[{al},\,{bl}]}} \left|{cfg['dx_name']}\right|"
        rf"= \left|{deriv_func_name}({_fmt(x_max, d)})\right|"
        rf"= {_fmt(max_val, d)}"
    )

    error_max = abs(coeff) * max_val
    st.markdown("**Cota del error máximo:**")
    st.latex(
        rf"E_{{max}} \approx \left|{cfg['coeff_tex'](bl, al, h, d)}\right| \cdot "
        rf"\max\left|{cfg['dx_name']}\right| "
        rf"= {_fmt(abs(coeff), d)} \cdot {_fmt(max_val, d)} = {_fmt(error_max, d)}"
    )
    st.latex(rf"\boxed{{\ |E| \leq {error_max:.{d}f}\ }}")


def _show_truncation_error(method, expr, x_sym, a, b, n, h, xi, xi_str, d, a_str, b_str):
    """
    Sección de Error de Truncamiento.
    Muestra la fórmula, la derivada simbólica, el ERROR MÁXIMO POSIBLE
    (calculado de forma exacta a partir del máximo de la derivada en [a,b],
    sin necesitar ξ) y — si el usuario ingresó ξ — el error puntual en ξ.
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

    # — Error Máximo Posible (NO requiere ξ) ──────────────────────────────────
    st.markdown("")
    _show_max_error(method, deriv_expr, deriv_latex, a, b, h, d, a_str, b_str)

    # — Evaluación puntual en ξ (opcional) ────────────────────────────────────
    st.markdown("")
    st.markdown("#### 📍 Error Puntual en ξ (opcional)")
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
            "además el error de truncamiento puntual (no es necesario para el "
            "error máximo posible, que ya se calculó arriba)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error Real (vs. integral exacta)
# ─────────────────────────────────────────────────────────────────────────────

def _show_real_error(expr, x_sym, a, b, result, d, a_str, b_str, exact_override=None):
    """
    Calcula la integral exacta con SymPy y compara con el resultado numérico.

    Si `exact_override` no es None, se usa ese valor directamente como
    integral exacta (ingresado manualmente por el usuario) en lugar de
    intentar resolverla simbólicamente con SymPy. Esto cubre los casos en
    los que SymPy no logra hallar una forma cerrada.
    """
    al, bl = _val_latex(a_str), _val_latex(b_str)

    st.markdown("---")
    st.markdown("### 🎯 Error Real (vs. integral exacta)")

    exact_sym = None

    if exact_override is not None:
        exact_val = exact_override
        st.info(
            "ℹ️ Se está usando el **valor exacto ingresado manualmente** "
            "(no se intentó resolver la integral con SymPy)."
        )
    else:
        try:
            with st.spinner("Calculando integral exacta con SymPy…"):
                exact_sym = sp.integrate(expr, (x_sym, a, b))
                exact_val = float(exact_sym.evalf(15))
            if np.isnan(exact_val) or np.isinf(exact_val):
                raise ValueError("La integral diverge o no es finita.")
        except Exception as exc:
            st.warning(f"⚠️ No se pudo obtener la integral exacta: {exc}")
            st.info(
                "💡 Si ya conocés el valor exacto de la integral, podés "
                "ingresarlo en el campo **«Valor exacto de la integral»** "
                "(antes del botón Calcular) y se usará en su lugar."
            )
            return

    abs_err = abs(result - exact_val)
    rel_err = abs_err / abs(exact_val) * 100 if abs(exact_val) > 1e-15 else float("inf")
    rel_str = f"{rel_err:.4f} %" if not np.isinf(rel_err) else "∞"

    # Mostrar integral exacta
    if exact_sym is not None:
        try:
            st.latex(
                rf"\int_{{{al}}}^{{{bl}}} \left({sp.latex(expr)}\right) dx"
                rf"= {sp.latex(exact_sym)} \approx {_fmt(exact_val, d)}"
            )
        except Exception:
            st.latex(rf"\text{{Valor exacto}} \approx {_fmt(exact_val, d)}")
    else:
        st.latex(
            rf"\int_{{{al}}}^{{{bl}}} \left({sp.latex(expr)}\right) dx"
            rf"= {_fmt(exact_val, d)} \quad \text{{(valor ingresado manualmente)}}"
        )

    # Cards de error
    st.markdown(f"""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid #7c3aed;">
            <div class="rc-label">Valor exacto</div>
            <div class="rc-value">{exact_val:.{d}f}</div>
            <div class="rc-sub">{"integral simbólica" if exact_sym is not None else "ingresado manualmente"}</div>
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

    # ── Teoría (desplegable, arriba de todo) ───────────────────────────────────
    with st.expander("📘 Teoría: ¿Qué son los métodos de Newton-Cotes?", expanded=False):
        st.markdown(r"""
### ¿Qué son?

Los **métodos de Newton-Cotes** son una familia de técnicas para aproximar una
integral definida

$$
\int_a^b f(x)\,dx
$$

cuando no se puede (o no conviene) resolver la integral de forma analítica.
La idea común a todos ellos es:

1. Dividir el intervalo $[a, b]$ en $n$ subintervalos iguales de ancho

$$
h = \frac{b-a}{n}
$$

2. **Reemplazar $f(x)$ por un polinomio** que pasa exactamente por los valores
   de $f$ en los nodos $x_0, x_1, \dots, x_n$ (interpolación con nodos
   equiespaciados).
3. **Integrar ese polinomio en vez de $f(x)$**, porque integrar un polinomio
   sí tiene una fórmula cerrada y exacta.

Lo único que cambia de un método a otro es **el grado del polinomio
interpolante** que se usa en cada tramo: a mayor grado, mayor precisión, pero
también más restricciones sobre cómo se puede elegir $n$.
""")

        st.markdown("### Los 4 métodos implementados en esta calculadora")

        st.markdown(r"""
#### 1️⃣ Rectángulo Medio (polinomio de grado 0, regla **abierta**)

Aproxima $f$ en cada subintervalo por una **constante**: el valor de $f$ en el
punto medio. Geométricamente, el área bajo la curva se aproxima con
rectángulos cuya altura es $f(\bar{x}_i)$, el punto medio de cada tramo.
""")
        st.latex(
            r"\int_a^b f(x)\,dx \;\approx\; h \sum_{i=0}^{n-1} f(\bar{x}_i)"
            r"\,,\qquad \bar{x}_i = a + \left(i+\tfrac12\right)h"
        )
        st.markdown(r"""
**Paso a paso:**
1. Calcular $h = (b-a)/n$.
2. Para cada subintervalo $i = 0, \dots, n-1$, calcular el punto medio
   $\bar{x}_i$ y evaluar $f(\bar{x}_i)$.
3. Sumar todos los $f(\bar{x}_i)$ y multiplicar por $h$.

**Restricciones:** ninguna sobre $n$ — funciona con cualquier entero $n \ge 1$.
Se llama "abierta" porque nunca evalúa $f$ en los extremos $a$ y $b$, lo cual
es útil si $f$ no está definida ahí.
""")

        st.markdown(r"""
#### 2️⃣ Trapecios (polinomio de grado 1, regla **cerrada**)

Aproxima $f$ en cada subintervalo por una **recta** que une $f(x_i)$ con
$f(x_{i+1})$. El área bajo la curva se aproxima con trapecios.
""")
        st.latex(
            r"\int_a^b f(x)\,dx \;\approx\; \frac{h}{2}"
            r"\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]"
        )
        st.markdown(r"""
**Paso a paso:**
1. Calcular $h = (b-a)/n$.
2. Evaluar $f$ en **todos** los nodos $x_0, x_1, \dots, x_n$ (incluyendo
   ambos extremos).
3. Sumar el doble de los nodos internos, más los dos extremos (sin
   duplicar), y multiplicar por $h/2$.

**Restricciones:** ninguna sobre $n$ — funciona con cualquier entero $n \ge 1$.
""")

        st.markdown(r"""
#### 3️⃣ Simpson 1/3 (polinomio de grado 2, regla **cerrada**)

Aproxima $f$ **de a pares de subintervalos** con una **parábola** que pasa
por $f(x_{i-1})$, $f(x_i)$, $f(x_{i+1})$. Al usar una curva en vez de una
recta, captura mejor la curvatura de $f$.
""")
        st.latex(
            r"\int_a^b f(x)\,dx \;\approx\; \frac{h}{3}"
            r"\left[f(x_0)+4f(x_1)+2f(x_2)+4f(x_3)+\cdots+4f(x_{n-1})+f(x_n)\right]"
        )
        st.markdown(r"""
**Paso a paso:**
1. Calcular $h = (b-a)/n$.
2. Evaluar $f$ en todos los nodos $x_0, \dots, x_n$.
3. Asignar coeficientes con el patrón **1 — 4 — 2 — 4 — 2 — ⋯ — 4 — 1**
   (los nodos impares llevan 4, los pares internos llevan 2, los extremos
   llevan 1).
4. Sumar $c_i \cdot f(x_i)$ y multiplicar por $h/3$.

**Restricciones:** requiere que **$n$ sea par**, porque el método agrupa los
subintervalos de a pares (cada parábola necesita 2 subintervalos = 3 nodos).
Con $n$ impar no se puede armar un número entero de parábolas.
""")

        st.markdown(r"""
#### 4️⃣ Simpson 3/8 (polinomio de grado 3, regla **cerrada**)

Aproxima $f$ **de a tríos de subintervalos** con un **polinomio cúbico** que
pasa por 4 nodos consecutivos. Suele ser útil para combinarlo con Simpson 1/3
cuando $n$ no es par, o cuando se busca un poco más de precisión local.
""")
        st.latex(
            r"\int_a^b f(x)\,dx \;\approx\; \frac{3h}{8}"
            r"\left[f(x_0)+3f(x_1)+3f(x_2)+2f(x_3)+\cdots+f(x_n)\right]"
        )
        st.markdown(r"""
**Paso a paso:**
1. Calcular $h = (b-a)/n$.
2. Evaluar $f$ en todos los nodos $x_0, \dots, x_n$.
3. Asignar coeficientes con el patrón **1 — 3 — 3 — 2 — 3 — 3 — 2 — ⋯ — 3 — 3 — 1**
   (cada grupo de 3 subintervalos lleva 3, 3, 2, salvo el primer y el
   último nodo que llevan 1).
4. Sumar $c_i \cdot f(x_i)$ y multiplicar por $\frac{3h}{8}$.

**Restricciones:** requiere que **$n$ sea múltiplo de 3**, porque cada tramo
cúbico necesita 3 subintervalos = 4 nodos consecutivos.
""")

        st.markdown("### Similitudes y diferencias")
        st.markdown(r"""
| | Rectángulo Medio | Trapecios | Simpson 1/3 | Simpson 3/8 |
|---|:---:|:---:|:---:|:---:|
| Grado del polinomio usado | 0 (constante) | 1 (recta) | 2 (parábola) | 3 (cúbica) |
| Tipo de regla | Abierta | Cerrada | Cerrada | Cerrada |
| ¿Usa los extremos $a, b$? | No | Sí | Sí | Sí |
| Orden del error global | $O(h^2)$ | $O(h^2)$ | $O(h^4)$ | $O(h^4)$ |
| Exacto para polinomios de grado ≤ | 1 | 1 | 3 | 3 |
| Restricción sobre $n$ | Ninguna | Ninguna | $n$ par | $n$ múltiplo de 3 |

**Similitudes:** los cuatro dividen $[a,b]$ en subintervalos de igual ancho
$h$, reemplazan $f$ por un polinomio en cada tramo, e integran ese polinomio
de forma exacta. Todos mejoran su precisión a medida que $n$ crece (h más
chico).

**Diferencias clave:** el grado del polinomio interpolante determina tanto la
**velocidad de convergencia** (Simpson converge mucho más rápido que
Trapecios o Rectángulo Medio al aumentar $n$) como las **restricciones**
sobre los valores válidos de $n$.
""")

        st.markdown("### Error de truncamiento y cota de error global")
        st.markdown(r"""
Cada método tiene un **error de truncamiento** asociado, que mide qué tan
lejos está la aproximación del valor exacto debido a reemplazar $f$ por un
polinomio. La fórmula general usa un punto $\xi \in (a, b)$ que existe por el
**Teorema del Valor Medio**, pero como no se conoce $\xi$ exactamente, se
suele acotar el error usando el **máximo** de la derivada correspondiente en
todo el intervalo (esto es lo que hace la sección "Error Máximo Posible" más
abajo, sin necesidad de que el usuario indique $\xi$).
""")
        st.latex(r"E_M \approx \frac{(b-a)\,h^2}{24}\,f''(\xi) \qquad \text{(Rectángulo Medio)}")
        st.latex(r"E_T \approx -\frac{(b-a)\,h^2}{12}\,f''(\xi) \qquad \text{(Trapecios)}")
        st.latex(r"E_{S_{1/3}} \approx -\frac{(b-a)\,h^4}{180}\,f^{(4)}(\xi) \qquad \text{(Simpson 1/3)}")
        st.latex(r"E_{S_{3/8}} \approx -\frac{(b-a)\,h^4}{80}\,f^{(4)}(\xi) \qquad \text{(Simpson 3/8)}")
        st.markdown(r"""
**Cómo se usa la cota de error máximo (sin conocer $\xi$):**
$$
|E| \;\le\; |\text{coeficiente}| \cdot \max_{x \,\in\, [a,b]} \left| f^{(k)}(x) \right|
$$
donde $k=2$ para Rectángulo Medio y Trapecios, y $k=4$ para ambos Simpson.
El máximo de $|f^{(k)}(x)|$ se calcula evaluando la derivada en los extremos
del intervalo y en sus puntos críticos (donde la siguiente derivada se anula),
y tomando el mayor valor absoluto entre todos esos candidatos — esto da el
**peor caso posible** del error, garantizando que el error real nunca lo
supera.

Nótese que Rectángulo Medio y Trapecios comparten el mismo orden de
convergencia $O(h^2)$ porque ambos usan polinomios de grado 1 o menor
(la constante del rectángulo medio es, sorprendentemente, tan precisa como
la recta del trapecio — y de signo opuesto, lo cual se aprovecha en métodos
combinados). Simpson 1/3 y Simpson 3/8 comparten el orden $O(h^4)$ porque
ambos son exactos para polinomios cúbicos, aunque sus constantes de error
son distintas ($1/180$ vs $1/80$).
""")

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
            "El error máximo posible se calcula automáticamente usando el máximo "
            "exacto de la derivada en [a, b], sin necesitar ξ. "
            "Si además querés el error puntual evaluado en un ξ ∈ (a, b) específico "
            "(Teorema del Valor Medio), ingresalo acá."
        ),
    )
    xi = None
    if xi_str.strip():
        try:
            xi = _parse_val(xi_str)
        except Exception:
            st.error("Valor de ξ inválido. Usá números o expresiones como pi/4, 0.5, e.")

    # ── Valor exacto de la integral (opcional, override manual) ──────────────
    exact_str = st.text_input(
        "Valor exacto de la integral — opcional",
        value="",
        placeholder="Ej: 1.45469  |  pi/4  |  2*e - 1",
        help=(
            "Por defecto el script intenta resolver la integral exacta con "
            "SymPy para calcular el Error Real. Si SymPy no logra hallarla "
            "(o tarda demasiado / no converge a una forma cerrada), podés "
            "ingresar acá el valor que ya calculaste por otro medio y se "
            "usará directamente en su lugar, sin que el script intente "
            "resolverla de nuevo."
        ),
    )
    exact_override = None
    if exact_str.strip():
        try:
            exact_override = _parse_val(exact_str)
        except Exception:
            st.error("Valor exacto inválido. Usá números o expresiones como pi/4, 2*e, sqrt(2).")

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
        _show_real_error(
            f.expr, f.x_sym, a, b, result, d, a_str, b_str, exact_override
        )


if __name__ == "__main__":
    run()