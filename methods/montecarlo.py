import streamlit as st
import sympy as sp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_LEVELS = {
    "90%  (z = 1.645)": 1.645,
    "95%  (z = 1.960)": 1.960,
    "99%  (z = 2.576)": 2.576,
}


def _local_dict_1d():
    x = sp.Symbol("x")
    return x, {
        "x":    x,
        "e":    sp.E,
        "E":    sp.E,
        "pi":   sp.pi,
        "ln":   sp.log,
        "log":  sp.log,
        "exp":  sp.exp,
        "sin":  sp.sin,
        "cos":  sp.cos,
        "tan":  sp.tan,
        "sqrt": sp.sqrt,
        "Abs":  sp.Abs,
        "abs":  sp.Abs,
    }


def _local_dict_2d():
    x, y = sp.symbols("x y")
    return x, y, {
        "x":    x,
        "y":    y,
        "e":    sp.E,
        "E":    sp.E,
        "pi":   sp.pi,
        "ln":   sp.log,
        "log":  sp.log,
        "exp":  sp.exp,
        "sin":  sp.sin,
        "cos":  sp.cos,
        "tan":  sp.tan,
        "sqrt": sp.sqrt,
        "Abs":  sp.Abs,
        "abs":  sp.Abs,
    }


def _parse_expr_1d(func_str: str):
    x, ld = _local_dict_1d()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f = sp.lambdify(x, expr, modules=["numpy"])
    return f, sp.latex(expr, ln_notation=True), expr


def _parse_expr_2d(func_str: str):
    x, y, ld = _local_dict_2d()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f = sp.lambdify((x, y), expr, modules=["numpy"])
    return f, sp.latex(expr, ln_notation=True), expr


def _integral_latex_1d(latex_expr: str, a: float, b: float) -> str:
    a_str = sp.latex(sp.nsimplify(a, rational=True), ln_notation=True)
    b_str = sp.latex(sp.nsimplify(b, rational=True), ln_notation=True)
    return rf"\int_{{{a_str}}}^{{{b_str}}} {latex_expr} \, dx"


def _integral_latex_2d(latex_expr: str, ax: float, bx: float,
                       ay: float, by: float) -> str:
    ax_s = sp.latex(sp.nsimplify(ax, rational=True), ln_notation=True)
    bx_s = sp.latex(sp.nsimplify(bx, rational=True), ln_notation=True)
    ay_s = sp.latex(sp.nsimplify(ay, rational=True), ln_notation=True)
    by_s = sp.latex(sp.nsimplify(by, rational=True), ln_notation=True)
    return (
        rf"\int_{{{ay_s}}}^{{{by_s}}} \int_{{{ax_s}}}^{{{bx_s}}} "
        rf"{latex_expr} \, dx \, dy"
    )


def _montecarlo_1d(f, a: float, b: float,
                   n: int, tol: float, z: float, seed: int):
    np.random.seed(seed)
    xs = np.random.uniform(a, b, n)
    ys = f(xs).astype(float)

    V = b - a                        # volumen (longitud) del intervalo
    integral = V * np.mean(ys)
    variance = np.var(ys, ddof=1)
    std      = np.sqrt(variance)
    se       = std / np.sqrt(n)      # error estándar de la media
    margin   = z * se * V            # margen del intervalo de confianza
    ci_lo    = integral - margin
    ci_hi    = integral + margin

    stats = {
        "integral": integral,
        "V":        V,
        "mean":     float(np.mean(ys)),
        "variance": float(variance),
        "std":      float(std),
        "se":       float(se),
        "margin":   float(margin),
        "ci_lo":    float(ci_lo),
        "ci_hi":    float(ci_hi),
        "n":        n,
        "tol":      tol,
        "z":        z,
    }
    return stats


def _montecarlo_2d(f, ax: float, bx: float, ay: float, by: float,
                   n: int, tol: float, z: float, seed: int):
    np.random.seed(seed)
    xs = np.random.uniform(ax, bx, n)
    ys_pts = np.random.uniform(ay, by, n)
    zs = f(xs, ys_pts).astype(float)

    V = (bx - ax) * (by - ay)
    integral = V * np.mean(zs)
    variance = np.var(zs, ddof=1)
    std      = np.sqrt(variance)
    se       = std / np.sqrt(n)
    margin   = z * se * V
    ci_lo    = integral - margin
    ci_hi    = integral + margin

    stats = {
        "integral": integral,
        "V":        V,
        "mean":     float(np.mean(zs)),
        "variance": float(variance),
        "std":      float(std),
        "se":       float(se),
        "margin":   float(margin),
        "ci_lo":    float(ci_lo),
        "ci_hi":    float(ci_hi),
        "n":        n,
        "tol":      tol,
        "z":        z,
    }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Solucion analitica
# ─────────────────────────────────────────────────────────────────────────────

# Funciones especiales no elementales — cuando aparecen, mostramos el
# valor numerico en lugar de la expresion simbolica para no confundir.
_NON_ELEMENTARY = (
    sp.erf, sp.erfc, sp.erfi, sp.erfinv,
    sp.Ei, sp.Si, sp.Ci, sp.li, sp.Li,
    sp.fresnels, sp.fresnelc,
    sp.hyper, sp.meijerg,
    sp.appellf1,
)

def _has_nonelementary(expr) -> bool:
    return expr.has(*_NON_ELEMENTARY)

def _num_latex(expr) -> str:
    """Devuelve el valor numerico de expr como string LaTeX limpio."""
    return sp.latex(expr.evalf(10), ln_notation=True)


def _classify_term(term, var):
    """
    Identifica que regla de integracion elemental corresponde a un termino
    de la forma c * g(var) (donde c puede depender de otras variables, p.ej.
    'y' al integrar en 'x' en una integral doble), y devuelve la antiderivada
    de ese termino en particular junto con metadatos para mostrar el paso a
    paso (nombre de la regla, formula general aplicada).

    Si el termino no calza con ningun patron elemental reconocido, se hace
    un fallback "generico": se integra igual con SymPy pero se etiqueta como
    'tabla de integrales / SymPy' en vez de nombrar una regla puntual.
    """
    try:
        c, g = term.as_independent(var, as_Add=False)  # term == c * g

        # Termino constante respecto de 'var' (incluye el caso de una
        # constante numerica pura, o una expresion que solo depende de
        # otra variable, como 'y' al integrar en 'x').
        if g == 1:
            return {
                "regla":   "Integral de una constante",
                "formula": r"\int k \, d" + sp.latex(var, ln_notation=True) + r" = k \," + sp.latex(var, ln_notation=True),
                "antider": term * var,
            }

        # Regla de la potencia: var**n  (n != -1)
        if g.is_Pow and g.base == var and not g.exp.has(var):
            n = g.exp
            if sp.simplify(n + 1) == 0:
                return {
                    "regla":   "Integral de 1/" + sp.latex(var, ln_notation=True) + " (logaritmo natural)",
                    "formula": r"\int \frac{1}{" + sp.latex(var, ln_notation=True) + r"}\, d" + sp.latex(var, ln_notation=True)
                               + r" = \ln\left|" + sp.latex(var, ln_notation=True) + r"\right|",
                    "antider": c * sp.log(sp.Abs(var)),
                }
            return {
                "regla":   "Regla de la potencia",
                "formula": r"\int " + sp.latex(var, ln_notation=True) + r"^n \, d" + sp.latex(var, ln_notation=True)
                           + r" = \frac{" + sp.latex(var, ln_notation=True) + r"^{n+1}}{n+1} \quad (n \neq -1)",
                "antider": c * var ** (n + 1) / (n + 1),
            }

        # var "a secas" (n = 1, sympy no lo deja como Pow)
        if g == var:
            return {
                "regla":   "Regla de la potencia",
                "formula": r"\int " + sp.latex(var, ln_notation=True) + r"\, d" + sp.latex(var, ln_notation=True)
                           + r" = \frac{" + sp.latex(var, ln_notation=True) + r"^2}{2}",
                "antider": c * var ** 2 / 2,
            }

        # Exponencial: e^(k*var)
        if isinstance(g, sp.exp):
            arg = g.args[0]
            k = sp.diff(arg, var)
            if not k.has(var):
                if k == 1:
                    return {
                        "regla":   "Integral exponencial",
                        "formula": r"\int e^{" + sp.latex(var, ln_notation=True) + r"}\, d" + sp.latex(var, ln_notation=True)
                                   + r" = e^{" + sp.latex(var, ln_notation=True) + r"}",
                        "antider": c * sp.exp(arg),
                    }
                return {
                    "regla":   "Integral exponencial (regla de la cadena)",
                    "formula": r"\int e^{k\," + sp.latex(var, ln_notation=True) + r"}\, d" + sp.latex(var, ln_notation=True)
                               + r" = \frac{e^{k\," + sp.latex(var, ln_notation=True) + r"}}{k}",
                    "antider": c * sp.exp(arg) / k,
                }

        # Exponencial de base distinta de e: a^var
        if g.is_Pow and not g.base.has(var) and g.exp == var:
            base = g.base
            return {
                "regla":   "Integral exponencial de base " + sp.latex(base, ln_notation=True),
                "formula": r"\int a^{" + sp.latex(var, ln_notation=True) + r"}\, d" + sp.latex(var, ln_notation=True)
                           + r" = \frac{a^{" + sp.latex(var, ln_notation=True) + r"}}{\ln a}",
                "antider": c * g / sp.log(base),
            }

        # Seno: sin(k*var)
        if isinstance(g, sp.sin):
            arg = g.args[0]
            k = sp.diff(arg, var)
            if not k.has(var):
                if k == 1:
                    return {
                        "regla":   "Integral de seno",
                        "formula": r"\int \sin(" + sp.latex(var, ln_notation=True) + r")\, d" + sp.latex(var, ln_notation=True)
                                   + r" = -\cos(" + sp.latex(var, ln_notation=True) + r")",
                        "antider": -c * sp.cos(arg),
                    }
                return {
                    "regla":   "Integral de seno (regla de la cadena)",
                    "formula": r"\int \sin(k\," + sp.latex(var, ln_notation=True) + r")\, d" + sp.latex(var, ln_notation=True)
                               + r" = -\frac{\cos(k\," + sp.latex(var, ln_notation=True) + r")}{k}",
                    "antider": -c * sp.cos(arg) / k,
                }

        # Coseno: cos(k*var)
        if isinstance(g, sp.cos):
            arg = g.args[0]
            k = sp.diff(arg, var)
            if not k.has(var):
                if k == 1:
                    return {
                        "regla":   "Integral de coseno",
                        "formula": r"\int \cos(" + sp.latex(var, ln_notation=True) + r")\, d" + sp.latex(var, ln_notation=True)
                                   + r" = \sin(" + sp.latex(var, ln_notation=True) + r")",
                        "antider": c * sp.sin(arg),
                    }
                return {
                    "regla":   "Integral de coseno (regla de la cadena)",
                    "formula": r"\int \cos(k\," + sp.latex(var, ln_notation=True) + r")\, d" + sp.latex(var, ln_notation=True)
                               + r" = \frac{\sin(k\," + sp.latex(var, ln_notation=True) + r")}{k}",
                    "antider": c * sp.sin(arg) / k,
                }

        # Tangente: tan(var)
        if isinstance(g, sp.tan) and g.args[0] == var:
            return {
                "regla":   "Integral de tangente",
                "formula": r"\int \tan(" + sp.latex(var, ln_notation=True) + r")\, d" + sp.latex(var, ln_notation=True)
                           + r" = -\ln\left|\cos(" + sp.latex(var, ln_notation=True) + r")\right|",
                "antider": -c * sp.log(sp.Abs(sp.cos(var))),
            }

    except Exception:
        pass

    # ── Fallback generico: SymPy integra el termino completo y lo
    #    etiquetamos como salido de tabla, sin nombrar una regla puntual.
    F_generic = sp.integrate(term, var)
    return {
        "regla":   "Integral de tabla (resuelta con SymPy)",
        "formula": None,
        "antider": F_generic,
    }


def _term_steps(expr, var):
    """
    Descompone expr en terminos (suma) y devuelve la lista de pasos de
    integracion termino a termino, usando _classify_term. Tambien indica
    si hubo mas de un termino (para mostrar o no el paso de linealidad).
    """
    terms = sp.Add.make_args(sp.expand(expr))
    steps = []
    for term in terms:
        info = _classify_term(term, var)
        steps.append({
            "term_latex":    sp.latex(term, ln_notation=True),
            "regla":         info["regla"],
            "formula":       info["formula"],
            "antider_latex": sp.latex(info["antider"], ln_notation=True),
        })
    return steps, len(terms) > 1


def _solve_analytical_1d(func_str: str, a: float, b: float):
    """
    Resuelve la integral definida en forma cerrada.

    IMPORTANTE: se usa sp.integrate(expr, (x, a, b)) -- integral DEFINIDA
    directa -- como fuente de verdad para el resultado, en vez de calcular
    la primitiva indefinida F(x) y restar F(b) - F(a) "a mano". Ese segundo
    enfoque (Barrow manual) puede dar resultados silenciosamente incorrectos
    cuando f(x) tiene una asintota o discontinuidad dentro de [a, b]: por
    ejemplo f(x) = 1/x**2 en [-1, 1] da F(b)-F(a) = -2 (un valor finito y
    encima negativo, aunque el integrando es siempre positivo y la integral
    en realidad diverge). sp.integrate con limites definidos si detecta
    estos casos. La primitiva F(x) se sigue calculando, pero solo para
    mostrar el desarrollo pedagogico paso a paso, y unicamente cuando
    coincide con el resultado confiable de la integral definida.
    """
    x, ld  = _local_dict_1d()
    expr   = sp.sympify(func_str.replace("^", "**"), locals=ld)
    a_sym  = sp.nsimplify(a, rational=True)
    b_sym  = sp.nsimplify(b, rational=True)

    # 1) Integral definida directa: esta es la fuente de verdad.
    try:
        result_sym = sp.integrate(expr, (x, a_sym, b_sym))
    except Exception as e:
        return {"error": str(e)}

    if result_sym.has(sp.Integral):
        return {"error": "SymPy no encontro una solucion en forma cerrada para esta integral."}

    result_sym = sp.simplify(result_sym)

    # 2) Validar que el resultado sea un numero real y finito. Descarta
    #    oo / -oo / zoo / nan (integral divergente) y resultados complejos
    #    (sintoma tipico de una asintota dentro de [a, b], como tan(x)
    #    cruzando pi/2).
    if result_sym.has(sp.oo, -sp.oo, sp.zoo, sp.nan) or result_sym.is_extended_real is False:
        return {
            "error": (
                f"La integral no converge a un valor real finito en el intervalo "
                f"[{a}, {b}] (SymPy obtuvo: {sp.latex(result_sym, ln_notation=True)}). Es probable que "
                "f(x) tenga una asintota, polo o discontinuidad dentro del intervalo "
                "de integracion."
            )
        }

    try:
        result_num = float(result_sym.evalf())
    except (TypeError, ValueError):
        return {"error": "No se pudo evaluar numericamente el resultado obtenido por SymPy."}

    ne_res = _has_nonelementary(result_sym)

    # 3) Primitiva indefinida F(x), SOLO para mostrar el desarrollo paso a
    #    paso (Teorema Fundamental del Calculo + Regla de Barrow). Se
    #    calcula aparte y se usa unicamente si coincide con el resultado
    #    confiable de arriba -- si no coincide (senal de que F no es
    #    continua en todo [a, b]), se omiten esos pasos pedagogicos y se
    #    muestra directamente el resultado de la integral definida.
    F = F_b = F_a = None
    show_steps = False
    try:
        F_candidate = sp.integrate(expr, x)
        if not F_candidate.has(sp.Integral):
            F_b_candidate = F_candidate.subs(x, b_sym)
            F_a_candidate = F_candidate.subs(x, a_sym)
            barrow_value = sp.simplify(F_b_candidate - F_a_candidate)
            if (barrow_value.is_extended_real
                    and sp.simplify(barrow_value - result_sym) == 0):
                F, F_b, F_a = F_candidate, F_b_candidate, F_a_candidate
                show_steps = True
    except Exception:
        pass

    ne_F = _has_nonelementary(F) if F is not None else False

    # 4) Desglose termino a termino de la integral (linealidad + regla
    #    aplicada a cada termino), solo para fines pedagogicos. Y los
    #    pasos de sustitucion explicita de la Regla de Barrow.
    term_steps = lineal = None
    F_b_raw_latex = F_a_raw_latex = None
    if show_steps:
        try:
            term_steps, lineal = _term_steps(expr, x)
        except Exception:
            term_steps, lineal = None, None
        # Version "cruda" de F(b) y F(a): la sustitucion antes de
        # simplificar/evaluar, para mostrar el paso intermedio.
        F_b_raw = F.subs(x, b_sym, simultaneous=True)
        F_a_raw = F.subs(x, a_sym, simultaneous=True)
        F_b_raw_latex = _num_latex(F_b_raw) if _has_nonelementary(F_b_raw) else sp.latex(F_b_raw, mul_symbol="dot", ln_notation=True)
        F_a_raw_latex = _num_latex(F_a_raw) if _has_nonelementary(F_a_raw) else sp.latex(F_a_raw, mul_symbol="dot", ln_notation=True)

    return {
        "expr":       sp.latex(expr, ln_notation=True),
        "show_steps": show_steps,
        "term_steps": term_steps,
        "lineal":     lineal,
        "F":          (_num_latex(F)   if ne_F else sp.latex(F, ln_notation=True))   if show_steps else None,
        "F_at_b":     (_num_latex(F_b) if ne_F else sp.latex(F_b, ln_notation=True)) if show_steps else None,
        "F_at_a":     (_num_latex(F_a) if ne_F else sp.latex(F_a, ln_notation=True)) if show_steps else None,
        "F_b_raw":    F_b_raw_latex,
        "F_a_raw":    F_a_raw_latex,
        "result_sym": _num_latex(result_sym) if ne_res else sp.latex(result_sym, ln_notation=True),
        "result_num": result_num,
        "a_sym":      sp.latex(a_sym, ln_notation=True),
        "b_sym":      sp.latex(b_sym, ln_notation=True),
        "error":      None,
    }


def _solve_analytical_2d(func_str: str,
                         ax: float, bx: float,
                         ay: float, by: float):
    x, y, ld = _local_dict_2d()
    expr  = sp.sympify(func_str.replace("^", "**"), locals=ld)
    ax_s  = sp.nsimplify(ax, rational=True)
    bx_s  = sp.nsimplify(bx, rational=True)
    ay_s  = sp.nsimplify(ay, rational=True)
    by_s  = sp.nsimplify(by, rational=True)

    try:
        Fx = sp.integrate(expr, (x, ax_s, bx_s))
    except Exception as e:
        return {"error": str(e)}
    if Fx.has(sp.Integral):
        return {"error": "SymPy no pudo resolver la integral interior en x."}

    Fx_simplified = sp.simplify(Fx)
    ne_x          = _has_nonelementary(Fx_simplified)

    try:
        result_sym = sp.integrate(Fx_simplified, (y, ay_s, by_s))
    except Exception as e:
        return {"error": str(e)}
    if result_sym.has(sp.Integral):
        return {"error": "SymPy no pudo resolver la integral exterior en y."}

    result_sym = sp.simplify(result_sym)

    if result_sym.has(sp.oo, -sp.oo, sp.zoo, sp.nan) or result_sym.is_extended_real is False:
        return {
            "error": (
                f"La integral no converge a un valor real finito en el dominio "
                f"dado (SymPy obtuvo: {sp.latex(result_sym, ln_notation=True)}). Es probable que "
                "f(x, y) tenga una asintota o discontinuidad dentro del dominio."
            )
        }

    try:
        result_num = float(result_sym.evalf())
    except (TypeError, ValueError):
        return {"error": "No se pudo evaluar numericamente el resultado obtenido por SymPy."}

    ne_res     = _has_nonelementary(result_sym)

    # Desglose pedagogico termino a termino de cada integral (interior en x,
    # exterior en y).
    try:
        x_term_steps, x_lineal = _term_steps(expr, x)
    except Exception:
        x_term_steps, x_lineal = None, None
    try:
        y_term_steps, y_lineal = _term_steps(Fx_simplified, y)
    except Exception:
        y_term_steps, y_lineal = None, None

    return {
        "expr":         sp.latex(expr, ln_notation=True),
        "Fx":           _num_latex(Fx_simplified) if ne_x  else sp.latex(Fx_simplified, ln_notation=True),
        "x_term_steps": x_term_steps,
        "x_lineal":     x_lineal,
        "y_term_steps": y_term_steps,
        "y_lineal":     y_lineal,
        "result_sym":   _num_latex(result_sym)    if ne_res else sp.latex(result_sym, ln_notation=True),
        "result_num":   result_num,
        "ax_s":         sp.latex(ax_s, ln_notation=True),
        "bx_s":         sp.latex(bx_s, ln_notation=True),
        "ay_s":         sp.latex(ay_s, ln_notation=True),
        "by_s":         sp.latex(by_s, ln_notation=True),
        "error":        None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pagina
# ─────────────────────────────────────────────────────────────────────────────

def run():
    st.markdown("""
    <style>
    html { font-size: 20px; }

    div.stButton { width: 100%; }
    div.stButton > button {
        background-color: #dc2626;
        color: #fff;
        border: none;
        border-radius: 6px;
        width: 100%;
        padding: .7rem 0;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background .15s;
    }
    div.stButton > button:hover  { background-color: #b91c1c; color: #fff; }
    div.stButton > button:active { background-color: #991b1b; color: #fff; }

    table { margin: 0.5rem auto 0 auto; }
    table th, table td { text-align: center !important; }

    .result-cards { display: flex; gap: 1.2rem; margin: 0.6rem 0 1.6rem 0; }
    .result-card {
        flex: 1;
        border-radius: 10px;
        padding: 1.3rem 1rem 1rem 1rem;
        text-align: center;
        border: 1px solid rgba(128,128,128,.18);
    }
    .result-card .rc-label {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        opacity: .5;
        margin-bottom: .55rem;
    }
    .result-card .rc-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        line-height: 1;
    }
    .result-card .rc-sub {
        font-size: 1rem;
        opacity: .35;
        margin-top: .35rem;
        font-style: italic;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .9rem;
        margin: 1rem 0 1.6rem 0;
    }
    .stat-item {
        border-radius: 8px;
        padding: .9rem .8rem .7rem .8rem;
        border: 1px solid rgba(128,128,128,.18);
        text-align: center;
    }
    .stat-item .si-label {
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .45;
        margin-bottom: .4rem;
    }
    .stat-item .si-value {
        font-size: 1.15rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }

    .formula-block {
        background: rgba(128,128,128,.07);
        border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: .8rem 1.1rem;
        margin: .6rem 0 1rem 0;
    }

    .tol-ok  { color: #16a34a; font-weight: 700; }
    .tol-fail { color: #dc2626; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("📘 Teoría: ¿Qué es el Método de Monte Carlo?", expanded=False):
                st.markdown(r"""
        ### ¿Qué es?

        El **Método de Monte Carlo** es una técnica numérica estocástica (basada en el uso de números aleatorios) que se utiliza para aproximar expresiones matemáticas complejas. En el cálculo, lo empleamos principalmente para **aproximar el valor de integrales definidas**.

        A diferencia de métodos numéricos tradicionales (como la regla del trapecio o la bisección que vimos para raíces), que son deterministas y usan grillas espaciadas de forma regular, Monte Carlo recurre al muestreo probabilístico.

        Una de sus mayores ventajas es que **puede usarse tanto para derivadas e integrales simples (cálculo de áreas en 1D) como para integrales dobles o múltiples (volúmenes en 2D, 3D y más)**, siendo de hecho el método más eficiente cuando la cantidad de dimensiones aumenta drásticamente.
        """)

                st.markdown("### ¿Cómo funciona el algoritmo?")
                st.markdown(r"""
        El principio subyacente es la Ley de los Grandes Números. Básicamente, calculamos el "valor promedio" de la función dentro del dominio y lo multiplicamos por el tamaño total de ese dominio.

        1. **Definir el dominio:** Determinar el área o volumen donde vamos a integrar. Por ejemplo, el intervalo $[a, b]$ para una integral simple, o un rectángulo de integración para una integral doble.
        2. **Generar $N$ puntos aleatorios:** Se generan coordenadas al azar con distribución uniforme dentro del dominio definido.
        3. **Evaluar la función:** Se calcula $f(x)$ (o $f(x, y)$ en integrales dobles) para cada uno de los $N$ puntos generados.
        4. **Calcular el promedio muestral:** Se suman todas las evaluaciones y se dividen por $N$.
        5. **Multiplicar por el tamaño del dominio:** Se multiplica el promedio por la longitud (en 1D), área (en 2D) o volumen del dominio.
        """)

                st.markdown("### Análisis Estadístico y Fórmulas")
                st.markdown(r"""
        Si llamamos $V$ al "volumen" del dominio de integración (donde $V = b - a$ en una integral simple), la integral $I$ se aproxima con la siguiente fórmula:
        """)
                st.latex(r"I \approx V \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)")

                st.markdown(r"""
        Como dependemos de números pseudoaleatorios, cada ejecución de la simulación arrojará un resultado ligeramente diferente. Para analizar la fiabilidad de nuestra aproximación, aplicamos estadística sobre la muestra de puntos evaluados:

        **1. Desvío Estándar Muestral ($S$):** Nos indica cuánta dispersión hay entre los valores de la función evaluada y su promedio ($\bar{f}$).
        """)
                st.latex(r"S = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (f(x_i) - \bar{f})^2}")

                st.markdown(r"""
        **2. Error Estándar ($SE$):**
        Es la métrica crucial de Monte Carlo. Cuantifica la incertidumbre de nuestra aproximación final respecto al valor real teórico de la integral.
        """)
                st.latex(r"SE = V \cdot \frac{S}{\sqrt{N}}")

                st.markdown(r"""
        > **💡 Sobre el comportamiento del Error:**
        > Al observar la fórmula del Error Estándar, notamos que el término $\sqrt{N}$ está en el denominador. La convergencia de Monte Carlo es del orden de $\mathcal{O}(1/\sqrt{N})$. Esto significa que **para reducir el error a la mitad, no basta con duplicar los puntos: es necesario cuadruplicar el tamaño de la muestra ($4N$)**.
        """)
                # ── NUEVA SECCIÓN DE CURIOSIDAD ──────────────────────────────────────────
                st.markdown("---")
                st.markdown(r'### 🎲 Curiosidad: El "Hola Mundo" de Monte Carlo (Estimando $\pi$)')
                st.markdown(r"""
        Es posible utilizar esta misma lógica probabilística para estimar el valor de $\pi$. Imagina un tablero cuadrado perfecto de lado $2r$, y dentro de él, dibujamos un círculo inscrito de radio $r$. 

        Si dividimos el área del círculo por el área del cuadrado, la geometría nos da una proporción exacta:
        """)
                st.latex(r"\frac{A_{\text{círculo}}}{A_{\text{cuadrado}}} = \frac{\pi r^2}{(2r)^2} = \frac{\pi r^2}{4r^2} = \frac{\pi}{4}")

                st.markdown(r"""
        Ahora entra **Monte Carlo**: En lugar de medir áreas, disparamos $N$ puntos aleatorios ("dardos") de forma uniforme hacia el cuadrado. La proporción de dardos que caigan *dentro* del círculo ($N_{\text{adentro}}$) tenderá a igualar esa misma relación matemática. Si despejamos $\pi$, obtenemos nuestra fórmula de estimación:
        """)
                st.latex(r"\pi \approx 4 \cdot \frac{N_{\text{adentro}}}{N_{\text{total}}}")

                st.markdown(r"""
        **¿Por qué no usamos esto en la práctica para calcular $\pi$?**
        Aquí es donde la teoría choca con la realidad del muestreo. Como vimos en el comportamiento del Error Estándar, la convergencia es extremadamente lenta ($\mathcal{O}(1/\sqrt{N})$). Para ganar apenas un decimal extra de precisión en nuestro número $\pi$, tendríamos que multiplicar la cantidad de puntos generados por 100. Llegar a una alta precisión requiere miles de millones de iteraciones, haciéndolo un método computacionalmente ineficiente para este fin, ¡pero fascinante como herramienta pedagógica!
        """)   
                    
    st.title("Integración por Monte Carlo")

    # ── Dimensiones ──────────────────────────────────────────────────────────
    dims = st.selectbox(
        "Cantidad de dimensiones",
        options=["1  —  ∫ f(x) dx", "2  —  ∬ f(x, y) dx dy"],
        index=0,
    )
    dim = 1 if dims.startswith("1") else 2

    # ── Función ──────────────────────────────────────────────────────────────
    if dim == 1:
        func_str = st.text_input(
            "f(x)",
            value="x**2",
            placeholder="Ej: x**2,  sin(x),  exp(-x**2),  x**3 - 2*x + 1",
        )
    else:
        func_str = st.text_input(
            "f(x, y)",
            value="x**2 + y**2",
            placeholder="Ej: x**2 + y**2,  sin(x)*cos(y),  x*y",
        )

    f = None
    latex_expr = ""
    if func_str:
        try:
            if dim == 1:
                f, latex_expr, _ = _parse_expr_1d(func_str)
            else:
                f, latex_expr, _ = _parse_expr_2d(func_str)
            st.latex(rf"f = {latex_expr}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Límites de integración ───────────────────────────────────────────────
    st.markdown("**Límites de integración**")

    a = b = ax = bx = ay = by = None

    if dim == 1:
        c1, c2 = st.columns(2)
        a_str = c1.text_input("Límite inferior  (a)", value="0")
        b_str = c2.text_input("Límite superior  (b)", value="1")
        try:
            a = float(a_str)
        except ValueError:
            c1.error("Valor inválido para a")
        try:
            b = float(b_str)
        except ValueError:
            c2.error("Valor inválido para b")
        if a is not None and b is not None and a >= b:
            st.error("El límite inferior debe ser menor que el superior.")
            a = b = None
    else:
        c1, c2 = st.columns(2)
        c1.markdown("**Intervalo en x**")
        c2.markdown("**Intervalo en y**")
        ax_str = c1.text_input("a  (x inferior)", value="0")
        bx_str = c1.text_input("b  (x superior)", value="1")
        ay_str = c2.text_input("c  (y inferior)", value="0")
        by_str = c2.text_input("d  (y superior)", value="1")
        try:
            ax = float(ax_str)
        except ValueError:
            c1.error("Valor inválido para a")
        try:
            bx = float(bx_str)
        except ValueError:
            c1.error("Valor inválido para b")
        try:
            ay = float(ay_str)
        except ValueError:
            c2.error("Valor inválido para c")
        try:
            by = float(by_str)
        except ValueError:
            c2.error("Valor inválido para d")
        if ax is not None and bx is not None and ax >= bx:
            st.error("El límite inferior en x debe ser menor que el superior.")
            ax = bx = None
        if ay is not None and by is not None and ay >= by:
            st.error("El límite inferior en y debe ser menor que el superior.")
            ay = by = None

    # ── Render integral simbólica ────────────────────────────────────────────
    limits_ok = (
        (dim == 1 and a is not None and b is not None) or
        (dim == 2 and ax is not None and bx is not None
         and ay is not None and by is not None)
    )
    if f is not None and limits_ok and latex_expr:
        if dim == 1:
            st.latex(_integral_latex_1d(latex_expr, a, b))
        else:
            st.latex(_integral_latex_2d(latex_expr, ax, bx, ay, by))

    # ── Parámetros ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    n_str   = c1.text_input("Máx. puntos (N)", value="1000")
    tol_str = c2.text_input("Tolerancia (ε)", value="0.01")
    conf_label = c3.selectbox(
        "Nivel de confianza",
        options=list(CONFIDENCE_LEVELS.keys()),
        index=1,
    )
    seed_str = c4.text_input("Semilla (seed)", value="42")

    n = tol = z = seed = None
    try:
        n = int(n_str)
        if n <= 0:
            raise ValueError
    except ValueError:
        c1.error("Valor inválido")
    try:
        tol = float(tol_str)
        if tol <= 0:
            raise ValueError
    except ValueError:
        c2.error("Valor inválido")
    z = CONFIDENCE_LEVELS[conf_label]
    try:
        seed = int(seed_str)
    except ValueError:
        c4.error("Semilla inválida")

    # ── Botón ─────────────────────────────────────────────────────────────────
    if st.button("Calcular integral", use_container_width=True):
        # Validaciones
        if f is None:
            st.error("Ingresá una función válida antes de calcular.")
            return
        if not limits_ok:
            st.error("Ingresá límites de integración válidos.")
            return
        if n is None or tol is None or seed is None:
            st.error("Verificá los parámetros ingresados.")
            return

        # ── Cálculo ───────────────────────────────────────────────────────────
        try:
            if dim == 1:
                stats = _montecarlo_1d(f, a, b, n, tol, z, seed)
            else:
                stats = _montecarlo_2d(f, ax, bx, ay, by, n, tol, z, seed)
        except Exception as e:
            st.error(f"Error durante el cálculo: {e}")
            return

        integral = stats["integral"]
        se_ok    = stats["se"] * stats["V"] <= tol

        # ── Resultado principal ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Resultado")

        st.markdown(f"""
        <div class="result-cards">
            <div class="result-card" style="border-top: 3px solid #2563eb;">
                <div class="rc-label">Integral aproximada</div>
                <div class="rc-value">{integral:.8f}</div>
                <div class="rc-sub">I ≈</div>
            </div>
            <div class="result-card" style="border-top: 3px solid #16a34a;">
                <div class="rc-label">Puntos generados</div>
                <div class="rc-value">{n}</div>
                <div class="rc-sub">N</div>
            </div>
            <div class="result-card" style="border-top: 3px solid #9333ea;">
                <div class="rc-label">Intervalo de confianza</div>
                <div class="rc-value" style="font-size:1.2rem;">
                    [{stats['ci_lo']:.6f},<br>{stats['ci_hi']:.6f}]
                </div>
                <div class="rc-sub">{conf_label.split()[0]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Cómo se calculó ───────────────────────────────────────────────────
        st.subheader("¿Cómo se calculó?")

        if dim == 1:
            st.markdown(
                "Se generaron **N puntos aleatorios** "
                r"$x_i \sim \mathcal{U}(a,\, b)$ "
                "y se evaluó $f(x_i)$ en cada uno. "
                "La estimación surge de:"
            )
            st.latex(
                r"I \approx (b - a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)"
            )
            st.markdown(
                f"Con $a={a}$, $b={b}$, el **volumen** (longitud) del "
                f"intervalo es $V = b - a = {stats['V']:.6g}$."
            )
        else:
            st.markdown(
                "Se generaron **N pares aleatorios** "
                r"$(x_i, y_i) \sim \mathcal{U}([a,b]\times[c,d])$ "
                "y se evaluó $f(x_i, y_i)$ en cada uno. "
                "La estimación surge de:"
            )
            st.latex(
                r"I \approx (b-a)(d-c) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i, y_i)"
            )
            st.markdown(
                f"El **volumen** del dominio de integración es "
                f"$V = (b-a)(d-c) = {stats['V']:.6g}$."
            )

        st.markdown("La estimación puede reescribirse como:")
        st.latex(r"I \approx V \cdot \bar{f}_N")
        st.markdown(
            rf"donde $\bar{{f}}_N = {stats['mean']:.8f}$ "
            "es la **media muestral** de los valores de la función."
        )

        # ── Análisis estadístico ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Análisis estadístico")

        # Varianza y desvío
        st.markdown("**Varianza muestral y desvío estándar**")
        st.latex(
            r"S^2 = \frac{1}{N-1} \sum_{i=1}^{N}(f_i - \bar{f}_N)^2"
            rf"\quad \Rightarrow \quad S^2 = {stats['variance']:.8f}"
        )
        st.latex(
            r"S = \sqrt{S^2}"
            rf"\quad \Rightarrow \quad S = {stats['std']:.8f}"
        )

        # Error estándar
        st.markdown("**Error estándar de la integral**")
        st.markdown(
            "Como la estimación es $I \\approx V \\cdot \\bar{f}_N$ y $V$ es una "
            "constante, el error estándar de $\\bar{f}_N$ se escala por $V$:"
        )
        st.latex(
            r"\text{SE}_I = V \cdot \frac{S}{\sqrt{N}}"
            rf"\quad \Rightarrow \quad \text{{SE}}_I = {stats['V']:.6g}"
            rf"\cdot \frac{{{stats['std']:.6f}}}{{\sqrt{{{n}}}}}"
            rf"= {stats['se'] * stats['V']:.8f}"
        )

        tol_class = "tol-ok" if se_ok else "tol-fail"
        tol_sym   = r"\leq" if se_ok else r">"
        tol_msg   = "✔ Tolerancia cumplida" if se_ok else "✘ Tolerancia no cumplida"
        st.markdown(
            rf"$$\text{{SE}}_I = {stats['se']*stats['V']:.8f}"
            rf"\; {tol_sym} \; \varepsilon = {tol}$$"
        )
        st.markdown(f'<span class="{tol_class}">{tol_msg}</span>', unsafe_allow_html=True)

        # Intervalo de confianza
        st.markdown("**Intervalo de confianza**")
        z_val = stats["z"]
        st.latex(
            rf"I \pm z_{{\alpha/2}} \cdot \text{{SE}}_I"
            rf"= {integral:.6f} \pm {z_val} \cdot {stats['se']*stats['V']:.6f}"
        )
        st.latex(
            rf"IC = \left[{stats['ci_lo']:.8f},\; {stats['ci_hi']:.8f}\right]"
        )

        # Grilla de estadísticos
        se_i = stats["se"] * stats["V"]
        st.markdown(
            f"""
<div class="stat-grid">
    <div class="stat-item">
        <div class="si-label">Volumen V</div>
        <div class="si-value">{stats['V']:.6g}</div>
    </div>
    <div class="stat-item">
        <div class="si-label">Media muestral f̄</div>
        <div class="si-value">{stats['mean']:.8f}</div>
    </div>
    <div class="stat-item">
        <div class="si-label">Varianza S²</div>
        <div class="si-value">{stats['variance']:.8f}</div>
    </div>
    <div class="stat-item">
        <div class="si-label">Desvío estándar S</div>
        <div class="si-value">{stats['std']:.8f}</div>
    </div>
    <div class="stat-item">
        <div class="si-label">Error estándar SE_I</div>
        <div class="si-value">{se_i:.8f}</div>
    </div>
    <div class="stat-item">
        <div class="si-label">Margen IC (±)</div>
        <div class="si-value">{stats['margin']:.8f}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Solución analítica ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Solución analítica")

        if dim == 1:
            sol = _solve_analytical_1d(func_str, a, b)
        else:
            sol = _solve_analytical_2d(func_str, ax, bx, ay, by)

        if sol["error"]:
            st.warning(
                f"No se pudo obtener una solución analítica: {sol['error']}"
            )
        else:
            if dim == 1:
                if sol["show_steps"]:
                    st.markdown(
                        "Vamos a resolver la integral en dos grandes etapas: "
                        "primero encontramos la **primitiva** $F(x)$ (Teorema "
                        "Fundamental del Cálculo) y después aplicamos la "
                        "**Regla de Barrow** para evaluarla en los límites."
                    )

                    # ── Paso 1: planteo + linealidad ────────────────────────
                    st.markdown("**Paso 1 — Planteamos la integral indefinida**")
                    st.markdown(
                        r"Buscamos $F(x)$ tal que $F'(x) = f(x)$, es decir:"
                    )
                    st.latex(rf"\int {sol['expr']} \, dx")

                    term_steps = sol["term_steps"] or []
                    step_n = 2

                    if sol["lineal"] and len(term_steps) > 1:
                        st.markdown(f"**Paso {step_n} — Propiedad de linealidad**")
                        st.markdown(
                            "Como el integrando es una suma/resta de términos, "
                            "separamos la integral en una integral por cada uno "
                            "(la integral de una suma es la suma de las integrales):"
                        )
                        terms_sum_latex = " + ".join(
                            rf"\int {t['term_latex']} \, dx" for t in term_steps
                        )
                        # Convertimos "+ -" en "- " para que se vea prolijo
                        terms_sum_latex = terms_sum_latex.replace("+ -", "- ")
                        st.latex(rf"\int {sol['expr']} \, dx = {terms_sum_latex}")
                        step_n += 1

                    # ── Paso(s): regla aplicada a cada término ──────────────
                    if len(term_steps) > 1:
                        st.markdown(f"**Paso {step_n} — Integramos cada término por separado**")
                    else:
                        st.markdown(f"**Paso {step_n} — Aplicamos la regla de integración correspondiente**")

                    for i, t in enumerate(term_steps, start=1):
                        label = f"{step_n}.{i}" if len(term_steps) > 1 else str(step_n)
                        st.markdown(f"*Paso {label}: {t['regla']}*")
                        if t["formula"]:
                            st.latex(t["formula"])
                        if len(term_steps) > 1:
                            st.latex(
                                rf"\int {t['term_latex']} \, dx = {t['antider_latex']}"
                            )
                        else:
                            st.latex(
                                rf"\int {sol['expr']} \, dx = {t['antider_latex']} + C"
                            )
                    step_n += 1

                    # ── Paso: ensamblar F(x) ────────────────────────────────
                    if len(term_steps) > 1:
                        st.markdown(f"**Paso {step_n} — Sumamos los resultados para obtener** $F(x)$")
                        st.markdown(
                            "Juntamos la antiderivada de cada término (la constante "
                            "de integración $C$ se agrega una sola vez al final):"
                        )
                        st.latex(rf"F(x) = {sol['F']} + C")
                        step_n += 1

                    # ── Paso: Regla de Barrow, sustitución explícita ────────
                    st.markdown(f"**Paso {step_n} — Regla de Barrow: evaluamos** $F$ **en los límites**")
                    st.markdown(
                        r"El Teorema Fundamental del Cálculo dice que "
                        r"$\int_a^b f(x)\,dx = F(b) - F(a)$. Evaluamos cada límite por separado."
                    )
                    st.markdown(f"*Sustituimos $x = {sol['b_sym']}$ en $F(x)$:*")
                    st.latex(rf"F({sol['b_sym']}) = {sol['F_b_raw']}")
                    st.markdown(f"*Sustituimos $x = {sol['a_sym']}$ en $F(x)$:*")
                    st.latex(rf"F({sol['a_sym']}) = {sol['F_a_raw']}")
                    step_n += 1

                    st.markdown(f"**Paso {step_n} — Restamos**")
                    st.latex(
                        rf"\int_{{{sol['a_sym']}}}^{{{sol['b_sym']}}} {sol['expr']} \, dx"
                        rf"= F({sol['b_sym']}) - F({sol['a_sym']})"
                        rf"= \left({sol['F_at_b']}\right) - \left({sol['F_at_a']}\right)"
                    )
                    step_n += 1

                    # ── Paso: resultado exacto ──────────────────────────────
                    st.markdown(f"**Paso {step_n} — Simplificamos y obtenemos el resultado exacto**")
                    st.latex(
                        rf"I_{{exacta}} = {sol['result_sym']} = {sol['result_num']:.10f}"
                    )
                else:
                    # f(x) tiene una primitiva discontinua en [a, b]: no mostramos
                    # los pasos de Barrow porque serian enganosos. Vamos directo
                    # al resultado, obtenido por integracion definida directa.
                    st.info(
                        "La primitiva de $f(x)$ no es continua en todo el intervalo "
                        "$[a, b]$ (suele pasar cuando hay una asíntota o "
                        "discontinuidad dentro del intervalo), así que la regla de "
                        "Barrow simple no es aplicable. SymPy resolvió la integral "
                        "definida directamente:"
                    )
                    st.latex(
                        rf"\int_{{{sol['a_sym']}}}^{{{sol['b_sym']}}} {sol['expr']} \, dx"
                        rf"= {sol['result_sym']}"
                    )
                    st.markdown("**Resultado exacto**")
                    st.latex(
                        rf"I_{{exacta}} = {sol['result_sym']} = {sol['result_num']:.10f}"
                    )

            else:
                st.markdown(
                    "Una integral doble se resuelve de **adentro hacia afuera**: "
                    "primero integramos en $x$ (tratando $y$ como una constante) "
                    "y, una vez obtenido ese resultado en función de $y$, lo "
                    "integramos en $y$."
                )

                # ── Integral interior en x ──────────────────────────────────
                st.markdown("**Paso 1 — Planteamos la integral interior en** $x$")
                st.markdown(
                    r"Tratamos $y$ como constante e integramos $f(x,y)$ respecto de $x$, "
                    r"entre los límites del eje $x$:"
                )
                st.latex(
                    rf"\int_{{{sol['ax_s']}}}^{{{sol['bx_s']}}} {sol['expr']} \, dx"
                )

                x_steps = sol["x_term_steps"] or []
                step_n = 2

                if sol["x_lineal"] and len(x_steps) > 1:
                    st.markdown(f"**Paso {step_n} — Linealidad: separamos por términos**")
                    terms_sum_latex = " + ".join(
                        rf"\int_{{{sol['ax_s']}}}^{{{sol['bx_s']}}} {t['term_latex']} \, dx"
                        for t in x_steps
                    )
                    terms_sum_latex = terms_sum_latex.replace("+ -", "- ")
                    st.latex(
                        rf"\int_{{{sol['ax_s']}}}^{{{sol['bx_s']}}} {sol['expr']} \, dx = {terms_sum_latex}"
                    )
                    step_n += 1

                if len(x_steps) > 1:
                    st.markdown(f"**Paso {step_n} — Integramos cada término en** $x$")
                else:
                    st.markdown(f"**Paso {step_n} — Aplicamos la regla de integración en** $x$")
                for i, t in enumerate(x_steps, start=1):
                    label = f"{step_n}.{i}" if len(x_steps) > 1 else str(step_n)
                    st.markdown(f"*Paso {label}: {t['regla']}*")
                    if t["formula"]:
                        st.latex(t["formula"])
                    st.latex(rf"\int {t['term_latex']} \, dx = {t['antider_latex']}")
                step_n += 1

                st.markdown(f"**Paso {step_n} — Evaluamos en los límites de** $x$ **(Barrow)**")
                st.markdown(
                    "Sumamos las antiderivadas de cada término, evaluamos en "
                    rf"$x={sol['bx_s']}$ y en $x={sol['ax_s']}$, y restamos. "
                    "El resultado es una función que ahora depende solo de $y$:"
                )
                st.latex(
                    rf"\int_{{{sol['ax_s']}}}^{{{sol['bx_s']}}} {sol['expr']} \, dx"
                    rf"= {sol['Fx']}"
                )
                step_n += 1

                # ── Integral exterior en y ──────────────────────────────────
                st.markdown(f"**Paso {step_n} — Planteamos la integral exterior en** $y$")
                st.markdown(
                    "Tomamos el resultado anterior (ya sin $x$) y lo integramos "
                    "respecto de $y$, entre los límites del eje $y$:"
                )
                st.latex(
                    rf"\int_{{{sol['ay_s']}}}^{{{sol['by_s']}}} \left({sol['Fx']}\right) dy"
                )
                step_n += 1

                y_steps = sol["y_term_steps"] or []

                if sol["y_lineal"] and len(y_steps) > 1:
                    st.markdown(f"**Paso {step_n} — Linealidad: separamos por términos**")
                    terms_sum_latex = " + ".join(
                        rf"\int_{{{sol['ay_s']}}}^{{{sol['by_s']}}} {t['term_latex']} \, dy"
                        for t in y_steps
                    )
                    terms_sum_latex = terms_sum_latex.replace("+ -", "- ")
                    st.latex(
                        rf"\int_{{{sol['ay_s']}}}^{{{sol['by_s']}}} \left({sol['Fx']}\right) dy = {terms_sum_latex}"
                    )
                    step_n += 1

                if len(y_steps) > 1:
                    st.markdown(f"**Paso {step_n} — Integramos cada término en** $y$")
                else:
                    st.markdown(f"**Paso {step_n} — Aplicamos la regla de integración en** $y$")
                for i, t in enumerate(y_steps, start=1):
                    label = f"{step_n}.{i}" if len(y_steps) > 1 else str(step_n)
                    st.markdown(f"*Paso {label}: {t['regla']}*")
                    if t["formula"]:
                        st.latex(t["formula"])
                    st.latex(rf"\int {t['term_latex']} \, dy = {t['antider_latex']}")
                step_n += 1

                st.markdown(f"**Paso {step_n} — Evaluamos en los límites de** $y$ **(Barrow) y simplificamos**")
                st.latex(
                    rf"\int_{{{sol['ay_s']}}}^{{{sol['by_s']}}} \left({sol['Fx']}\right) dy"
                    rf"= {sol['result_sym']}"
                )
                step_n += 1

                # ── Resultado exacto ─────────────────────────────────────────
                st.markdown(f"**Paso {step_n} — Resultado exacto**")
                st.latex(
                    rf"I_{{exacta}} = {sol['result_sym']} = {sol['result_num']:.10f}"
                )

            # ── Comparación ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Comparación Monte Carlo vs. Analítica")

            exact   = sol["result_num"]
            approx  = stats["integral"]
            abs_err = abs(approx - exact)
            rel_err = abs_err / abs(exact) * 100 if exact != 0 else float("inf")
            within  = stats["ci_lo"] <= exact <= stats["ci_hi"]

            within_class = "tol-ok"  if within else "tol-fail"
            within_msg   = "✔ El valor exacto cae dentro del intervalo de confianza" \
                           if within else \
                           "✘ El valor exacto cae fuera del intervalo de confianza"

            st.markdown(
                f"""
<div class="result-cards">
    <div class="result-card" style="border-top: 3px solid #2563eb;">
        <div class="rc-label">Valor exacto</div>
        <div class="rc-value">{exact:.8f}</div>
        <div class="rc-sub">I analítica</div>
    </div>
    <div class="result-card" style="border-top: 3px solid #16a34a;">
        <div class="rc-label">Aproximación Monte Carlo</div>
        <div class="rc-value">{approx:.8f}</div>
        <div class="rc-sub">I ≈</div>
    </div>
    <div class="result-card" style="border-top: 3px solid #f59e0b;">
        <div class="rc-label">Error absoluto</div>
        <div class="rc-value">{abs_err:.8f}</div>
        <div class="rc-sub">|I − I*|</div>
    </div>
    <div class="result-card" style="border-top: 3px solid #dc2626;">
        <div class="rc-label">Error relativo</div>
        <div class="rc-value">{rel_err:.4f}%</div>
        <div class="rc-sub">|I − I*| / |I*|</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("**Error absoluto y relativo**")
            st.latex(
                rf"|I_{{MC}} - I_{{exacta}}| = |{approx:.8f} - {exact:.8f}|"
                rf"= {abs_err:.8f}"
            )
            st.latex(
                rf"\text{{Error relativo}} = \frac{{|I_{{MC}} - I_{{exacta}}|}}"
                rf"{{|I_{{exacta}}|}} \times 100"
                rf"= \frac{{{abs_err:.8f}}}{{{abs(exact):.8f}}} \times 100"
                rf"= {rel_err:.4f}\%"
            )

            st.markdown("**¿El valor exacto cae dentro del intervalo de confianza?**")
            st.latex(
                rf"IC = [{stats['ci_lo']:.8f},\; {stats['ci_hi']:.8f}]"
                rf"\quad I_{{exacta}} = {exact:.8f}"
            )
            st.markdown(
                f'<span class="{within_class}">{within_msg}</span>',
                unsafe_allow_html=True,
            )

        # ── Relación entre el Error y N (NUEVO APARTADO) ─────────────────────
        st.markdown("---")
        st.subheader("Relación entre el Error y el Tamaño de Muestra (N)")

        st.markdown(
            "Como observamos en el análisis estadístico, el error estándar del "
            "método de Monte Carlo sigue una relación inversamente proporcional a "
            "la raíz cuadrada del tamaño de la muestra ($N$):"
        )
        st.latex(r"\text{Error} \propto \frac{1}{\sqrt{N}}")

        st.markdown("**¿Qué se necesita para reducir el error a la mitad?**")
        st.markdown(
            "Asumiendo que la varianza muestral ($S^2$) se mantiene constante, "
            "si queremos que el nuevo error ($E_2$) sea exactamente la mitad del "
            "error actual ($E_1$), la demostración matemática es la siguiente:"
        )

        # Paso 1
        st.latex(r"E_1 = \frac{C}{\sqrt{N_1}} \quad \text{donde } C = V \cdot S")
        st.latex(r"E_2 = \frac{E_1}{2} = \frac{1}{2} \left( \frac{C}{\sqrt{N_1}} \right)")

        # Paso 2
        st.markdown("Planteamos la ecuación para hallar el nuevo tamaño de muestra $N_2$:")
        st.latex(r"\frac{C}{\sqrt{N_2}} = \frac{C}{2\sqrt{N_1}}")

        # Paso 3
        st.markdown("Simplificamos la constante $C$ en ambos lados e invertimos:")
        st.latex(r"\sqrt{N_2} = 2\sqrt{N_1}")

        # Paso 4
        st.markdown("Por último, elevamos al cuadrado ambos términos para despejar $N_2$:")
        st.latex(r"N_2 = (2\sqrt{N_1})^2 = 4N_1")

        new_n = 4 * n
        st.markdown(
            f"**Conclusión:** Para reducir el error a la mitad, la matemática demuestra "
            f"que siempre es necesario **cuadruplicar** el tamaño de la muestra. "
            f"Partiendo de tu muestra actual $N_1 = {n}$, el nuevo valor debería ser:"
        )
        st.latex(rf"N_2 = 4 \times {n} = {new_n}")