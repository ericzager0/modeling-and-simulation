import math
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _local_dict():
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    return x, y, {
        "x":    x,    "y":    y,
        "e":    sp.E, "E":    sp.E,
        "pi":   sp.pi,
        "ln":   sp.log,  "log":  sp.log,
        "exp":  sp.exp,
        "sin":  sp.sin,  "cos":  sp.cos,  "tan":  sp.tan,
        "sqrt": sp.sqrt,
        "Abs":  sp.Abs,  "abs":  sp.Abs,
    }


def _parse_ode(func_str: str):
    """Parsea la RHS de y' = f(x, y). Devuelve (callable, expr, latex)."""
    x, y, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f    = sp.lambdify((x, y), expr, modules=["numpy"])
    return f, expr, sp.latex(expr)


def _parse_val(s: str) -> float:
    _, _, ld = _local_dict()
    return float(sp.sympify(s.strip().replace("^", "**"), locals=ld))


def _fmt(v, prec: int = 8) -> str:
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return str(fv)
        rnd = round(fv, prec)
        return str(int(rnd)) if rnd == int(rnd) and abs(rnd) < 1e15 else f"{fv:.{prec}g}"
    except Exception:
        return str(v)


def _safe_real(v, tol: float = 1e-6):
    """Convierte v a float si es (esencialmente) real; si no, devuelve None."""
    try:
        c = complex(v)
    except Exception:
        return None
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        return None
    if abs(c.imag) > tol * max(1.0, abs(c.real)):
        return None
    return c.real


def _solve_exact(expr, t0: float, y0: float):
    """
    Intenta resolver y' = expr con y(x0) = y0 de forma simbólica.
    Devuelve (callable, rhs_expr) o (None, None).

    Estrategia en dos pasos:
      1) Camino rápido: dsolve con ics directamente (funciona en la mayoría
         de los casos lineales / estándar).
      2) Si falla (p.ej. porque la condición inicial produce varias
         soluciones posibles para la constante, algo común con raíces o
         potencias pares), se obtiene la solución general, se despeja la
         constante de integración a mano, y cada candidata se valida
         numéricamente comprobando que su derivada coincide con f(x,y)
         cerca de x0 — esto descarta ramas espurias (p.ej. de sqrt) y
         se queda con la solución correcta.
    """
    x, y, _ = _local_dict()
    yf = sp.Function("y")
    x0_s = sp.nsimplify(t0, rational=True)
    y0_s = sp.nsimplify(y0, rational=True)
    ode_eq = sp.Eq(yf(x).diff(x), expr.subs(y, yf(x)))

    candidates = []

    # ── Paso 1: camino rápido ────────────────────────────────────────────
    try:
        sol = sp.dsolve(ode_eq, yf(x), ics={yf(x0_s): y0_s})
        sols = sol if isinstance(sol, list) else [sol]
        candidates.extend(s.rhs for s in sols if s.lhs == yf(x))
    except Exception:
        pass

    # ── Paso 2: solución general + despeje manual de la constante ───────
    if not candidates:
        try:
            gen = sp.dsolve(ode_eq, yf(x))
            gens = gen if isinstance(gen, list) else [gen]
            for g in gens:
                explicit_list = [g.rhs] if g.lhs == yf(x) else []
                if not explicit_list:
                    try:
                        explicit_list = list(sp.solve(g, yf(x)))
                    except Exception:
                        continue
                for rhs in explicit_list:
                    consts = sorted(rhs.free_symbols - {x}, key=str)
                    if len(consts) != 1:
                        continue  # solo manejamos 1 constante (EDO de 1er orden)
                    c = consts[0]
                    try:
                        csols = sp.solve(sp.Eq(rhs.subs(x, x0_s), y0_s), c)
                    except Exception:
                        continue
                    for cs in csols:
                        if cs.is_real is False:
                            continue
                        candidates.append(rhs.subs(c, cs))
        except Exception:
            pass

    if not candidates:
        return None, None

    # ── Validación numérica: y(x0)=y0 y además y'(x) ≈ f(x, y(x)) ───────
    f_num = sp.lambdify((x, y), expr, modules=["numpy"])
    t0f, y0f = float(t0), float(y0)
    test_xs = [t0f + dx for dx in (1e-3, 1e-2, 5e-2, 0.1)]
    hfd = 1e-6

    for cand in candidates:
        try:
            cand_s = sp.simplify(cand)
            func = sp.lambdify(x, cand_s, modules=["numpy"])

            y0_check = _safe_real(func(t0f))
            if y0_check is None or abs(y0_check - y0f) > 1e-6 * max(1, abs(y0f)):
                continue

            ok = True
            for tx in test_xs:
                yv = _safe_real(func(tx))
                if yv is None:
                    ok = False
                    break
                y_fwd = _safe_real(func(tx + hfd))
                y_bwd = _safe_real(func(tx - hfd))
                if y_fwd is None or y_bwd is None:
                    ok = False
                    break
                dyv  = (y_fwd - y_bwd) / (2 * hfd)
                fval = _safe_real(f_num(tx, yv))
                if fval is None:
                    continue
                if abs(dyv - fval) > 1e-3 * max(1, abs(fval)):
                    ok = False
                    break
            if ok:
                return func, cand_s
        except Exception:
            continue

    return None, None


def _classify_hint_es(expr, x, y):
    """Devuelve el nombre del primer 'hint' que sympy usaría para resolver
    la EDO (vía classify_ode), en formato legible, o None si no se puede
    clasificar."""
    yf = sp.Function("y")
    ode_eq = sp.Eq(yf(x).diff(x), expr.subs(y, yf(x)))
    try:
        hints = sp.classify_ode(ode_eq, yf(x))
        if hints:
            return hints[0].replace("_", " ")
    except Exception:
        pass
    return None


def _real_const(csol_list):
    """Dada una lista de soluciones candidatas para la constante C (resultado
    de sp.solve), devuelve la primera que sea real, o None si ninguna lo es.
    Esto evita mostrar pasos con una rama compleja espuria (p.ej. originada
    por log de un número negativo)."""
    for cs in csol_list:
        try:
            cs_s = sp.simplify(cs)
            if cs_s.is_real is False:
                continue
            if abs(complex(cs_s).imag) > 1e-9:
                continue
            return cs_s
        except Exception:
            continue
    return None


def _build_solution_steps(expr, t0: float, y0: float, y_ex_expr):
    """
    Construye, en la medida de lo posible, una explicación paso a paso de
    cómo se llega a la solución exacta `y_ex_expr` de  y' = expr,  y(t0)=y0.

    Devuelve una lista de tuplas ('md', texto_markdown) | ('latex', codigo_latex)
    para ser renderizadas en la UI. Se reconocen automáticamente tres familias
    de EDO de primer orden (las más comunes en la práctica):

      1) Integración directa   →  y' = g(x)               (no depende de y)
      2) Variables separables  →  y' = g(x) · h(y)
      3) Lineal de 1er orden   →  y' + P(x) y = Q(x)

    Si la ecuación no encaja en ninguno de estos esquemas (p.ej. Bernoulli,
    exacta, homogénea, etc.) se devuelve una explicación más general que
    identifica el tipo de ecuación detectado por SymPy y muestra la solución
    final ya verificada.
    """
    x, y, _ = _local_dict()
    expr = sp.sympify(expr)
    x0_l = sp.nsimplify(t0, rational=True)
    y0_l = sp.nsimplify(y0, rational=True)
    C = sp.Symbol("C")

    # ── Caso 1: integración directa (el lado derecho no depende de y) ──────
    if y not in expr.free_symbols:
        try:
            steps = []
            steps.append(("md",
                "**Tipo de ecuación: integración directa.** El lado derecho "
                "no depende de $y$, por lo que la ecuación se resuelve "
                "integrando directamente respecto de $x$."))
            steps.append(("latex", rf"\frac{{dy}}{{dx}} = {sp.latex(expr)}"))
            steps.append(("md", "Se integran ambos miembros respecto de $x$:"))
            G = sp.integrate(expr, x)
            steps.append(("latex",
                rf"y(x) = \int \Big({sp.latex(expr)}\Big)\,dx = {sp.latex(G)} + C"))
            Csol = sp.solve(sp.Eq(G.subs(x, x0_l) + C, y0_l), C)
            Cval = _real_const(Csol)
            if Cval is not None:
                steps.append(("md",
                    rf"Se aplica la condición inicial $y({_fmt(t0)})={_fmt(y0)}$ "
                    rf"para hallar $C$:"))
                steps.append(("latex",
                    rf"{sp.latex(G.subs(x, x0_l))} + C = {sp.latex(y0_l)} "
                    rf"\;\;\Longrightarrow\;\; C = {sp.latex(Cval)}"))
                steps.append(("md", "**Solución particular:**"))
                steps.append(("latex",
                    rf"y(x) = {sp.latex(sp.simplify(G + Cval))}"))
                return steps
        except Exception:
            pass

    # ── Caso 2: variables separables  y' = g(x)·h(y) ────────────────────────
    try:
        sep = sp.separatevars(sp.together(sp.expand(expr)), symbols=(x, y), dict=True)
    except Exception:
        sep = None

    if isinstance(sep, dict):
        g = sp.simplify(sep.get("coeff", sp.Integer(1)) * sep.get(x, sp.Integer(1)))
        h = sp.simplify(sep.get(y, sp.Integer(1)))
        if y in h.free_symbols:
            try:
                steps = []
                steps.append(("md",
                    "**Tipo de ecuación: variables separables.** El lado "
                    "derecho puede escribirse como el producto de una función "
                    "de $x$ por una función de $y$, lo que permite agrupar "
                    "cada variable a un lado de la igualdad."))
                steps.append(("latex",
                    rf"\frac{{dy}}{{dx}} = \underbrace{{{sp.latex(g)}}}_{{g(x)}}"
                    rf"\;\cdot\;\underbrace{{{sp.latex(h)}}}_{{h(y)}}"))
                steps.append(("md",
                    "Se separan las variables, dividiendo por $h(y)$ "
                    "(suponiendo $h(y)\\neq 0$):"))
                steps.append(("latex",
                    rf"\frac{{dy}}{{{sp.latex(h)}}} = {sp.latex(g)}\;dx"))
                steps.append(("md", "Se integra cada miembro por separado:"))
                H = sp.integrate(1 / h, y)
                G = sp.integrate(g, x)
                steps.append(("latex",
                    rf"\int \frac{{dy}}{{{sp.latex(h)}}} = \int {sp.latex(g)}\;dx"))
                steps.append(("latex", rf"{sp.latex(H)} = {sp.latex(G)} + C"))
                Csol = sp.solve(sp.Eq(H.subs(y, y0_l), G.subs(x, x0_l) + C), C)
                Cval = _real_const(Csol)
                if Cval is not None:
                    steps.append(("md",
                        rf"Se aplica la condición inicial $y({_fmt(t0)})={_fmt(y0)}$ "
                        rf"para hallar $C$:"))
                    steps.append(("latex",
                        rf"{sp.latex(H.subs(y, y0_l))} = {sp.latex(G.subs(x, x0_l))} + C "
                        rf"\;\;\Longrightarrow\;\; C = {sp.latex(Cval)}"))
                    steps.append(("md",
                        "Reemplazando $C$ y despejando $y$ se obtiene la "
                        "solución particular:"))
                    steps.append(("latex",
                        rf"{sp.latex(H)} = {sp.latex(G)} + \left({sp.latex(Cval)}\right)"))
                    steps.append(("latex",
                        rf"y(x) = {sp.latex(sp.simplify(y_ex_expr))}"))
                    return steps
            except Exception:
                pass

    # ── Caso 3: lineal de primer orden  y' + P(x) y = Q(x) ──────────────────
    try:
        poly = sp.Poly(sp.expand(expr), y)
        deg  = poly.degree()
    except Exception:
        poly, deg = None, None

    if poly is not None and deg is not None and deg <= 1:
        coeffs = poly.all_coeffs()
        a1, a0 = (coeffs if deg == 1 else (sp.Integer(0), coeffs[0] if coeffs else sp.Integer(0)))
        if y not in a1.free_symbols and y not in a0.free_symbols:
            P = sp.simplify(-a1)
            Q = sp.simplify(a0)
            try:
                steps = []
                steps.append(("md",
                    "**Tipo de ecuación: lineal de primer orden.** Se puede "
                    "escribir en la forma estándar $y' + P(x)\\,y = Q(x)$, lo "
                    "que permite resolverla mediante un **factor integrante**."))
                steps.append(("latex",
                    rf"y' + \underbrace{{{sp.latex(P)}}}_{{P(x)}}\,y = "
                    rf"\underbrace{{{sp.latex(Q)}}}_{{Q(x)}}"))
                steps.append(("md", "Se calcula el factor integrante:"))
                intP = sp.integrate(P, x)
                mu = sp.simplify(sp.exp(intP))
                steps.append(("latex",
                    rf"\mu(x) = e^{{\int P(x)\,dx}} = e^{{{sp.latex(intP)}}} = {sp.latex(mu)}"))
                steps.append(("md",
                    "Se multiplican ambos miembros de la ecuación por $\\mu(x)$: "
                    "el lado izquierdo queda como la derivada de un producto, "
                    "$(\\mu(x)\\,y)'$:"))
                rhs = sp.simplify(mu * Q)
                steps.append(("latex", rf"\big(\mu(x)\,y\big)' = \mu(x)\,Q(x) = {sp.latex(rhs)}"))
                steps.append(("md", "Se integran ambos miembros respecto de $x$:"))
                rhs_int = sp.integrate(rhs, x)
                steps.append(("latex",
                    rf"\mu(x)\,y = \int {sp.latex(rhs)}\,dx = {sp.latex(rhs_int)} + C"))
                steps.append(("md", "Se despeja $y$:"))
                y_gen = sp.simplify((rhs_int + C) / mu)
                steps.append(("latex",
                    rf"y(x) = \dfrac{{{sp.latex(rhs_int)} + C}}{{{sp.latex(mu)}}}"))
                Csol = sp.solve(sp.Eq(y_gen.subs(x, x0_l), y0_l), C)
                Cval = _real_const(Csol)
                if Cval is not None:
                    steps.append(("md",
                        rf"Se aplica la condición inicial $y({_fmt(t0)})={_fmt(y0)}$ "
                        rf"para hallar $C$:"))
                    steps.append(("latex", rf"C = {sp.latex(Cval)}"))
                    steps.append(("md", "**Solución particular:**"))
                    steps.append(("latex",
                        rf"y(x) = {sp.latex(sp.simplify(y_gen.subs(C, Cval)))}"))
                    return steps
            except Exception:
                pass

    # ── Fallback: tipo no reconocido por los 3 esquemas anteriores ──────────
    steps = []
    hint = _classify_hint_es(expr, x, y)
    if hint:
        steps.append(("md", rf"**Tipo de ecuación detectado por SymPy:** *{hint}*."))
    steps.append(("md",
        "Esta ecuación se resolvió con un método simbólico específico para "
        "este tipo de EDO. La derivación detallada de este caso particular "
        "no se descompone automáticamente paso a paso, pero la solución "
        "obtenida fue **verificada numéricamente**: se comprobó que cumple "
        "tanto la condición inicial como la propia ecuación diferencial "
        "$y' = f(x,y)$ en varios puntos cercanos a $x_0$."))
    steps.append(("md", "**Solución particular obtenida:**"))
    steps.append(("latex", rf"y(x) = {sp.latex(sp.simplify(y_ex_expr))}"))
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Métodos numéricos  →  list[dict]
# ─────────────────────────────────────────────────────────────────────────────

def _euler(f, t0, y0, t_end, h):
    N = round((t_end - t0) / h)
    rows, t_n, y_n = [], t0, y0
    for k in range(N):
        y_next = y_n + h * f(t_n, y_n)
        rows.append({"n": k, "x": t_n, "y": y_n, "y_next": y_next})
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({"n": N, "x": t_n, "y": y_n, "y_next": None})
    return rows


def _heun(f, t0, y0, t_end, h):
    N = round((t_end - t0) / h)
    rows, t_n, y_n = [], t0, y0
    for k in range(N):
        k1     = f(t_n, y_n)
        k2     = f(t_n + h, y_n + h * k1)
        y_next = y_n + (h / 2) * (k1 + k2)
        rows.append({"n": k, "x": t_n, "y": y_n, "y_next": y_next})
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({"n": N, "x": t_n, "y": y_n, "y_next": None})
    return rows


def _rk4(f, t0, y0, t_end, h):
    N = round((t_end - t0) / h)
    rows, t_n, y_n = [], t0, y0
    for k in range(N):
        k1     = f(t_n,       y_n)
        k2     = f(t_n + h/2, y_n + (h/2) * k1)
        k3     = f(t_n + h/2, y_n + (h/2) * k2)
        k4     = f(t_n + h,   y_n +  h     * k3)
        y_next = y_n + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        rows.append({
            "n": k, "x": t_n, "y": y_n,
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "y_next": y_next,
        })
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({
        "n": N, "x": t_n, "y": y_n,
        "k1": None, "k2": None, "k3": None, "k4": None,
        "y_next": None,
    })
    return rows


def _enrich(rows: list, y_exact_func) -> list:
    """Agrega y_real y error a cada fila si existe solución exacta."""
    if y_exact_func is None:
        return rows
    for r in rows:
        try:
            yr = float(y_exact_func(r["x"]))
            r["y_real"] = yr
            r["error"]  = abs(yr - r["y"])
        except Exception:
            r["y_real"] = None
            r["error"] = None
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Constructores de DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def _col_cfg(df: pd.DataFrame) -> dict:
    cfg = {}
    for col in df.columns:
        if col == "n":
            cfg[col] = st.column_config.NumberColumn("n", format="%d")
        elif col == "|E|":
            cfg[col] = st.column_config.NumberColumn("|E|", format="%.4e")
        else:
            cfg[col] = st.column_config.NumberColumn(col, format="%.6f")
    return cfg


def _df_euler_heun(rows: list, has_exact: bool) -> pd.DataFrame:
    d = {
        "n":     [r["n"]           for r in rows],
        "xₙ":    [r["x"]           for r in rows],
        "yₙ":    [r["y"]           for r in rows],
        "yₙ₊₁": [r.get("y_next")  for r in rows],
    }
    if has_exact:
        d["y_real"] = [r.get("y_real") for r in rows]
        d["|E|"]    = [r.get("error")  for r in rows]
    return pd.DataFrame(d)


def _df_rk4(rows: list, has_exact: bool) -> pd.DataFrame:
    d = {
        "n":     [r["n"]           for r in rows],
        "xₙ":    [r["x"]           for r in rows],
        "yₙ":    [r["y"]           for r in rows],
        "k₁":    [r.get("k1")      for r in rows],
        "k₂":    [r.get("k2")      for r in rows],
        "k₃":    [r.get("k3")      for r in rows],
        "k₄":    [r.get("k4")      for r in rows],
        "yₙ₊₁": [r.get("y_next")  for r in rows],
    }
    if has_exact:
        d["y_real"] = [r.get("y_real") for r in rows]
        d["|E|"]    = [r.get("error")  for r in rows]
    return pd.DataFrame(d)


def _df_comparison(e_rows, h_rows, r_rows, has_exact) -> pd.DataFrame:
    d = {
        "n":       [r["n"]  for r in e_rows],
        "xₙ":      [r["x"]  for r in e_rows],
    }
    if has_exact:
        d["y_real"] = [r.get("y_real") for r in e_rows]
    d["y_Euler"] = [r["y"] for r in e_rows]
    d["y_Heun"]  = [r["y"] for r in h_rows]
    d["y_RK4"]   = [r["y"] for r in r_rows]
    return pd.DataFrame(d)


# ─────────────────────────────────────────────────────────────────────────────
# Página
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

    .result-cards { display: flex; gap: 1.2rem; margin: 0.6rem 0 1.6rem 0; flex-wrap: wrap; }
    .result-card {
        flex: 1; min-width: 140px;
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
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        line-height: 1;
    }
    .result-card .rc-sub {
        font-size: .9rem;
        opacity: .35;
        margin-top: .35rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Teoría (desplegable, arriba de todo) ───────────────────────────────────
    with st.expander("📘 Teoría: ¿Cómo funcionan los métodos de Euler, Heun y RK4?", expanded=False):
        st.markdown(r"""
### ¿Qué hacen estos tres métodos?

Euler, Heun y Runge–Kutta de orden 4 (RK4) son **métodos numéricos para resolver
problemas de valor inicial (PVI)** de la forma:
""")
        st.latex(r"y' = f(x,y), \qquad y(x_0) = y_0")
        st.markdown(r"""
Cuando no se puede (o es muy difícil) encontrar la solución exacta $y(x)$ de forma
analítica, estos métodos permiten **aproximarla numéricamente**, generando una sucesión
de puntos $(x_0,y_0),\,(x_1,y_1),\,(x_2,y_2),\dots$ que se acercan a la curva real.
""")

        st.markdown("### ¿Qué tienen en común?")
        st.markdown(r"""
1. **Discretizan el intervalo.** Dividen $[x_0,\,x_{end}]$ en pasos de tamaño $h$,
   generando los puntos $x_n = x_0 + n\,h$.
2. **Son métodos de un paso ("one-step").** Para calcular $y_{n+1}$ solo usan la
   información del punto anterior $(x_n, y_n)$ — no necesitan recordar todo el
   historial previo, a diferencia de los métodos multipaso.
3. **Comparten la misma estructura general:**
""")
        st.latex(r"y_{n+1} = y_n + h \cdot \varphi(x_n,\, y_n,\, h)")
        st.markdown(r"""
   donde $\varphi$ es una *función de incremento* que estima la **pendiente promedio**
   de la solución dentro del tramo $[x_n,\,x_n+h]$, a partir de una o más evaluaciones
   de $f(x,y)$.
4. **Lo único que cambia entre ellos es cómo se calcula esa pendiente $\varphi$:**
   cuántas veces se evalúa $f$ por paso y en qué puntos del intervalo. Esa diferencia
   es justamente lo que determina qué tan preciso es cada método (su **orden** de
   convergencia).
5. **Ninguno da la solución exacta.** En cada paso se comete un pequeño error de
   *truncamiento local*, que se va acumulando a lo largo de todos los pasos hasta
   convertirse en el error *global*. Cuantas más evaluaciones de $f$ use el método
   por paso, menor es ese error — pero más caro resulta computacionalmente.
""")

        st.markdown("---")
        st.markdown("### 1️⃣ Método de Euler (orden 1)")
        st.latex(r"y_{n+1} = y_n + h \cdot f(x_n,\, y_n)")
        st.markdown(r"""
Es el más simple de los tres. En cada paso:

1. Evalúa la pendiente de la curva **una sola vez**, en el punto actual:
   $k_1 = f(x_n, y_n)$.
2. Avanza en línea recta con esa pendiente durante todo el paso $h$.

Geométricamente equivale a seguir la **recta tangente** a la solución en $(x_n, y_n)$
a lo largo de un tramo de longitud $h$. Como la pendiente solo se calcula al *inicio*
del intervalo, si la curva se curva mucho dentro del paso, el método se va "despegando"
de la solución real.

Es un método de **orden 1**: el error de truncamiento local es $O(h^2)$ y el error
global acumulado es $O(h)$. Es el más económico de los tres (1 sola evaluación de $f$
por paso), pero también el menos preciso.
""")

        st.markdown("### 2️⃣ Método de Heun (orden 2, Euler mejorado)")
        st.markdown(r"""
Heun corrige el principal defecto de Euler —usar solo la pendiente al inicio del
paso— agregando un esquema de **predicción + corrección**:
""")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(r"""
**Predictor** — igual que Euler: calcula la pendiente al inicio y, con ella, estima
provisoriamente el punto siguiente.
""")
            st.latex(r"k_1 = f(x_n,\, y_n)")
            st.latex(r"\hat{y}_{n+1} = y_n + h\,k_1")
        with col_t2:
            st.markdown(r"""
**Corrector** — evalúa la pendiente en ese punto estimado, y promedia ambas
pendientes para dar el paso definitivo.
""")
            st.latex(r"k_2 = f(x_n+h,\, \hat{y}_{n+1})")
            st.latex(r"y_{n+1} = y_n + \dfrac{h}{2}\,(k_1 + k_2)")
        st.markdown(r"""
En vez de usar solo la pendiente al inicio del paso (como Euler), Heun también "mira"
cuál sería la pendiente al final del paso —estimada con el predictor— y avanza con el
**promedio** de ambas. Eso compensa el sesgo direccional de Euler y mejora notablemente
la precisión.

Es un método de **orden 2** (también llamado RK2): error local $O(h^3)$, error global
$O(h^2)$. A cambio de esa mejora, paga el costo de evaluar $f$ **dos veces** por paso
en lugar de una.
""")

        st.markdown("### 3️⃣ Runge–Kutta de orden 4 (RK4)")
        st.markdown(r"""
RK4 lleva la idea de Heun —promediar varias pendientes dentro del paso— un poco más
lejos: evalúa $f$ en **cuatro** puntos distintos del intervalo $[x_n,\,x_n+h]$,
usando cada pendiente recién calculada para "asomarse" un poco más adentro del paso:
""")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.latex(r"k_1 = f(x_n,\, y_n)")
            st.latex(r"k_2 = f\!\left(x_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_1\right)")
            st.latex(r"k_3 = f\!\left(x_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_2\right)")
            st.latex(r"k_4 = f(x_n+h,\; y_n+h\,k_3)")
        with col_r2:
            st.markdown(r"""
- $k_1$: pendiente al **inicio** del paso.
- $k_2$: pendiente en el **punto medio**, estimada avanzando medio paso con $k_1$.
- $k_3$: una **segunda estimación** de la pendiente en el punto medio, esta vez
  avanzando medio paso con $k_2$ (más afinada que $k_2$).
- $k_4$: pendiente al **final** del paso, estimada avanzando el paso completo
  con $k_3$.
""")
        st.markdown("Luego combina las cuatro pendientes con un promedio ponderado:")
        st.latex(r"y_{n+1} = y_n + \dfrac{h}{6}\,(k_1 + 2k_2 + 2k_3 + k_4)")
        st.markdown(r"""
Los pesos $1,2,2,1$ le dan más peso a las dos estimaciones del punto medio —que son
más representativas de cómo se comporta la curva *dentro* del paso— la misma idea que
está detrás de la regla de Simpson para integrar.

Es un método de **orden 4**: error local $O(h^5)$, error global $O(h^4)$. Es mucho más
preciso que Euler y Heun usando el mismo paso $h$, a costa de evaluar $f$ **cuatro
veces** por paso.
""")

        st.markdown("---")
        st.markdown("### Comparación rápida")
        st.markdown(r"""
| Método | Evaluaciones de $f$ por paso | Orden | Error global | Costo computacional |
|---|:---:|:---:|:---:|:---:|
| Euler | 1 | 1 | $O(h)$   | Bajo  |
| Heun  | 2 | 2 | $O(h^2)$ | Medio |
| RK4   | 4 | 4 | $O(h^4)$ | Alto  |
""")
        st.markdown(r"""
**¿Por qué importa el orden?** Si se reduce el paso $h$ a la mitad, el error de Euler
se reduce aproximadamente a la mitad (factor $2$), el de Heun a un cuarto (factor $4$)
y el de RK4 a un dieciseisavo (factor $16$). Por eso RK4 logra mucha más precisión sin
necesitar pasos extremadamente chicos, aunque pague el precio de evaluar $f$ más veces
por paso.
""")

    st.title("Métodos Numéricos para EDOs")

    # ── EDO ───────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "y' = f(x, y)  —  lado derecho de la ecuación diferencial",
        value="",
        placeholder="Ej: y - x**2 + 1   |   -2*x*y   |   sin(x) + y",
    )
    f_func = f_expr = latex_f = None
    if func_str.strip():
        try:
            f_func, f_expr, latex_f = _parse_ode(func_str)
            st.latex(rf"\dfrac{{dy}}{{dx}} \;=\; {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Condición inicial ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        t0_str = st.text_input("x₀  —  punto inicial", value="0",
                               placeholder="Ej: 0")
    with c2:
        y0_str = st.text_input("y₀  —  valor inicial  y(x₀)", value="",
                               placeholder="Ej: 0.5")

    t0 = y0 = None
    if t0_str.strip() and y0_str.strip():
        try:
            t0 = _parse_val(t0_str)
            y0 = _parse_val(y0_str)
            st.latex(rf"y\!\left({_fmt(t0)}\right) = {_fmt(y0)}")
        except Exception:
            st.error("Valores inválidos para la condición inicial.")

    # ── t_end ─────────────────────────────────────────────────────────────────
    tend_str = st.text_input(
        "x_end  —  extremo derecho del intervalo",
        value="", placeholder="Ej: 2"
    )
    t_end = None
    if tend_str.strip():
        try:
            t_end = _parse_val(tend_str)
            if t0 is not None:
                st.latex(
                    rf"x \;\in\; \left[\,{_fmt(t0)},\;{_fmt(t_end)}\,\right]"
                )
        except Exception:
            st.error("Valor inválido para t_end.")

    # ── Paso h ────────────────────────────────────────────────────────────────
    h_str = st.text_input(
        "h  —  paso de integración",
        value="", placeholder="Ej: 0.25  |  0.1  |  1/4"
    )
    h_val = None
    if h_str.strip():
        try:
            h_val = _parse_val(h_str)
            if h_val <= 0:
                st.error("El paso h debe ser positivo.")
                h_val = None
        except Exception:
            st.error("Valor inválido para h.")

    # ── Botón ─────────────────────────────────────────────────────────────────
    calc = st.button("Calcular", use_container_width=True)

    if calc:
        missing = []
        if f_func is None:  missing.append("la EDO")
        if t0 is None:      missing.append("x₀ e y₀")
        if t_end is None:   missing.append("x_end")
        if h_val is None:   missing.append("h")
        if missing:
            st.error(f"Faltá ingresar: {', '.join(missing)}.")
            st.stop()
        if t_end <= t0:
            st.error("x_end debe ser mayor que x₀.")
            st.stop()
        if h_val >= t_end - t0:
            st.error("El paso h es demasiado grande para el intervalo dado.")
            st.stop()

        N_total = round((t_end - t0) / h_val)
        if N_total > 500:
            st.warning(
                f"Con h = {_fmt(h_val)} se generarían {N_total} pasos. "
                "Se muestran los primeros 500."
            )
            t_end_eff = t0 + 500 * h_val
        else:
            t_end_eff = t_end

        # ── Solución exacta simbólica ─────────────────────────────────────────
        with st.spinner("Buscando solución exacta simbólica..."):
            y_ex_func, y_ex_expr = (
                _solve_exact(f_expr, t0, y0) if f_expr is not None
                else (None, None)
            )
        has_exact = y_ex_func is not None

        # ── Integración numérica ──────────────────────────────────────────────
        try:
            e_rows = _enrich(_euler(f_func, t0, y0, t_end_eff, h_val), y_ex_func)
            h_rows = _enrich(_heun( f_func, t0, y0, t_end_eff, h_val), y_ex_func)
            r_rows = _enrich(_rk4(  f_func, t0, y0, t_end_eff, h_val), y_ex_func)
        except Exception as exc:
            st.error(f"Error durante la integración numérica: {exc}")
            st.stop()

        st.session_state["edo_results"] = {
            "e_rows":    e_rows,
            "h_rows":    h_rows,
            "r_rows":    r_rows,
            "has_exact": has_exact,
            "y_ex_func": y_ex_func,
            "y_ex_expr": y_ex_expr,
            "t0":        t0,
            "y0":        y0,
            "t_end":     t_end_eff,
            "h":         h_val,
            "f_expr":    f_expr,
        }

    # ── Guard: nada calculado aún ─────────────────────────────────────────────
    if "edo_results" not in st.session_state:
        return

    res       = st.session_state["edo_results"]
    e_rows    = res["e_rows"]
    h_rows    = res["h_rows"]
    r_rows    = res["r_rows"]
    has_exact = res["has_exact"]
    t0_r      = res["t0"]
    t_end_r   = res["t_end"]

    # ── Banner solución exacta ────────────────────────────────────────────────
    st.markdown("---")
    if has_exact:
        st.success(
            rf"Solución exacta encontrada: $\quad y(x) = {sp.latex(res['y_ex_expr'])}$"
        )
        with st.expander("🧮 Ver el paso a paso de cómo se llegó a esta solución", expanded=False):
            try:
                steps = _build_solution_steps(
                    res["f_expr"], t0_r, res["y0"], res["y_ex_expr"]
                )
            except Exception as exc:
                steps = [("md", f"No se pudo generar el detalle paso a paso ({exc}).")]
            for kind, content in steps:
                if kind == "latex":
                    st.latex(content)
                else:
                    st.markdown(content)
    else:
        st.info(
            "No se encontró solución exacta analítica. "
            "Las columnas **y_real** y **|E|** no se muestran."
        )

    # ── Result cards ──────────────────────────────────────────────────────────
    fe = e_rows[-1]["y"]
    fh = h_rows[-1]["y"]
    fr = r_rows[-1]["y"]

    cards = f"""<div class="result-cards">
        <div class="result-card" style="border-top:3px solid #dc2626;">
            <div class="rc-label">Euler — y(x_end)</div>
            <div class="rc-value">{_fmt(fe, 6)}</div>
            <div class="rc-sub">x = {_fmt(t_end_r)}</div>
        </div>
        <div class="result-card" style="border-top:3px solid #f59e0b;">
            <div class="rc-label">Heun — y(x_end)</div>
            <div class="rc-value">{_fmt(fh, 6)}</div>
            <div class="rc-sub">x = {_fmt(t_end_r)}</div>
        </div>
        <div class="result-card" style="border-top:3px solid #16a34a;">
            <div class="rc-label">RK4 — y(x_end)</div>
            <div class="rc-value">{_fmt(fr, 6)}</div>
            <div class="rc-sub">x = {_fmt(t_end_r)}</div>
        </div>"""
    if has_exact:
        fx = float(res["y_ex_func"](t_end_r))
        cards += f"""
        <div class="result-card" style="border-top:3px solid #2563eb;">
            <div class="rc-label">Exacto — y(x_end)</div>
            <div class="rc-value">{_fmt(fx, 6)}</div>
            <div class="rc-sub">y_real</div>
        </div>"""
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tabs — una por método
    # ══════════════════════════════════════════════════════════════════════════
    tab_e, tab_h, tab_r = st.tabs(["  Euler  ", "  Heun  ", "  RK4  "])

    with tab_e:
        st.markdown("#### Método de Euler")
        st.latex(r"y_{n+1} = y_n + h \cdot f(x_n,\; y_n)")
        df_e = _df_euler_heun(e_rows, has_exact)
        st.dataframe(df_e, column_config=_col_cfg(df_e),
                     hide_index=True, use_container_width=True)

    with tab_h:
        st.markdown("#### Método de Heun  (Euler mejorado)")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"k_1 = f(x_n,\; y_n)")
            st.latex(r"k_2 = f(x_n + h,\; y_n + h\,k_1)")
        with col2:
            st.latex(r"y_{n+1} = y_n + \dfrac{h}{2}\,(k_1 + k_2)")
        df_h = _df_euler_heun(h_rows, has_exact)
        st.dataframe(df_h, column_config=_col_cfg(df_h),
                     hide_index=True, use_container_width=True)

    with tab_r:
        st.markdown("#### Runge–Kutta de orden 4")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"k_1 = f(x_n,\; y_n)")
            st.latex(
                r"k_2 = f\!\left(x_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_1\right)"
            )
            st.latex(
                r"k_3 = f\!\left(x_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_2\right)"
            )
            st.latex(r"k_4 = f(x_n+h,\; y_n+h\,k_3)")
        with col2:
            st.latex(
                r"y_{n+1} = y_n + \dfrac{h}{6}\,(k_1 + 2k_2 + 2k_3 + k_4)"
            )
        df_r = _df_rk4(r_rows, has_exact)
        st.dataframe(df_r, column_config=_col_cfg(df_r),
                     hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Comparación
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Comparación de métodos")

    df_cmp = _df_comparison(e_rows, h_rows, r_rows, has_exact)
    st.dataframe(df_cmp, column_config=_col_cfg(df_cmp),
                 hide_index=True, use_container_width=True)

    # ── Gráfico ───────────────────────────────────────────────────────────────
    st.markdown("#### Gráfico")

    ck1, ck2, ck3, ck4 = st.columns(4)
    show_ex = ck1.checkbox("Solución exacta", value=has_exact,
                           disabled=not has_exact, key="chk_exact")
    show_eu = ck2.checkbox("Euler",  value=True, key="chk_euler")
    show_hu = ck3.checkbox("Heun",   value=True, key="chk_heun")
    show_rk = ck4.checkbox("RK4",    value=True, key="chk_rk4")

    t_pts = [r["x"] for r in e_rows]
    fig   = go.Figure()

    if show_ex and has_exact:
        t_dense = np.linspace(t0_r, t_end_r, 600)
        try:
            y_dense = np.array(res["y_ex_func"](t_dense), dtype=float)
        except Exception:
            y_dense = np.array(
                [float(res["y_ex_func"](tv)) for tv in t_dense]
            )
        y_dense = np.where(np.isfinite(y_dense), y_dense, np.nan)
        fig.add_trace(go.Scatter(
            x=t_dense, y=y_dense,
            mode="lines", name="Solución exacta",
            line=dict(color="#2563eb", width=2.5, dash="dash"),
        ))

    if show_eu:
        fig.add_trace(go.Scatter(
            x=t_pts, y=[r["y"] for r in e_rows],
            mode="lines+markers", name="Euler",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=5),
        ))

    if show_hu:
        fig.add_trace(go.Scatter(
            x=t_pts, y=[r["y"] for r in h_rows],
            mode="lines+markers", name="Heun",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
        ))

    if show_rk:
        fig.add_trace(go.Scatter(
            x=t_pts, y=[r["y"] for r in r_rows],
            mode="lines+markers", name="RK4",
            line=dict(color="#16a34a", width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y(x)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        zeroline=True, zerolinecolor="rgba(128,128,128,0.35)", zerolinewidth=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        zeroline=True, zerolinecolor="rgba(128,128,128,0.35)", zerolinewidth=1,
    )
    st.plotly_chart(fig, use_container_width=True)