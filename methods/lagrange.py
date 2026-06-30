import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _local_dict():
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


def _parse_f(func_str: str):
    """Devuelve (callable, sympy_expr, latex_str) o lanza excepción."""
    x, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f = sp.lambdify(x, expr, modules=["numpy"])
    return f, expr, sp.latex(expr)


def _fmt(v) -> str:
    """Formatea un número: quita el .0 si el valor es entero."""
    try:
        fv = round(float(v), 10)          # elimina ruido de punto flotante
        return str(int(fv)) if fv == int(fv) else str(fv)
    except Exception:
        return str(v)


def _parse_val(s: str) -> float:
    """Convierte un string que puede contener pi, e, e**2, etc. a float."""
    _, ld = _local_dict()
    return float(sp.sympify(s.strip().replace("^", "**"), locals=ld))


def _parse_points(xs_str: str, ys_str: str):
    """Parsea x e y desde strings de valores separados por comas."""
    xs = [_parse_val(v.strip()) for v in xs_str.split(",")]
    ys = [_parse_val(v.strip()) for v in ys_str.split(",")]
    if len(xs) != len(ys):
        raise ValueError("La cantidad de valores x e y no coincide.")
    if len(set(xs)) != len(xs):
        raise ValueError("Los valores de x deben ser distintos (nodos únicos).")
    return xs, ys


def _sym(v: float):
    """Convierte un float a expresión simbólica racional exacta."""
    return sp.nsimplify(v, rational=True)


def _build_lagrange(xs: list, ys: list):
    """
    Construye las bases de Lagrange y el polinomio interpolador P(x).

    Returns
    -------
    bases : list[dict]
        Información de cada L_i(x): expresión simbólica, partes LaTeX, etc.
    P_exp : sympy.Expr
        Polinomio interpolador expandido y simplificado.
    """
    x = sp.Symbol("x")
    n = len(xs)
    bases = []
    P_sym = sp.Integer(0)

    for i in range(n):
        xi, yi = xs[i], ys[i]

        # ── Expresión simbólica exacta ─────────────────────────────────────
        Li_num         = sp.Integer(1)
        den_val        = sp.Integer(1)
        den_factors_sym = []

        for j in range(n):
            if j != i:
                xj_s = _sym(xs[j])
                xi_s = _sym(xi)
                Li_num *= (x - xj_s)
                fac     = xi_s - xj_s
                den_factors_sym.append(fac)
                den_val *= fac

        Li_sym      = sp.expand(Li_num / den_val)
        contrib_sym = sp.expand(_sym(yi) * Li_sym)
        P_sym      += contrib_sym

        # ── Partes LaTeX para el desarrollo paso a paso ────────────────────
        num_parts, den_str_parts = [], []
        den_factor_vals = [float(f) for f in den_factors_sym]   # valores exactos

        for j in range(n):
            if j != i:
                xj    = xs[j]
                xj_s_ = _fmt(xj)
                xi_s_ = _fmt(xi)
                # Numerador: (x − xj)
                num_parts.append(
                    rf"(x - {xj_s_})" if xj >= 0 else rf"(x + {_fmt(-xj)})"
                )
                # Denominador: (xi − xj), con signo correcto
                den_str_parts.append(
                    rf"({xi_s_} - {xj_s_})" if xj >= 0 else rf"({xi_s_} + {_fmt(-xj)})"
                )

        bases.append({
            "i":               i,
            "xi":              xi,
            "yi":              yi,
            "Li_sym":          Li_sym,
            "Li_latex":        sp.latex(Li_sym),
            "num_parts":       num_parts,
            "den_str_parts":   den_str_parts,
            "den_factor_vals": den_factor_vals,
            "den_val_latex":   sp.latex(den_val),
            "contrib_sym":     contrib_sym,
            "contrib_latex":   sp.latex(contrib_sym),
        })

    P_exp = sp.expand(sp.simplify(P_sym))
    return bases, P_exp


# ── NUEVO: helper de formato ──────────────────────────────────────────────────
def _expr_to_latex(expr, mode: str, prec: int = 6) -> str:
    """
    Devuelve la representación LaTeX de expr según el modo de visualización.

    mode = 'Fracción' → forma exacta con fracciones simbólicas (sin cambios).
    mode = 'Decimal'  → coeficientes numéricos redondeados a `prec` decimales;
                        los enteros se mantienen como enteros (sin .0).
    """
    if mode == "Fracción":
        return sp.latex(expr)
    # Modo decimal: reemplaza solo los átomos no enteros con flotantes
    replacements = {}
    for atom in expr.atoms(sp.Number):
        if not isinstance(atom, sp.Integer):
            replacements[atom] = round(float(atom), prec)
    return sp.latex(expr.xreplace(replacements))


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
    </style>
    """, unsafe_allow_html=True)

    # ── Teoría (desplegable, arriba de todo) ───────────────────────────────────
    with st.expander("📘 Teoría: ¿Qué es el Polinomio Interpolador de Lagrange?", expanded=False):
        st.markdown(r"""
### ¿Qué es?

Dado un conjunto de $n+1$ puntos $(x_0, y_0), (x_1, y_1), \dots, (x_n, y_n)$ con nodos
$x_i$ todos distintos, el **polinomio interpolador de Lagrange** $P_n(x)$ es el
**único** polinomio de grado a lo sumo $n$ que pasa exactamente por todos esos puntos:

$$P_n(x_i) = y_i \qquad \text{para todo } i = 0, 1, \dots, n$$

La idea central es que, si los puntos provienen de evaluar una función $f(x)$ que no
conocemos del todo (o es difícil de calcular), $P_n(x)$ sirve como una **aproximación**
de $f(x)$ en todo el intervalo, construida solo a partir de esos pares de valores.
""")

        st.markdown("### ¿Cómo se construye, paso a paso?")
        st.markdown(r"""
**1. Definir las bases de Lagrange $L_i(x)$.**

Para cada nodo $x_i$ se construye un polinomio especial $L_i(x)$ que cumple una
propiedad muy particular: vale **1** exactamente en $x_i$, y **0** en todos los demás
nodos $x_j$ (con $j \neq i$):
""")
        st.latex(
            r"L_i(x) = \prod_{\substack{j\,=\,0 \\ j\,\neq\,i}}^{n}"
            r"\dfrac{x - x_j}{x_i - x_j}"
        )
        st.markdown(r"""
Esto se logra dividiendo el producto de $(x - x_j)$ para todo $j \neq i$ (que se anula
en cada $x_j$) por ese mismo producto evaluado en $x_i$ (una constante, que garantiza
que el resultado valga 1 justo en $x_i$).

**2. Combinar las bases con los valores $y_i$.**

Una vez que se tienen todas las $L_i(x)$, el polinomio interpolador se arma como una
suma ponderada: cada base $L_i(x)$ se multiplica por el valor $y_i$ que le corresponde,
y se suman todos los términos:
""")
        st.latex(r"P_n(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)")
        st.markdown(r"""
**3. ¿Por qué funciona?**

Al evaluar $P_n(x)$ en un nodo $x_k$ cualquiera, todas las bases $L_i(x_k)$ con
$i \neq k$ valen 0 (por construcción), y solo sobrevive el término $i = k$, donde
$L_k(x_k) = 1$. Entonces $P_n(x_k) = y_k$: el polinomio efectivamente pasa por todos
los puntos dados.

**4. Expandir y simplificar.**

Por último, se desarrolla la suma anterior y se agrupan términos semejantes para
obtener $P_n(x)$ en su forma polinómica habitual (potencias de $x$ con coeficientes),
que es la que se suele usar para evaluar o graficar.
""")

        st.markdown("### Fórmulas del error")
        st.markdown(r"""
Cuando los puntos provienen de una función conocida $f(x)$ que es derivable
$n+1$ veces, el polinomio $P_n(x)$ no coincide exactamente con $f(x)$ fuera de los
nodos: existe un **error de interpolación**. Existen dos formas de cuantificarlo.

**Error exacto (forma de Lagrange del resto).** Para cada $x$ fijo, existe algún
$\xi$ (dependiente de $x$) dentro del intervalo que contiene a los nodos, tal que:
""")
        st.latex(
            r"f(x) - P_n(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!}\,\prod_{i=0}^{n}(x - x_i)"
        )
        st.markdown(r"""
El problema práctico es que no se conoce $\xi$ exactamente, solo que existe. Por eso,
en la práctica se usa una **cota superior** del error, válida para cualquier punto del
intervalo:
""")
        st.latex(
            r"E_{m\acute{a}x} = \dfrac{M_1}{(n+1)!}\,M_2"
        )
        st.latex(
            r"M_1 = \max_{t\,\in\,[x_0,\,x_n]}\left|f^{(n+1)}(t)\right|"
        )
        st.latex(
            r"M_2 = \max_{t\,\in\,[x_0,\,x_n]}\left|\,\prod_{i=0}^{n}(t - x_i)\,\right|"
        )
        st.markdown(r"""
Acá $M_1$ es el mayor valor (en valor absoluto) que toma la derivada de orden $n+1$
de $f$ dentro del intervalo, y $M_2$ es el mayor valor (en valor absoluto) que toma el
producto $\prod (t - x_i)$ en ese mismo intervalo. Esta cota **no depende de un punto
$\xi$ particular**: es válida para todo el intervalo de interpolación.

**Error local (cuando se conoce $f$).** Si además se cuenta con la expresión exacta de
$f(x)$, el error real en un punto puntual $\xi$ se puede calcular directamente, sin
aproximaciones, comparando $f(\xi)$ con $P_n(\xi)$:
""")
        st.latex(r"E(\xi) = \bigl|f(\xi) - P_n(\xi)\bigr|")
        st.markdown(r"""
Este error local siempre debe ser menor o igual que la cota teórica $E_{m\acute{a}x}$
calculada arriba — si no lo es, suele deberse a que $M_1$ se estimó por muestreo
numérico y subestimó el verdadero máximo de la derivada.
""")

        st.markdown(r"""
### Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Pasa **exactamente** por todos los puntos dados | Con muchos nodos puede oscilar mucho entre ellos (fenómeno de Runge) |
| No requiere resolver sistemas de ecuaciones | Agregar un nuevo punto obliga a recalcular todo el polinomio desde cero |
| Fórmula cerrada, fácil de programar | Numéricamente puede ser inestable si los nodos están muy juntos o hay muchos |
""")

    st.title("Interpolación de Lagrange")

    # ── f(x) — opcional ───────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)  —  función original (opcional)",
        value="",
        placeholder="Ej: sin(x),  exp(x),  x**3 - 2*x + 1",
    )

    f_func = f_expr = latex_f = None
    if func_str.strip():
        try:
            f_func, f_expr, latex_f = _parse_f(func_str)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Puntos ────────────────────────────────────────────────────────────────
    xs_str = st.text_input(
        "Valores de x  (separados por comas)",
        value="0,1,2",
        placeholder="Ej: 0, 1, 2, 3  —  o también: 0, pi, 2*pi",
    )
    ys_str = st.text_input(
        "Valores de y  (separados por comas)",
        value="1,3,2",
        placeholder="Ej: 1, e**2, pi, -1",
    )

    xs = ys = None
    if xs_str.strip() and ys_str.strip():
        try:
            xs, ys = _parse_points(xs_str, ys_str)
            n_pts = len(xs)
            cols  = "c" * n_pts
            hdr_i = " & ".join(str(k)       for k in range(n_pts))
            hdr_x = " & ".join(_fmt(xs[k])  for k in range(n_pts))
            hdr_y = " & ".join(_fmt(ys[k])  for k in range(n_pts))
            st.latex(
                rf"\begin{{array}}{{c|{cols}}}"
                rf"i   & {hdr_i} \\ \hline "
                rf"x_i & {hdr_x} \\ "
                rf"y_i & {hdr_y}"
                rf"\end{{array}}"
            )
        except Exception as e:
            st.error(str(e))

    # ── ξ — punto de evaluación (opcional) ───────────────────────────────────
    xi_str = st.text_input(
        "ξ  —  punto de evaluación (opcional)",
        value="",
        placeholder="Ej: 1.5  —  o también: pi/2, e",
    )

    xi_val = None
    if xi_str.strip():
        try:
            xi_val = _parse_val(xi_str)          # ← soporta pi, e, e**2, etc.
            if xs is not None:
                lo, hi = min(xs), max(xs)
                if lo <= xi_val <= hi:
                    st.info(
                        f"ξ = {xi_val} ∈ [{_fmt(lo)}, {_fmt(hi)}]  —  "
                        "dentro del intervalo de interpolación ✓"
                    )
                else:
                    st.warning(
                        f"ξ = {xi_val} está fuera del intervalo de interpolación "
                        f"[{_fmt(lo)}, {_fmt(hi)}]. "
                        "La extrapolación puede ser imprecisa."
                    )
        except ValueError:
            st.error("Valor inválido para ξ.")

    # ── NUEVO: selector de formato para los resultados ────────────────────────
    col_fmt, _ = st.columns([1, 2])
    with col_fmt:
        mode = st.selectbox(
            "Formato de coeficientes en los resultados",
            options=["Fracción", "Decimal"],
            index=0,
            help=(
                "Fracción: coeficientes exactos como fracciones simbólicas.\n"
                "Decimal: coeficientes numéricos aproximados (6 cifras decimales). "
                "Útil cuando las fracciones resultan muy largas."
            ),
        )

    # ── Botón ─────────────────────────────────────────────────────────────────
    if st.button("Calcular polinomio", use_container_width=True):
        if xs is None or ys is None:
            st.error("Ingresá puntos válidos antes de calcular.")
            return
        if len(xs) < 2:
            st.warning("Necesitás al menos 2 puntos para obtener un polinomio no trivial.")
            return

        x_sym = sp.Symbol("x")
        bases, P_exp = _build_lagrange(xs, ys)
        n = len(xs)

        # Grado efectivo — calculado antes de paso 1 para usarlo en toda la notación P_n
        try:
            degree = int(sp.Poly(P_exp, x_sym).degree())
            if degree < 0:
                degree = 0
        except Exception:
            degree = 0

        # ══════════════════════════════════════════════════════════════════════
        # PASO 1 — Bases de Lagrange
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Paso 1 — Bases de Lagrange")
        st.markdown(
            "Cada polinomio base $L_i(x)$ vale **1** en el nodo $x_i$ "
            "y **0** en todos los demás nodos $x_j$ con $j \\neq i$:"
        )
        st.latex(
            r"L_i(x) \;=\; "
            r"\prod_{\substack{j\,=\,0 \\ j\,\neq\,i}}^{n-1}"
            r"\dfrac{x - x_j}{x_i - x_j}"
        )

        for b in bases:
            i, xi_v, yi_v = b["i"], b["xi"], b["yi"]

            st.markdown("---")
            st.markdown(
                f"#### $L_{{{i}}}(x)$ &ensp;—&ensp; "
                f"nodo $x_{{{i}}} = {_fmt(xi_v)}$,&ensp;$y_{{{i}}} = {_fmt(yi_v)}$"
            )

            # ── 1a. Fórmula general con índices concretos ──────────────────
            num_idx = r"\,\cdot\,".join(
                rf"(x - x_{{{j}}})" for j in range(n) if j != i
            )
            den_idx = r"\,\cdot\,".join(
                rf"(x_{{{i}}} - x_{{{j}}})" for j in range(n) if j != i
            )
            st.latex(rf"L_{{{i}}}(x) = \dfrac{{{num_idx}}}{{{den_idx}}}")

            # ── 1b. Sustitución de los valores numéricos ───────────────────
            num_sub = r"\,\cdot\,".join(b["num_parts"])
            den_sub = r"\,\cdot\,".join(b["den_str_parts"])
            st.latex(rf"L_{{{i}}}(x) = \dfrac{{{num_sub}}}{{{den_sub}}}")

            # ── 1c. Cálculo explícito del denominador ──────────────────────
            den_vals_str = r"\,\cdot\,".join(
                rf"({_fmt(fv)})" for fv in b["den_factor_vals"]
            )
            st.latex(
                rf"\text{{Denominador:}}\quad"
                rf"{den_sub} = {den_vals_str} = {b['den_val_latex']}"
            )

            # ── 1d. Numerador sobre el denominador ya calculado ────────────
            st.latex(
                rf"L_{{{i}}}(x) = \dfrac{{{num_sub}}}{{{b['den_val_latex']}}}"
            )

            # ── 1e. Resultado simplificado (boxed) — respeta modo ─────────
            st.latex(rf"\boxed{{\ L_{{{i}}}(x) = {_expr_to_latex(b['Li_sym'], mode)}\ }}")

            # ── 1f. Verificación: L_i(x_i) = 1 ───────────────────────────
            Li_at_xi = float(sp.N(b["Li_sym"].subs(x_sym, xi_v)))
            st.markdown(
                f"✔&ensp;Verificación:&ensp;"
                f"$L_{{{i}}}\\!\\left({_fmt(xi_v)}\\right) = {Li_at_xi:.6g}$"
            )

            # ── 1g. Contribución al polinomio — respeta modo ──────────────
            st.latex(
                rf"y_{{{i}}} \cdot L_{{{i}}}(x)"
                rf"= {_fmt(yi_v)} \cdot \left({_expr_to_latex(b['Li_sym'], mode)}\right)"
                rf"= {_expr_to_latex(b['contrib_sym'], mode)}"
            )

        # ══════════════════════════════════════════════════════════════════════
        # PASO 2 — Construcción de P_n(x)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown(f"### Paso 2 — Construcción del polinomio interpolador $P_{{{degree}}}(x)$")

        # Fórmula general
        sum_idx = " + ".join(
            rf"y_{{{i}}} \cdot L_{{{i}}}(x)" for i in range(n)
        )
        st.latex(
            rf"P_{{{degree}}}(x) = \sum_{{i=0}}^{{{n - 1}}} y_i \cdot L_i(x)"
            rf"\;=\; {sum_idx}"
        )

        # Con valores concretos — respeta modo
        sum_vals = r"\;+\;".join(
            rf"{_fmt(b['yi'])} \cdot \left({_expr_to_latex(b['Li_sym'], mode)}\right)"
            for b in bases
        )
        st.latex(rf"P_{{{degree}}}(x) = {sum_vals}")

        # Expandido — respeta modo
        P_latex = _expr_to_latex(P_exp, mode)
        # st.latex(rf"P_{{{degree}}}(x) = {P_latex}")

        st.latex(rf"\boxed{{\ P_{{{degree}}}(x) = {P_latex}\ }}")
        st.markdown(
            f"**Grado del polinomio:** {degree}"
            f"&ensp;(máximo esperado con {n} nodos: {n - 1})"
        )

        # Result cards
        st.markdown(f"""
        <div class="result-cards">
            <div class="result-card" style="border-top: 3px solid #2563eb;">
                <div class="rc-label">Nodos interpolados</div>
                <div class="rc-value">{n}</div>
                <div class="rc-sub">puntos</div>
            </div>
            <div class="result-card" style="border-top: 3px solid #16a34a;">
                <div class="rc-label">Grado efectivo</div>
                <div class="rc-value">{degree}</div>
                <div class="rc-sub">P(x)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # GRÁFICO
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Gráfico")

        try:
            # ── Rango en x: nodos + margen; se extiende si ξ queda fuera ──
            span_x = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
            pad    = max(span_x * 0.3, 0.5)
            x_lo   = min(xs) - pad
            x_hi   = max(xs) + pad
            if xi_val is not None:
                x_lo = min(x_lo, xi_val - pad * 0.4)
                x_hi = max(x_hi, xi_val + pad * 0.4)
            x_plot = np.linspace(x_lo, x_hi, 600)

            # ── Evaluar P(x) ───────────────────────────────────────────────
            P_plot_func = sp.lambdify(x_sym, P_exp, modules=["numpy"])
            y_P_raw = np.array(P_plot_func(x_plot), dtype=float)
            y_P_raw = np.where(np.isfinite(y_P_raw), y_P_raw, np.nan)

            # Rango y de referencia: percentiles del polinomio + nodos (siempre visibles)
            y_nodes = np.array(ys, dtype=float)
            finite_P = y_P_raw[np.isfinite(y_P_raw)]
            p5, p95 = (np.percentile(finite_P, [5, 95]) if len(finite_P) > 1
                       else (y_nodes.min(), y_nodes.max()))
            refs   = np.concatenate([y_nodes, [p5, p95]])
            y_rng  = max(float(refs.max() - refs.min()), 0.5)
            y_lo   = float(refs.min()) - y_rng * 0.2
            y_hi   = float(refs.max()) + y_rng * 0.2

            # Clip: valores que sobrepasan el doble del rango visible se ocultan
            y_P = np.where((y_P_raw >= y_lo - y_rng) & (y_P_raw <= y_hi + y_rng),
                           y_P_raw, np.nan)

            fig = go.Figure()

            # ── f(x) ───────────────────────────────────────────────────────
            if f_func is not None:
                try:
                    y_f_raw = np.array(f_func(x_plot), dtype=float)
                    y_f_raw = np.where(np.isfinite(y_f_raw), y_f_raw, np.nan)
                    # Ampliar rango y si f(x) lo requiere
                    f_finite = y_f_raw[np.isfinite(y_f_raw)]
                    if len(f_finite) > 1:
                        pf5, pf95 = np.percentile(f_finite, [5, 95])
                        refs2  = np.concatenate([refs, [pf5, pf95]])
                        y_rng  = max(float(refs2.max() - refs2.min()), 0.5)
                        y_lo   = float(refs2.min()) - y_rng * 0.2
                        y_hi   = float(refs2.max()) + y_rng * 0.2
                    y_f = np.where((y_f_raw >= y_lo - y_rng) & (y_f_raw <= y_hi + y_rng),
                                   y_f_raw, np.nan)
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_f,
                        mode="lines", name="f(x)",
                        line=dict(color="#2563eb", width=2, dash="dash"),
                    ))
                except Exception:
                    pass

            # ── P(x) ───────────────────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_P,
                mode="lines",
                name=f"P<sub>{degree}</sub>(x)",
                line=dict(color="#16a34a", width=2.5),
            ))

            # ── Nodos (xᵢ, yᵢ) ────────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                name="Nodos (xᵢ, yᵢ)",
                marker=dict(color="#dc2626", size=10,
                            line=dict(color="white", width=2)),
                text=[f"({_fmt(xv)}, {_fmt(yv)})" for xv, yv in zip(xs, ys)],
                textposition="top center",
                textfont=dict(size=11),
            ))

            # ── ξ: línea vertical + marcadores P(ξ) y f(ξ) ────────────────
            if xi_val is not None:
                fig.add_vline(
                    x=xi_val,
                    line=dict(color="rgba(120,120,120,0.6)", width=1.5, dash="dot"),
                    annotation_text=f"ξ = {_fmt(xi_val)}",
                    annotation_position="top right",
                    annotation_font=dict(size=12),
                )
                P_xi_graph = float(P_plot_func(xi_val))
                fig.add_trace(go.Scatter(
                    x=[xi_val], y=[P_xi_graph],
                    mode="markers",
                    name=f"P<sub>{degree}</sub>(ξ) = {P_xi_graph:.6g}",
                    marker=dict(color="#16a34a", size=13, symbol="diamond",
                                line=dict(color="white", width=2)),
                ))
                if f_func is not None:
                    try:
                        f_xi_graph = float(f_func(xi_val))
                        fig.add_trace(go.Scatter(
                            x=[xi_val], y=[f_xi_graph],
                            mode="markers",
                            name=f"f(ξ) = {f_xi_graph:.6g}",
                            marker=dict(color="#2563eb", size=13, symbol="diamond",
                                        line=dict(color="white", width=2)),
                        ))
                    except Exception:
                        pass

            # ── Layout ─────────────────────────────────────────────────────
            fig.update_layout(
                xaxis_title="x",
                yaxis_title="y",
                yaxis=dict(range=[y_lo, y_hi]),
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

        except Exception as _graph_err:
            st.warning(f"No se pudo generar el gráfico: {_graph_err}")

        # ══════════════════════════════════════════════════════════════════════
        # PASO 3 (opcional) — Evaluación en ξ
        # ══════════════════════════════════════════════════════════════════════
        if xi_val is not None:
            st.markdown("---")
            st.markdown(r"### Paso 3 — Evaluación en $\xi$")

            P_func  = sp.lambdify(x_sym, P_exp, modules=["numpy"])
            P_xi    = float(P_func(xi_val))
            xi_disp = _fmt(xi_val)

            st.latex(
                rf"P_{{{degree}}}(\xi) = P_{{{degree}}}\!\left({xi_disp}\right)"
                rf"= \left.{P_latex}\right|_{{x\,=\,{xi_disp}}}"
            )
            st.latex(rf"\boxed{{\ P_{{{degree}}}({xi_disp}) = {P_xi:.10g}\ }}")

            # ══════════════════════════════════════════════════════════════════
            # PASO 4 (opcional) — Cota teórica del error (error máximo)
            # ══════════════════════════════════════════════════════════════════
            cota_calculada = None
            if f_func is not None:
                st.markdown("---")
                st.markdown(r"### Paso 4 — Cota teórica del error (error máximo)")

                deriv_order = n  # con n nodos, la fórmula usa la derivada de orden n
                interval_lo, interval_hi = min(xs), max(xs)

                st.markdown(
                    f"Con **{n} nodos**, la fórmula del error de Lagrange utiliza "
                    f"la derivada de orden **{deriv_order}** de $f$. Esta es una cota "
                    "**teórica**, válida para cualquier punto del intervalo "
                    f"$[{_fmt(interval_lo)}, {_fmt(interval_hi)}]$ — no depende de ξ:"
                )
                st.latex(
                    rf"E_{{m\acute{{a}}x}} = \dfrac{{M_1}}{{{deriv_order}!}}\,M_2"
                )
                st.latex(
                    rf"M_1 = \max_{{t\,\in\,[{_fmt(interval_lo)},\,{_fmt(interval_hi)}]}}"
                    rf"\left|f^{{({deriv_order})}}(t)\right|"
                )
                st.latex(
                    rf"M_2 = \max_{{t\,\in\,[{_fmt(interval_lo)},\,{_fmt(interval_hi)}]}}"
                    rf"\left|g(t)\right|"
                )
                st.latex(
                    rf"g(x) = \prod_{{i=0}}^{{{n - 1}}}(x - x_i)"
                )

                try:
                    import math

                    # ── M1: máximo de |f^(n)(t)| en [min xi, max xi] ────────────
                    # Se obtiene analíticamente: raíces reales de f^(n+1)(t) = 0
                    # (puntos críticos de f^(n)) dentro del intervalo, más los
                    # extremos; se evalúa f^(n) en esos candidatos y se toma el máx.
                    f_deriv_expr  = sp.diff(f_expr, x_sym, deriv_order)
                    f_deriv2_expr = sp.diff(f_deriv_expr, x_sym)  # críticos de f^(n)
                    f_deriv_func  = sp.lambdify(x_sym, f_deriv_expr, modules=["numpy"])

                    m1_candidates = {interval_lo, interval_hi}
                    try:
                        crit_f = sp.solve(sp.Eq(f_deriv2_expr, 0), x_sym)
                        for cp in crit_f:
                            if cp.is_real:
                                cpf = float(cp)
                                if interval_lo - 1e-9 <= cpf <= interval_hi + 1e-9:
                                    m1_candidates.add(cpf)
                    except Exception:
                        pass  # si no resuelve simbólicamente, usamos extremos + muestreo

                    # Refuerzo numérico: muestreo denso por si f^(n) no es polinómica
                    # (ej. sin, exp) y sympy no resuelve f^(n+1)=0 en forma cerrada
                    t_samples = np.linspace(interval_lo, interval_hi, 2001)
                    with np.errstate(all="ignore"):
                        deriv_vals = np.array(f_deriv_func(t_samples), dtype=float)
                    finite_mask = np.isfinite(deriv_vals)

                    cand_vals = []
                    for c in m1_candidates:
                        try:
                            v = abs(float(f_deriv_func(c)))
                            if np.isfinite(v):
                                cand_vals.append((c, v))
                        except Exception:
                            pass
                    if deriv_vals[finite_mask].size > 0:
                        idx_max = np.argmax(np.abs(deriv_vals[finite_mask]))
                        t_at_max = t_samples[finite_mask][idx_max]
                        v_at_max = float(np.abs(deriv_vals[finite_mask][idx_max]))
                        cand_vals.append((t_at_max, v_at_max))

                    if not cand_vals:
                        raise ValueError(
                            f"f^({deriv_order}) no pudo evaluarse en el intervalo."
                        )

                    t1_star, M1 = max(cand_vals, key=lambda p: p[1])

                    # ── M2: máximo de |g(t)| en [min xi, max xi] (analítico) ────
                    g_sym = sp.Integer(1)
                    for xv in xs:
                        g_sym *= (x_sym - _sym(xv))
                    g_sym = sp.expand(g_sym)
                    g_prime = sp.diff(g_sym, x_sym)

                    m2_candidates = [sp.Float(interval_lo), sp.Float(interval_hi)]
                    g_poly = sp.Poly(g_prime, x_sym)
                    if g_poly.degree() >= 1:
                        for r in g_poly.real_roots():
                            rf_ = float(r)
                            if interval_lo - 1e-9 <= rf_ <= interval_hi + 1e-9:
                                m2_candidates.append(r)

                    g2_vals = [(c, abs(float(g_sym.subs(x_sym, c)))) for c in m2_candidates]
                    t2_star, M2 = max(g2_vals, key=lambda p: p[1])

                    fact = math.factorial(deriv_order)
                    cota = (M1 / fact) * M2
                    cota_calculada = cota

                    # ── Desarrollo visual ────────────────────────────────────────
                    st.latex(rf"f^{{({deriv_order})}}(x) = {sp.latex(f_deriv_expr)}")
                    st.latex(
                        rf"M_1 = \max_{{t\,\in\,[{_fmt(interval_lo)},\,{_fmt(interval_hi)}]}}"
                        rf"\left|f^{{({deriv_order})}}(t)\right|"
                        rf" = \left|f^{{({deriv_order})}}({_fmt(t1_star)})\right| \approx {M1:.6g}"
                    )

                    st.latex(rf"g(x) = {_expr_to_latex(g_sym, 'Decimal')}")
                    st.latex(rf"g'(x) = {_expr_to_latex(g_prime, 'Decimal')}")
                    st.markdown(
                        "Puntos críticos de $g$ dentro del intervalo "
                        f"$[{_fmt(interval_lo)}, {_fmt(interval_hi)}]$ "
                        "(raíces de $g'(x)=0$) y extremos del intervalo:"
                    )
                    cand_str = r",\ ".join(_fmt(c) for c, _ in g2_vals)
                    st.latex(rf"t \in \{{{cand_str}\}}")

                    # ── Evaluación de g en cada candidato + cómo se elige el máximo ──
                    eval_lines = r" \\[4pt] ".join(
                        rf"|g({_fmt(c)})| = {v:.6g}" for c, v in g2_vals
                    )
                    st.latex(rf"\begin{{aligned}} {eval_lines} \end{{aligned}}")
                    st.markdown(
                        f"El mayor valor en valor absoluto se da en $t = {_fmt(t2_star)}$, "
                        f"por lo tanto:"
                    )
                    st.latex(
                        rf"M_2 = \max\left|g(t)\right|"
                        rf" = \left|g({_fmt(t2_star)})\right| \approx {M2:.6g}"
                    )

                    st.latex(
                        rf"E_{{m\acute{{a}}x}} = \dfrac{{M_1}}{{{deriv_order}!}}\,M_2"
                        rf"= \dfrac{{{M1:.6g}}}{{{fact}}} \cdot {M2:.6g}"
                    )

                    cota_plain = f"{cota:.10g}"
                    st.latex(rf"\boxed{{\ E_{{m\acute{{a}}x}} \approx {cota_plain}\ }}")

                    st.markdown(f"""
                    <div class="result-cards">
                        <div class="result-card" style="border-top: 3px solid #f59e0b;">
                            <div class="rc-label">Cota teórica del error</div>
                            <div class="rc-value">{cota_plain}</div>
                            <div class="rc-sub">M₁·M₂ / {deriv_order}!</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption(
                        "Cota válida para todo el intervalo (no depende de ξ): "
                        "M₁ es el máximo de |f⁽ⁿ⁾(t)| y M₂ es el máximo de |g(t)| = "
                        "|∏(t−xᵢ)|, ambos hallados analíticamente sobre "
                        f"[{_fmt(interval_lo)}, {_fmt(interval_hi)}]."
                    )

                except Exception as e:
                    st.error(f"No se pudo calcular la cota teórica del error: {e}")

            # ══════════════════════════════════════════════════════════════════
            # PASO 5 (opcional) — Error local en ξ
            # ══════════════════════════════════════════════════════════════════
            if f_func is not None:
                st.markdown("---")
                st.markdown(r"### Paso 5 — Error local en $\xi$")

                try:
                    f_xi  = float(f_func(xi_val))
                    error = abs(f_xi - P_xi)

                    # Formato sin notación científica: decimales adaptativos, sin ceros finales
                    import math
                    _dec = max(4, -int(math.floor(math.log10(error))) + 4) if error > 0 else 4
                    error_plain = f"{error:.{_dec}f}".rstrip("0").rstrip(".") if error > 0 else "0"

                    st.latex(
                        rf"f(\xi) = f\!\left({xi_disp}\right) = {f_xi:.10g}"
                    )
                    st.latex(
                        rf"P_{{{degree}}}(\xi) = P_{{{degree}}}\!\left({xi_disp}\right) = {P_xi:.10g}"
                    )
                    st.latex(
                        rf"\bigl|f(\xi) - P_{{{degree}}}(\xi)\bigr|"
                        rf"= \bigl|{f_xi:.10g} - {P_xi:.10g}\bigr|"
                        rf"= {error_plain}"
                    )
                    st.latex(
                        rf"\boxed{{\ E(\xi)"
                        rf"= \bigl|f(\xi) - P_{{{degree}}}(\xi)\bigr|"
                        rf"= {error_plain}\ }}"
                    )

                    st.markdown(f"""
                    <div class="result-cards">
                        <div class="result-card" style="border-top: 3px solid #dc2626;">
                            <div class="rc-label">Error en <span style="text-transform:none">ξ</span> = {xi_disp}</div>
                            <div class="rc-value">{error_plain}</div>
                            <div class="rc-sub">|f(ξ) − P(ξ)|</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ══════════════════════════════════════════════════════════
                    # Verificación — error local vs. cota teórica
                    # ══════════════════════════════════════════════════════════
                    if cota_calculada is not None:
                        st.markdown("---")
                        st.markdown("### Verificación — error local vs. cota teórica")

                        cota_disp = f"{cota_calculada:.10g}"
                        cumple    = error <= cota_calculada + 1e-9

                        if cumple:
                            st.success(
                                rf"✔ Se cumple la cota: "
                                rf"$|f(\xi) - P_{{{degree}}}(\xi)| = {error_plain} "
                                rf"\;\leq\; E_{{m\acute{{a}}x}} \approx {cota_disp}$"
                            )
                        else:
                            st.warning(
                                rf"⚠ El error local **supera** la cota teórica calculada: "
                                rf"$|f(\xi) - P_{{{degree}}}(\xi)| = {error_plain} "
                                rf"\;>\; E_{{m\acute{{a}}x}} \approx {cota_disp}$. "
                                rf"Esto suele deberse a que $M$ se estimó por muestreo "
                                rf"numérico de $f^{{({n})}}(t)$ y puede subestimar el "
                                rf"verdadero máximo de la derivada."
                            )

                except Exception as e:
                    st.error(f"No se pudo evaluar f(ξ): {e}")