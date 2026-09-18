import re
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_func(*args):
    """Interpreta log:
    - log(x)       -> logaritmo decimal o en base 10
    - log(x, base) -> logaritmo en la base indicada
    """
    if len(args) == 1:
        return sp.log(args[0]) / sp.log(10)
    return sp.log(args[0]) / sp.log(args[1])


def _local_dict(x_sym=None):
    if x_sym is None:
        x_sym = sp.Symbol("x")
    return x_sym, {
        "x":     x_sym,
        "e":     sp.E,
        "E":     sp.E,
        "pi":    sp.pi,
        "ln":    sp.log,       # ln(x) -> logaritmo natural (base e)
        "log":   _log_func,    # log(x) -> logaritmo decimal (base 10)
        "log10": lambda arg: sp.log(arg) / sp.log(10),
        "log2":  lambda arg: sp.log(arg) / sp.log(2),
        "exp":   sp.exp,
        "sin":   sp.sin,
        "cos":   sp.cos,
        "tan":   sp.tan,
        "sqrt":  sp.sqrt,
        "Abs":   sp.Abs,
        "abs":   sp.Abs,
    }


def _to_latex(expr) -> str:
    r"""Convierte una expresión SymPy a LaTeX diferenciando claramente entre:
    - ln(...)  -> logaritmo natural (\ln)
    - log(...) -> logaritmo en base 10 (\log_{10}) o en la base especificada (\log_{b})
    """
    if expr is None:
        return ""
    try:
        tex = sp.latex(expr, ln_notation=True)

        def _repl_log(m):
            prefix = m.group(1) or ""
            arg = m.group(2) or m.group(3)
            base = m.group(4) or m.group(5)
            if base == "10":
                return rf"{prefix}\log_{{10}}\left({arg}\right)"
            return rf"{prefix}\log_{{{base}}}\left({arg}\right)"

        tex = re.sub(
            r"\\frac\{(.*?)\\ln(?:\{\\left\((.*?)\\right\)\}|\{(.*?)\})\}\{\\ln(?:\{\\left\(([0-9]+)\s*\\right\)\}|\{([0-9]+)\s*\})\}",
            _repl_log,
            tex,
        )
        return tex
    except Exception:
        return sp.latex(expr) if hasattr(expr, "free_symbols") else str(expr)


def _parse_f(func_str: str):
    """Devuelve (callable, latex_str) o lanza excepción."""
    x, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f = sp.lambdify(x, expr, modules=["numpy"])
    return f, _to_latex(expr)


def _parse_g(func_str: str):
    """Parsea g(x) y su derivada.
    Devuelve (callable_g, callable_gp, latex_g, latex_gp) o lanza excepción."""
    x, ld = _local_dict()
    expr     = sp.sympify(func_str.replace("^", "**"), locals=ld)
    expr_der = sp.diff(expr, x)
    g  = sp.lambdify(x, expr,     modules=["numpy"])
    gp = sp.lambdify(x, expr_der, modules=["numpy"])
    return g, gp, _to_latex(expr), _to_latex(expr_der)


def _fixed_point(g, x0: float, tol: float, max_iter: int):
    rows = []
    xn = x0
    for i in range(max_iter):
        xn1 = float(g(xn))
        rows.append({"n": i, "xn": xn, "xn1": xn1})
        if abs(xn1 - xn) <= tol:
            return xn1, rows, True
        xn = xn1
    return xn, rows, False


def _cobweb_path(x0: float, rows):
    """Construye la trayectoria en 'escalera / telaraña' a partir de x0 y las
    filas de la iteración. Devuelve (xs, ys) listos para graficar con un
    único trazo de líneas (alterna tramos verticales y horizontales entre
    la curva g(x) y la recta y = x)."""
    xs, ys = [x0], [x0]
    for row in rows:
        xn, xn1 = row["xn"], row["xn1"]
        # vertical: de (xn, xn) en la diagonal hasta (xn, g(xn)) en la curva
        xs.append(xn)
        ys.append(xn1)
        # horizontal: de (xn, g(xn)) hasta (g(xn), g(xn)) en la diagonal
        xs.append(xn1)
        ys.append(xn1)
    return xs, ys


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
    with st.expander("📘 Teoría: ¿Cómo funciona el Método de Punto Fijo?", expanded=False):
        st.markdown(r"""
### ¿Qué es?

El **método de punto fijo** (o iteración de punto fijo) es un método numérico para
encontrar raíces de $f(x) = 0$. La idea central es **reescribir la ecuación** en la forma

$$x = g(x)$$

despejando $x$ de alguna manera algebraica a partir de $f(x)=0$. Un valor $x^{*}$ que cumple
$x^{*} = g(x^{*})$ se llama **punto fijo** de $g$, y es exactamente una raíz de la $f$ original.

A partir de un valor inicial $x_0$, se genera una secuencia aplicando $g$ una y otra vez:
""")
        st.latex(r"x_{n+1} = g(x_n)")
        st.markdown(r"""
Si la secuencia converge, lo hace hacia un punto fijo de $g$ — es decir, hacia una raíz de $f$.
""")

        st.markdown("### Paso a paso del algoritmo")
        st.markdown(r"""
1. **Despejar $x$ de $f(x) = 0$** para obtener una función de iteración $g(x)$ tal que $x = g(x)$.
   *(Una misma $f$ puede despejarse de varias formas distintas, y no todas convergen igual — más abajo se explica por qué).*
2. **Elegir un valor inicial $x_0$**, idealmente cercano a la raíz que se busca.
3. **Calcular el siguiente término:**
""")
        st.latex(r"x_{n+1} = g(x_n)")
        st.markdown(r"""
4. **Medir el error entre dos iteraciones consecutivas:**
""")
        st.latex(r"|x_{n+1} - x_n|")
        st.markdown(r"""
5. **Verificar el criterio de parada.** Si $|x_{n+1} - x_n| \le \varepsilon$ (tolerancia),
   se detiene: $x_{n+1}$ es la raíz aproximada.
6. **Si no, repetir** desde el paso 3 usando $x_{n+1}$ como nuevo punto de partida,
   hasta cumplir el criterio de parada o alcanzar el máximo de iteraciones.

### Interpretación gráfica: ¿qué representa la recta $y = x$?

Buscar un punto fijo significa buscar un $x^{*}$ tal que $x^{*} = g(x^{*})$. Geométricamente,
eso equivale a buscar dónde la curva $y = g(x)$ se cruza con la **recta $y = x$** — la diagonal
donde, por definición, la coordenada $y$ de cualquier punto es igual a su coordenada $x$. Por
eso, exactamente en el punto donde $g(x)$ corta esa recta se cumple $g(x) = x$: ahí está la
raíz buscada.

Esa misma recta es también la que permite "trasladar" un resultado de vuelta al eje $x$ para
poder iterar de nuevo, y es la base del clásico **diagrama de telaraña** (*cobweb diagram*)
con el que se suele visualizar este método:

1. Se parte de $x_0$ sobre la recta $y=x$ y se traza una línea **vertical** hasta tocar la
   curva $g(x)$: ese punto tiene altura $g(x_0) = x_1$.
2. Desde ahí se traza una línea **horizontal** hasta volver a tocar la recta $y=x$: al llegar,
   la coordenada $x$ de ese punto ya es $x_1$, lista para repetir el paso 1.
3. Se repite el proceso (vertical hasta $g(x)$, horizontal hasta $y=x$) una y otra vez.

El resultado es una sucesión de escalones que, si el método converge, se va cerrando como una
telaraña hacia el punto donde $g(x)$ cruza $y=x$ (la raíz). Si diverge, la telaraña se aleja
de ese cruce en lugar de acercarse — algo que se puede ver directamente en el gráfico de esta
calculadora.

### ¿Cuándo converge? — Criterio de Lipschitz / contracción

No cualquier $g(x)$ funciona: la convergencia depende de qué tan "expansiva" o "contractiva"
es $g$ cerca de la raíz. La condición clásica (ligada al **Teorema del Punto Fijo de Banach**)
es:
""")
        st.latex(r"|g'(x)| < 1")
        st.markdown(r"""
en un entorno de la raíz buscada. Cuando se cumple, $g$ es una **contracción** y la iteración
converge sí o sí desde cualquier $x_0$ suficientemente cercano.

En esta calculadora se evalúa $g'(x_0)$ como una **verificación rápida** (heurística) antes de
iterar. Hay que tener en cuenta que, en rigor, la condición debería cumplirse en todo un entorno
de la raíz (que todavía no conocemos) y no solo en el punto de partida — por eso puede pasar que
el chequeo en $x_0$ dé bien y aun así la iteración no converja, o viceversa. Es una guía útil,
no una garantía absoluta.

### Velocidad de convergencia

- Es, en general, **convergencia lineal**: el error se reduce en un factor aproximadamente
  constante ($\approx |g'(x^{*})|$) en cada paso.
- Cuanto más chico sea $|g'(x^{*})|$, más rápido converge. Si $g'(x^{*}) = 0$, puede llegar a
  converger más rápido que linealmente.

### Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| No necesita un intervalo con cambio de signo (a diferencia de bisección) | La convergencia **no está garantizada**: depende totalmente de cómo se elija $g(x)$ |
| Simple de implementar y de entender | Puede divergir u oscilar si $\lvert g'(x)\rvert \ge 1$ |
| Puede converger rápido si $g$ está bien elegida | Es sensible a la elección de $x_0$ |
""")

    st.title("Método de Punto Fijo")

    # ── f(x) ─────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)",
        value="x**3 - x - 2",
        placeholder="Ej: x**2 - 4,  sin(x) - x/2,  ln(x) - 1,  log(x)",
        help="ln(x) = logaritmo natural (base e) · log(x) = logaritmo decimal (base 10) · log(x, b) = base b",
    )

    f = None
    if func_str:
        try:
            f, latex_f = _parse_f(func_str)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── g(x) ─────────────────────────────────────────────────────────────────
    g_str = st.text_input(
        "g(x) — función de iteración  [despejada como x = g(x)]",
        value="(x + 2)**(1/3)",
        placeholder="Ej: (x + 2)**(1/3),  cos(x),  ln(x + 2),  log(x)",
        help="ln(x) = logaritmo natural (base e) · log(x) = logaritmo decimal (base 10) · log(x, b) = base b",
    )

    g = gp = None
    if g_str:
        try:
            g, gp, latex_g, latex_gp = _parse_g(g_str)
            st.latex(rf"g(x) = {latex_g}")
            st.latex(rf"g'(x) = {latex_gp}")
        except Exception as e:
            st.error(f"No se pudo interpretar g(x): {e}")

    # ── x₀ ───────────────────────────────────────────────────────────────────
    x0_str = st.text_input("x₀  (valor inicial)", value="1.5",
                           placeholder="Ej: 1.5 · pi · pi/4 · e**2 · sqrt(2)")
    x0 = None
    try:
        _, ld = _local_dict()
        x0 = float(sp.sympify(x0_str.replace("^", "**"), locals=ld))
    except Exception:
        st.error("Valor inválido para x₀. Ejemplos: 1.5 · pi · pi/4 · e**2 · sqrt(2)")

    # ── Criterio de Lipschitz  |g'(x₀)| < 1 ─────────────────────────────────
    lipschitz_ok = False
    if gp is not None and x0 is not None:
        try:
            gp_val = float(gp(x0))
            abs_gp = abs(gp_val)
            sign   = r"<" if abs_gp < 1 else r"\geq"
            lip_expr = (
                rf"$|g'(x_0)| = |g'({x0})| = |{gp_val:.6g}|"
                rf" = {abs_gp:.6g} \; {sign} \; 1$"
            )
            lipschitz_ok = abs_gp < 1
            if lipschitz_ok:
                st.success(f"**Criterio de Lipschitz se cumple:** {lip_expr}")
            else:
                st.error(f"**Criterio de Lipschitz no se cumple:** {lip_expr}")
        except Exception as e:
            st.error(f"Error al evaluar g'(x₀): {e}")

    # ── Parámetros ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    tol_str = c1.text_input("Tolerancia (ε)", value="0.000001")
    try:
        tol = float(tol_str)
    except ValueError:
        tol = None
        c1.error("Ingresá un número válido, ej: 0.0001")
    max_iter_str = c2.text_input("Máx. iteraciones", value="50")
    try:
        max_iter = int(max_iter_str)
    except ValueError:
        max_iter = None
        c2.error("Valor inválido")
    decimals_str = c3.text_input("Decimales en tabla", value="6")
    try:
        decimals = int(decimals_str)
    except ValueError:
        decimals = None
        c3.error("Valor inválido")

    # ── Botón ─────────────────────────────────────────────────────────────────
    if st.button("Calcular raíz", use_container_width=True):
        if not g:
            st.error("Ingresá una función g(x) válida antes de calcular.")
        elif x0 is None:
            st.error("Ingresá un valor válido para x₀.")
        elif tol is None:
            st.error("Ingresá una tolerancia válida.")
        elif max_iter is None or decimals is None:
            st.error("Ingresá valores válidos para iteraciones y decimales.")
        elif max_iter < 1:
            st.error("El número de iteraciones debe ser al menos 1.")
        elif decimals < 0:
            st.error("Los decimales deben ser un número entero ≥ 0.")
        elif not lipschitz_ok:
            st.warning(
                "El criterio de Lipschitz no se cumple: "
                "la convergencia no está garantizada en x₀."
            )
        else:
            root, rows, converged = _fixed_point(g, x0, tol, max_iter)
            n = len(rows)

            if converged:
                st.success(f"**Convergencia alcanzada en {n} iteraciones**")
            else:
                st.warning(
                    f"**Sin convergencia** tras {n} iteraciones. "
                    "Mostrando mejor aproximación."
                )

            st.markdown(f"""
            <div class="result-cards">
                <div class="result-card" style="border-top: 3px solid #2563eb;">
                    <div class="rc-label">Raíz hallada</div>
                    <div class="rc-value">{root:.{decimals}f}</div>
                    <div class="rc-sub">x *</div>
                </div>
                <div class="result-card" style="border-top: 3px solid #16a34a;">
                    <div class="rc-label">Iteraciones</div>
                    <div class="rc-value">{n}</div>
                    <div class="rc-sub">n</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            d = decimals
            lines = [
                r"| $n$ | $x_n$ | $x_{n+1} = g(x_n)$ |",
                "|:---:|:---:|:---:|",
            ]
            for row in rows:
                lines.append(
                    f"| {row['n']} "
                    f"| {row['xn']:.{d}f} "
                    f"| {row['xn1']:.{d}f} |"
                )
            st.markdown("\n".join(lines))

            # ══════════════════════════════════════════════════════════════════
            # GRÁFICO — diagrama de telaraña: g(x), y = x, trayectoria y x₀
            # ══════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### Gráfico (diagrama de telaraña)")

            try:
                # ── Rango en x: en base a x0 y toda la trayectoria recorrida ──
                traj_xs = [x0] + [row["xn1"] for row in rows]
                span_x  = max(traj_xs) - min(traj_xs) if max(traj_xs) != min(traj_xs) else 1.0
                pad     = max(span_x * 0.6, 1.0)
                x_lo    = min(traj_xs) - pad
                x_hi    = max(traj_xs) + pad
                x_plot  = np.linspace(x_lo, x_hi, 600)

                # ── Evaluar g(x) en el rango ──────────────────────────────
                y_g_raw = np.array(g(x_plot), dtype=float)
                y_g_raw = np.where(np.isfinite(y_g_raw), y_g_raw, np.nan)

                # Rango y de referencia: percentiles de g(x) + la trayectoria
                finite_g = y_g_raw[np.isfinite(y_g_raw)]
                p5, p95  = (np.percentile(finite_g, [5, 95]) if len(finite_g) > 1
                            else (x_lo, x_hi))
                refs  = np.array(traj_xs + [float(p5), float(p95), x_lo, x_hi])
                y_rng = max(float(refs.max() - refs.min()), 0.5)
                y_lo  = float(refs.min()) - y_rng * 0.15
                y_hi  = float(refs.max()) + y_rng * 0.15

                # Clip: valores que sobrepasan el rango visible se ocultan
                y_g = np.where((y_g_raw >= y_lo - y_rng) & (y_g_raw <= y_hi + y_rng),
                               y_g_raw, np.nan)

                fig = go.Figure()

                # ── Recta y = x ─────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=[x_lo, x_hi], y=[x_lo, x_hi],
                    mode="lines", name="y = x",
                    line=dict(color="#16a34a", width=2, dash="dash"),
                ))

                # ── g(x) ────────────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=x_plot, y=y_g,
                    mode="lines", name="g(x)",
                    line=dict(color="#2563eb", width=2.5),
                ))

                # ── Trayectoria (telaraña) ───────────────────────────────
                cob_x, cob_y = _cobweb_path(x0, rows)
                fig.add_trace(go.Scatter(
                    x=cob_x, y=cob_y,
                    mode="lines",
                    name="Trayectoria",
                    line=dict(color="#f97316", width=1.8),
                ))

                # ── x₀ ────────────────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=[x0], y=[x0],
                    mode="markers+text",
                    name=f"x₀ = {x0:.{d}f}",
                    marker=dict(color="#9333ea", size=11, symbol="circle",
                                line=dict(color="white", width=2)),
                    text=[f"x₀ = {x0:.{d}f}"],
                    textposition="bottom center",
                    textfont=dict(size=11),
                ))

                # ── Raíz / punto fijo hallado ─────────────────────────────
                fig.add_vline(
                    x=root,
                    line=dict(color="rgba(120,120,120,0.6)", width=1.5, dash="dot"),
                )
                fig.add_trace(go.Scatter(
                    x=[root], y=[root],
                    mode="markers",
                    name=f"x* = {root:.{d}f}",
                    marker=dict(color="#dc2626", size=13, symbol="diamond",
                                line=dict(color="white", width=2)),
                ))

                # ── Layout ────────────────────────────────────────────────
                fig.update_layout(
                    xaxis_title="x",
                    yaxis_title="g(x)",
                    yaxis=dict(range=[y_lo, y_hi]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                    margin=dict(l=50, r=20, t=50, b=50),
                    hovermode="closest",
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