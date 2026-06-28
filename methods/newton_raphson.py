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
    """Parsea f(x) y calcula f'(x) automáticamente.
    Devuelve (callable_f, callable_fp, latex_f, latex_fp) o lanza excepción."""
    x, ld = _local_dict()
    expr     = sp.sympify(func_str.replace("^", "**"), locals=ld)
    expr_der = sp.diff(expr, x)
    f  = sp.lambdify(x, expr,     modules=["numpy"])
    fp = sp.lambdify(x, expr_der, modules=["numpy"])
    return f, fp, sp.latex(expr), sp.latex(expr_der)


def _newton(f, fp, x0: float, tol: float, max_iter: int):
    """Iteración de Newton-Raphson:

       xₙ₊₁ = xₙ − f(xₙ) / f'(xₙ)

       Criterio de parada: |xₙ₊₁ − xₙ| ≤ tol.
       Si f'(xₙ) ≈ 0 en alguna iteración se aborta (denominador nulo).
    """
    rows = []
    xn = x0
    for i in range(max_iter):
        fxn  = float(f(xn))
        fpxn = float(fp(xn))
        if abs(fpxn) < 1e-15:
            rows.append({"n": i, "xn": xn, "fxn": fxn, "fpxn": fpxn, "xn1": None})
            return xn, rows, False
        xn1 = xn - fxn / fpxn
        rows.append({"n": i, "xn": xn, "fxn": fxn, "fpxn": fpxn, "xn1": xn1})
        if abs(xn1 - xn) <= tol:
            return xn1, rows, True
        xn = xn1
    return xn, rows, False


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
    with st.expander("📘 Teoría: ¿Cómo funciona el Método de Newton-Raphson?", expanded=False):
        st.markdown(r"""
### ¿Qué es?

El **método de Newton-Raphson** busca raíces de $f(x) = 0$ usando la **recta tangente**
a la curva en cada punto. Es, en general, el método más rápido de todos los que vimos
hasta ahora (bisección, punto fijo, Aitken) — pero a cambio necesita poder calcular $f'(x)$.

### La idea geométrica

Parados en un punto $x_n$:

1. Trazamos la **recta tangente** a $f$ en $(x_n,\, f(x_n))$. Su pendiente es $f'(x_n)$.
2. Esa recta corta al eje $x$ en algún punto — lo llamamos $x_{n+1}$.
3. Como la tangente "sigue" a la curva, $x_{n+1}$ suele estar más cerca de la raíz real que $x_n$.
4. Repetimos el proceso desde $x_{n+1}$.

*(En el gráfico que se genera después de calcular, podés ver justamente esto: el punto de
arranque $x_0$ sobre la curva, y la raíz final marcada sobre el eje $x$.)*

### De la geometría a la fórmula

La recta tangente en $x_n$ es:
""")
        st.latex(r"y - f(x_n) = f'(x_n)\,(x - x_n)")
        st.markdown(r"Buscamos dónde esa recta cruza el eje $x$ (es decir, $y=0$):")
        st.latex(r"0 - f(x_n) = f'(x_n)\,(x_{n+1} - x_n) \;\;\Longrightarrow\;\; x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")

        st.markdown("### Paso a paso del algoritmo")
        st.markdown(r"""
1. **Definir $f(x)$** (esta calculadora deriva $f'(x)$ automáticamente).
2. **Elegir un valor inicial $x_0$**, idealmente cerca de la raíz buscada y con $f'(x_0) \neq 0$.
3. **Calcular el siguiente punto:**
""")
        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
        st.markdown(r"""
4. **Verificar el criterio de parada:** si $|x_{n+1} - x_n| \le \varepsilon$, $x_{n+1}$ es la raíz aproximada.
5. **Si no, repetir** desde el paso 3 usando $x_{n+1}$ como nuevo $x_n$, hasta convergencia o
   máximo de iteraciones.
6. **Si en algún paso $f'(x_n) \approx 0$**, la tangente queda horizontal y no corta al eje $x$
   en ningún punto razonable — el método se detiene ahí (esta calculadora lo detecta y lo avisa).

### Velocidad de convergencia

Cerca de una raíz **simple** (donde $f'(x^{*}) \neq 0$), Newton-Raphson converge
**cuadráticamente**: la cantidad de cifras correctas aproximadamente se **duplica** en cada
iteración. Por eso, para $f(x) = x^3 - x - 2$ con $x_0 = 1.5$, alcanza una tolerancia de
$10^{-12}$ en solo **4 iteraciones** — mucho menos que bisección, punto fijo o incluso Aitken.

Si la raíz es **múltiple** (es decir, $f'(x^{*}) = 0$ también), la convergencia se degrada y
vuelve a ser solo lineal, como en los métodos anteriores.

### Cuándo puede fallar

- **$f'(x_n) \approx 0$ en algún paso:** la tangente queda (casi) horizontal — no hay buen
  siguiente punto. El método se detiene.
- **$x_0$ mal elegido:** lejos de la raíz, Newton puede divergir, oscilar entre dos valores,
  o "saltar" hacia una raíz distinta de la que se buscaba.
- **Funciones con curvatura fuerte o varias raíces cercanas:** el resultado puede ser sensible
  al punto de partida.

### Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Convergencia muy rápida (cuadrática) cerca de una raíz simple | Necesita poder calcular $f'(x)$ |
| No necesita un intervalo con cambio de signo, como bisección | Puede diverger si $x_0$ está lejos de la raíz o si $f'(x)$ es chica |
| Suele necesitar muy pocas iteraciones en la práctica | Pierde velocidad en raíces múltiples ($f'(x^{*})=0$) |
""")

    st.title("Método de Newton-Raphson")

    # ── f(x)  y  f'(x) calculada automáticamente ─────────────────────────────
    func_str = st.text_input(
        "f(x)",
        value="x**3 - x - 2",
        placeholder="Ej: x**2 - 4,  sin(x) - x/2,  exp(x) - 3",
    )

    f = fp = None
    if func_str:
        try:
            f, fp, latex_f, latex_fp = _parse_f(func_str)
            st.latex(rf"f(x)  = {latex_f}")
            st.latex(rf"f'(x) = {latex_fp}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── x₀ ───────────────────────────────────────────────────────────────────
    x0_str = st.text_input("x₀  (valor inicial)", value="1.5")
    x0 = None
    try:
        x0 = float(x0_str)
    except ValueError:
        st.error("Valor inválido para x₀")

    # ── Verificación f'(x₀) ≠ 0 ──────────────────────────────────────────────
    deriv_ok = False
    if fp is not None and x0 is not None:
        try:
            fp_val = float(fp(x0))
            abs_fp = abs(fp_val)
            deriv_ok = abs_fp > 1e-10
            sign     = r"\neq" if deriv_ok else r"="
            deriv_expr = (
                rf"$f'(x_0) = f'({x0}) = {fp_val:.6g} \; {sign} \; 0$"
            )
            if deriv_ok:
                st.success(f"**f'(x₀) es válida:** {deriv_expr}")
            else:
                st.error(
                    f"**f'(x₀) = 0:** {deriv_expr} — "
                    "el método no puede arrancar desde este punto."
                )
        except Exception as e:
            st.error(f"Error al evaluar f'(x₀): {e}")

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
        if not f:
            st.error("Ingresá una función válida antes de calcular.")
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
        elif not deriv_ok:
            st.warning(
                "f'(x₀) = 0: elegí un x₀ distinto donde la derivada no se anule."
            )
        else:
            root, rows, converged = _newton(f, fp, x0, tol, max_iter)
            n = len(rows)

            # ── Mensaje de estado ─────────────────────────────────────────────
            last = rows[-1]
            if last["xn1"] is None:
                st.error(
                    f"f'(xₙ) = 0 en la iteración {last['n']}: "
                    "denominador nulo, el método se detuvo."
                )
            elif converged:
                st.success(f"**Convergencia alcanzada en {n} iteraciones**")
            else:
                st.warning(
                    f"**Sin convergencia** tras {n} iteraciones. "
                    "Mostrando mejor aproximación."
                )

            # ── Tarjetas ──────────────────────────────────────────────────────
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

            # ── Tabla ─────────────────────────────────────────────────────────
            d = decimals
            lines = [
                r"| $n$ | $x_n$ | $f(x_n)$ | $f'(x_n)$ |"
                r" $x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}$ |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ]
            for row in rows:
                xn1_str = (
                    f"{row['xn1']:.{d}f}"
                    if row["xn1"] is not None
                    else "—"
                )
                lines.append(
                    f"| {row['n']} "
                    f"| {row['xn']:.{d}f} "
                    f"| {row['fxn']:.{d}f} "
                    f"| {row['fpxn']:.{d}f} "
                    f"| {xn1_str} |"
                )
            st.markdown("\n".join(lines))

            # ══════════════════════════════════════════════════════════════════
            # GRÁFICO — f(x) con la raíz hallada
            # ══════════════════════════════════════════════════════════════════
            if converged or last["xn1"] is not None:
                st.markdown("---")
                st.markdown("### Gráfico")

                try:
                    # ── Rango en x: alrededor de x₀ y la raíz, con margen ──
                    pts_x  = [x0, root]
                    span_x = max(pts_x) - min(pts_x) if max(pts_x) != min(pts_x) else 1.0
                    pad    = max(span_x * 0.5, 1.0)
                    x_lo   = min(pts_x) - pad
                    x_hi   = max(pts_x) + pad
                    x_plot = np.linspace(x_lo, x_hi, 600)

                    # ── Evaluar f(x) ────────────────────────────────────────
                    y_f_raw = np.array(f(x_plot), dtype=float)
                    y_f_raw = np.where(np.isfinite(y_f_raw), y_f_raw, np.nan)

                    # Rango y de referencia: percentiles de f(x) + puntos clave
                    finite_f = y_f_raw[np.isfinite(y_f_raw)]
                    p5, p95  = (np.percentile(finite_f, [5, 95]) if len(finite_f) > 1
                                else (-1.0, 1.0))
                    refs  = np.array([0.0, p5, p95])
                    y_rng = max(float(refs.max() - refs.min()), 0.5)
                    y_lo  = float(refs.min()) - y_rng * 0.2
                    y_hi  = float(refs.max()) + y_rng * 0.2

                    # Clip: valores que sobrepasan el rango visible se ocultan
                    y_f = np.where((y_f_raw >= y_lo - y_rng) & (y_f_raw <= y_hi + y_rng),
                                   y_f_raw, np.nan)

                    fig = go.Figure()

                    # ── f(x) ────────────────────────────────────────────────
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_f,
                        mode="lines", name="f(x)",
                        line=dict(color="#2563eb", width=2.5),
                    ))

                    # ── Semilla x₀ ──────────────────────────────────────────
                    fig.add_trace(go.Scatter(
                        x=[x0], y=[float(f(x0))],
                        mode="markers+text",
                        name=f"x₀ = {x0:.{d}f}",
                        marker=dict(color="#9333ea", size=11, symbol="circle",
                                    line=dict(color="white", width=2)),
                        text=[f"x₀ = {x0:.{d}f}"],
                        textposition="top center",
                        textfont=dict(size=11),
                    ))

                    # ── Raíz hallada ────────────────────────────────────────
                    fig.add_vline(
                        x=root,
                        line=dict(color="rgba(120,120,120,0.6)", width=1.5, dash="dot"),
                    )
                    fig.add_trace(go.Scatter(
                        x=[root], y=[0.0],
                        mode="markers",
                        name=f"Raíz x* = {root:.{d}f}",
                        marker=dict(color="#dc2626", size=13, symbol="diamond",
                                    line=dict(color="white", width=2)),
                    ))

                    # ── Layout ──────────────────────────────────────────────
                    fig.update_layout(
                        xaxis_title="x",
                        yaxis_title="f(x)",
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