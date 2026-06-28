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
    """Devuelve (callable, latex_str) o lanza excepción."""
    x, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f = sp.lambdify(x, expr, modules=["numpy"])
    return f, sp.latex(expr)


def _parse_g(func_str: str):
    """Parsea g(x) y su derivada.
    Devuelve (callable_g, callable_gp, latex_g, latex_gp) o lanza excepción."""
    x, ld = _local_dict()
    expr     = sp.sympify(func_str.replace("^", "**"), locals=ld)
    expr_der = sp.diff(expr, x)
    g  = sp.lambdify(x, expr,     modules=["numpy"])
    gp = sp.lambdify(x, expr_der, modules=["numpy"])
    return g, gp, sp.latex(expr), sp.latex(expr_der)


def _aitken(g, x0: float, tol: float, max_iter: int):
    """Método de Aitken:

       Dado xₙ, se calculan:
         xₙ₊₁  = g(xₙ)
         xₙ₊₂  = g(xₙ₊₁)
         Δxₙ   = xₙ₊₁ − xₙ
         Δ²xₙ  = xₙ₊₂ − 2·xₙ₊₁ + xₙ
         x̂ₙ   = xₙ − (Δxₙ)² / Δ²xₙ   ← aceleración de Aitken

       x̂ₙ se usa como punto de arranque de la siguiente iteración.
       Criterio de parada: |x̂ₙ − xₙ| ≤ tol.
    """
    rows = []
    xn = x0
    for i in range(max_iter):
        xn1   = float(g(xn))
        xn2   = float(g(xn1))
        denom = xn2 - 2.0 * xn1 + xn          # Δ²xₙ
        if abs(denom) < 1e-15:                  # denominador nulo: no se puede continuar
            rows.append({"n": i, "xn": xn, "xn1": xn1, "xn2": xn2, "xstar": None})
            return xn, rows, False
        xstar = xn - (xn1 - xn) ** 2 / denom   # fórmula de Aitken
        rows.append({"n": i, "xn": xn, "xn1": xn1, "xn2": xn2, "xstar": xstar})
        if abs(xstar - xn) <= tol:
            return xstar, rows, True
        xn = xstar
    return xn, rows, False


def _cobweb_path_aitken(x0: float, rows):
    """Construye dos trayectorias separadas para el diagrama de telaraña de
    Aitken/Steffensen:

       - (real_x, real_y): los pasos GENUINOS de iteración simple, dos por cada
         ciclo (xn -> xn1 -> xn2), dibujados como los escalones vertical/horizontal
         de siempre. Cada ciclo se separa del anterior con un None para que
         Plotly no los conecte entre sí con una línea recta.
       - (jump_x, jump_y): los SALTOS de extrapolación de Aitken, de (xn2, xn2)
         a (x̂n, x̂n). No son evaluaciones de g(x) — son la fórmula de Aitken — y
         por eso van en una serie aparte, para graficarse con otro estilo y no
         confundirse con un paso real de la iteración.
    """
    real_x, real_y = [], []
    jump_x, jump_y = [], []
    for row in rows:
        xn, xn1, xn2, xstar = row["xn"], row["xn1"], row["xn2"], row["xstar"]

        if real_x:
            real_x.append(None)
            real_y.append(None)
        # escalón 1: (xn,xn)   -> (xn,xn1)   -> (xn1,xn1)
        # escalón 2: (xn1,xn1) -> (xn1,xn2)  -> (xn2,xn2)
        real_x += [xn, xn, xn1, xn1, xn2]
        real_y += [xn, xn1, xn1, xn2, xn2]

        if xstar is not None:
            jump_x += [xn2, xstar, None]
            jump_y += [xn2, xstar, None]

    return real_x, real_y, jump_x, jump_y


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
    with st.expander("📘 Teoría: ¿Cómo funciona el Método de Aitken?", expanded=False):
        st.markdown(r"""
### ¿Qué es?

El **método de Aitken** (proceso $\Delta^2$ de Aitken) no es un método de punto fijo nuevo,
sino una técnica para **acelerar** una sucesión que ya converge — en este caso, la sucesión
generada por la iteración de punto fijo $x_{n+1} = g(x_n)$.

La iteración de punto fijo simple converge, en general, **linealmente**: el error de un paso
es aproximadamente proporcional al error del paso anterior. Aitken aprovecha justamente ese
patrón: si conocemos tres términos consecutivos de la sucesión ($x_n$, $x_{n+1}$, $x_{n+2}$)
y el error decae de forma aproximadamente geométrica, podemos "extrapolar" matemáticamente
hacia dónde está convergiendo la sucesión, sin esperar a que llegue ahí por sí sola.
""")

        st.markdown("### La fórmula")
        st.markdown(r"A partir de $x_n$, se generan dos pasos de la iteración simple:")
        st.latex(r"x_{n+1} = g(x_n) \qquad x_{n+2} = g(x_{n+1})")
        st.markdown("Y se calculan las diferencias:")
        st.latex(r"\Delta x_n = x_{n+1} - x_n \qquad \Delta^2 x_n = x_{n+2} - 2x_{n+1} + x_n")
        st.markdown("La estimación acelerada de la raíz es:")
        st.latex(r"\hat{x}_n = x_n - \frac{(\Delta x_n)^2}{\Delta^2 x_n}")

        st.markdown("### Paso a paso del algoritmo")
        st.markdown(r"""
1. **Despejar $g(x)$** de $f(x) = 0$ (igual que en punto fijo) y elegir un valor inicial $x_0$.
2. **Generar dos pasos** de la iteración simple a partir de $x_n$: $x_{n+1} = g(x_n)$ y $x_{n+2} = g(x_{n+1})$.
3. **Calcular $\Delta^2 x_n = x_{n+2} - 2x_{n+1} + x_n$.**
   - Si $\Delta^2 x_n \approx 0$ → no se puede dividir, el proceso se detiene (ver más abajo).
4. **Calcular el valor acelerado** $\hat{x}_n = x_n - \dfrac{(\Delta x_n)^2}{\Delta^2 x_n}$.
5. **Verificar el criterio de parada:** si $|\hat{x}_n - x_n| \le \varepsilon$, $\hat{x}_n$ es la raíz aproximada.
6. **Si no, usar $\hat{x}_n$ como nuevo $x_n$** y repetir desde el paso 2, hasta convergencia o máximo de iteraciones.

*(Aplicar la aceleración una sola vez sobre la sucesión ya generada se conoce como "proceso $\Delta^2$
de Aitken"; repetir este proceso paso a paso, usando siempre el último valor acelerado como nueva
semilla — que es justo lo que hace esta calculadora — se conoce como **método de Steffensen**.)*

### ¿Por qué es más rápido?

Donde el punto fijo simple necesita muchos pasos chiquitos para acercarse a la raíz, Aitken
"salta" directamente a una mejor estimación usando la información de los tres últimos puntos.
En la práctica esto suele traducirse en **muchas menos iteraciones** para la misma tolerancia
— por ejemplo, con $f(x) = x^3 - x - 2$, $g(x) = \sqrt[3]{x+2}$ y $x_0 = 1.5$, el punto fijo
simple necesita 11 iteraciones para una tolerancia de $10^{-10}$, mientras que Aitken llega
al mismo resultado en solo 3.

### Cuando algo puede salir mal

- **Denominador nulo ($\Delta^2 x_n \approx 0$):** pasa cuando los tres puntos están casi
  alineados (la sucesión casi no está curvándose). Ahí no se puede dividir, y la calculadora
  lo marca explícitamente en vez de devolver un número sin sentido.
- **Inestabilidad numérica:** si $\Delta^2 x_n$ es muy chico (aunque no exactamente cero), dividir
  por él puede amplificar errores de redondeo. Por eso conviene no confiar ciegamente en
  iteraciones donde el denominador es casi nulo.
- Esta calculadora sigue pidiendo el mismo chequeo de Lipschitz ($|g'(x_0)| < 1$) que el punto
  fijo simple antes de arrancar. En la práctica, Aitken/Steffensen a veces puede converger
  incluso en casos límite donde la iteración simple es muy lenta — pero seguimos usando el
  mismo chequeo preliminar para mantener un criterio consistente entre todos los métodos.

### Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Acelera notablemente la convergencia lineal de punto fijo | Puede fallar si $\Delta^2 x_n \approx 0$ (división por (casi) cero) |
| No necesita calcular derivadas de $f$ ni de $g$ para acelerar | Sensible a errores de redondeo cuando el denominador es chico |
| Convergencia cercana a la cuadrática (similar a Newton-Raphson) cuando funciona bien | Sigue dependiendo de tener una $g(x)$ razonable, igual que punto fijo |
""")

    st.title("Método de Aitken")

    # ── f(x) ─────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)",
        value="x**3 - x - 2",
        placeholder="Ej: x**2 - 4,  sin(x) - x/2,  exp(x) - 3",
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
        placeholder="Ej: (x + 2)**(1/3),  cos(x),  (x**2 + 4) / 3",
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
            root, rows, converged = _aitken(g, x0, tol, max_iter)
            n = len(rows)

            # ── Mensaje de estado ─────────────────────────────────────────────
            last = rows[-1]
            if last["xstar"] is None:
                st.error(
                    "Δ²xₙ = 0 en la iteración "
                    f"{last['n']}: denominador nulo, no se puede aplicar Aitken."
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
                r"| $n$ | $x_n$ | $x_{n+1} = g(x_n)$ | $x_{n+2} = g(x_{n+1})$ |"
                r" $x_n^* = x_n - \dfrac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}$ |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ]
            for row in rows:
                xstar_str = (
                    f"{row['xstar']:.{d}f}"
                    if row["xstar"] is not None
                    else "—"
                )
                lines.append(
                    f"| {row['n']} "
                    f"| {row['xn']:.{d}f} "
                    f"| {row['xn1']:.{d}f} "
                    f"| {row['xn2']:.{d}f} "
                    f"| {xstar_str} |"
                )
            st.markdown("\n".join(lines))

            # ══════════════════════════════════════════════════════════════════
            # GRÁFICO — telaraña de los pasos reales de g(x) + saltos de Aitken
            # ══════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### Gráfico (telaraña + saltos de extrapolación)")
            st.caption(
                "🟠 Pasos reales de la iteración simple, g(xₙ).   "
                "🟢 punteado: salto de extrapolación de Aitken — no es una evaluación de "
                "g(x), es la fórmula x̂ₙ = xₙ − (Δxₙ)²/Δ²xₙ, por eso avanza sobre la "
                "diagonal y = x en vez de subir hasta la curva."
            )

            try:
                # ── Rango en x: x0 + todos los valores recorridos / saltados ──
                traj_xs = [x0]
                for row in rows:
                    traj_xs += [row["xn1"], row["xn2"]]
                    if row["xstar"] is not None:
                        traj_xs.append(row["xstar"])

                span_x = max(traj_xs) - min(traj_xs) if max(traj_xs) != min(traj_xs) else 1.0
                pad    = max(span_x * 0.6, 1.0)
                x_lo   = min(traj_xs) - pad
                x_hi   = max(traj_xs) + pad
                x_plot = np.linspace(x_lo, x_hi, 600)

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

                # ── Trayectorias: pasos reales vs. saltos de Aitken ─────────
                real_x, real_y, jump_x, jump_y = _cobweb_path_aitken(x0, rows)

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

                # ── Pasos reales (telaraña genuina) ───────────────────────
                fig.add_trace(go.Scatter(
                    x=real_x, y=real_y,
                    mode="lines",
                    name="Pasos reales: g(xₙ)",
                    line=dict(color="#f97316", width=1.8),
                ))

                # ── Saltos de extrapolación de Aitken ─────────────────────
                if jump_x:
                    fig.add_trace(go.Scatter(
                        x=jump_x, y=jump_y,
                        mode="lines+markers",
                        name="Salto de Aitken (extrapolación)",
                        line=dict(color="#0d9488", width=2.4, dash="dot"),
                        marker=dict(color="#0d9488", size=7, symbol="circle"),
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