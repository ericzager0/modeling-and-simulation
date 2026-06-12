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
    t = sp.Symbol("t")
    y = sp.Symbol("y")
    return t, y, {
        "t":    t,    "y":    y,
        "e":    sp.E, "E":    sp.E,
        "pi":   sp.pi,
        "ln":   sp.log,  "log":  sp.log,
        "exp":  sp.exp,
        "sin":  sp.sin,  "cos":  sp.cos,  "tan":  sp.tan,
        "sqrt": sp.sqrt,
        "Abs":  sp.Abs,  "abs":  sp.Abs,
    }


def _parse_ode(func_str: str):
    """Parsea la RHS de y' = f(t, y). Devuelve (callable, expr, latex)."""
    t, y, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    f    = sp.lambdify((t, y), expr, modules=["numpy"])
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


def _solve_exact(expr, t0: float, y0: float):
    """
    Intenta resolver y' = expr con y(t0) = y0 de forma simbólica.
    Devuelve (callable, rhs_expr) o (None, None).
    """
    t, y, _ = _local_dict()
    yf = sp.Function("y")
    try:
        t0_s = sp.nsimplify(t0, rational=True)
        y0_s = sp.nsimplify(y0, rational=True)
        ode_eq = sp.Eq(yf(t).diff(t), expr.subs(y, yf(t)))
        sol    = sp.dsolve(ode_eq, yf(t), ics={yf(t0_s): y0_s})
        if isinstance(sol, list):
            sol = sol[0]
        rhs  = sol.rhs
        func = sp.lambdify(t, rhs, modules=["numpy"])
        # sanity check: evaluable y finita en t0
        if not math.isfinite(float(func(float(t0)))):
            raise ValueError("Valor no finito en t0.")
        return func, rhs
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Métodos numéricos  →  list[dict]
# ─────────────────────────────────────────────────────────────────────────────

def _euler(f, t0, y0, t_end, h):
    N = round((t_end - t0) / h)
    rows, t_n, y_n = [], t0, y0
    for k in range(N):
        y_next = y_n + h * f(t_n, y_n)
        rows.append({"n": k, "t": t_n, "y": y_n, "y_next": y_next})
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({"n": N, "t": t_n, "y": y_n, "y_next": None})
    return rows


def _heun(f, t0, y0, t_end, h):
    N = round((t_end - t0) / h)
    rows, t_n, y_n = [], t0, y0
    for k in range(N):
        k1     = f(t_n, y_n)
        k2     = f(t_n + h, y_n + h * k1)
        y_next = y_n + (h / 2) * (k1 + k2)
        rows.append({"n": k, "t": t_n, "y": y_n, "y_next": y_next})
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({"n": N, "t": t_n, "y": y_n, "y_next": None})
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
            "n": k, "t": t_n, "y": y_n,
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "y_next": y_next,
        })
        t_n, y_n = t0 + (k + 1) * h, y_next
    rows.append({
        "n": N, "t": t_n, "y": y_n,
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
            yr = float(y_exact_func(r["t"]))
            r["y_real"] = yr
            r["error"]  = abs(yr - r["y"])
        except Exception:
            r["y_real"] = None
            r["error"]  = None
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
        "tₙ":    [r["t"]           for r in rows],
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
        "tₙ":    [r["t"]           for r in rows],
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
        "tₙ":      [r["t"]  for r in e_rows],
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

    st.title("Métodos Numéricos para EDOs")

    # ── EDO ───────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "y' = f(t, y)  —  lado derecho de la ecuación diferencial",
        value="",
        placeholder="Ej: y - t**2 + 1   |   -2*t*y   |   sin(t) + y",
    )
    f_func = f_expr = latex_f = None
    if func_str.strip():
        try:
            f_func, f_expr, latex_f = _parse_ode(func_str)
            st.latex(rf"\dfrac{{dy}}{{dt}} \;=\; {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Condición inicial ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        t0_str = st.text_input("t₀  —  tiempo inicial", value="0",
                               placeholder="Ej: 0")
    with c2:
        y0_str = st.text_input("y₀  —  valor inicial  y(t₀)", value="",
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
        "t_end  —  extremo derecho del intervalo",
        value="", placeholder="Ej: 2"
    )
    t_end = None
    if tend_str.strip():
        try:
            t_end = _parse_val(tend_str)
            if t0 is not None:
                st.latex(
                    rf"t \;\in\; \left[\,{_fmt(t0)},\;{_fmt(t_end)}\,\right]"
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
        if t0 is None:      missing.append("t₀ e y₀")
        if t_end is None:   missing.append("t_end")
        if h_val is None:   missing.append("h")
        if missing:
            st.error(f"Faltá ingresar: {', '.join(missing)}.")
            st.stop()
        if t_end <= t0:
            st.error("t_end debe ser mayor que t₀.")
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
            "t_end":     t_end_eff,
            "h":         h_val,
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
            rf"Solución exacta encontrada: $\quad y(t) = {sp.latex(res['y_ex_expr'])}$"
        )
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
            <div class="rc-label">Euler — y(t_end)</div>
            <div class="rc-value">{_fmt(fe, 6)}</div>
            <div class="rc-sub">t = {_fmt(t_end_r)}</div>
        </div>
        <div class="result-card" style="border-top:3px solid #f59e0b;">
            <div class="rc-label">Heun — y(t_end)</div>
            <div class="rc-value">{_fmt(fh, 6)}</div>
            <div class="rc-sub">t = {_fmt(t_end_r)}</div>
        </div>
        <div class="result-card" style="border-top:3px solid #16a34a;">
            <div class="rc-label">RK4 — y(t_end)</div>
            <div class="rc-value">{_fmt(fr, 6)}</div>
            <div class="rc-sub">t = {_fmt(t_end_r)}</div>
        </div>"""
    if has_exact:
        fx = float(res["y_ex_func"](t_end_r))
        cards += f"""
        <div class="result-card" style="border-top:3px solid #2563eb;">
            <div class="rc-label">Exacto — y(t_end)</div>
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
        st.latex(r"y_{n+1} = y_n + h \cdot f(t_n,\; y_n)")
        df_e = _df_euler_heun(e_rows, has_exact)
        st.dataframe(df_e, column_config=_col_cfg(df_e),
                     hide_index=True, use_container_width=True)

    with tab_h:
        st.markdown("#### Método de Heun  (Euler mejorado)")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"k_1 = f(t_n,\; y_n)")
            st.latex(r"k_2 = f(t_n + h,\; y_n + h\,k_1)")
        with col2:
            st.latex(r"y_{n+1} = y_n + \dfrac{h}{2}\,(k_1 + k_2)")
        df_h = _df_euler_heun(h_rows, has_exact)
        st.dataframe(df_h, column_config=_col_cfg(df_h),
                     hide_index=True, use_container_width=True)

    with tab_r:
        st.markdown("#### Runge–Kutta de orden 4")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"k_1 = f(t_n,\; y_n)")
            st.latex(
                r"k_2 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_1\right)"
            )
            st.latex(
                r"k_3 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_2\right)"
            )
            st.latex(r"k_4 = f(t_n+h,\; y_n+h\,k_3)")
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

    t_pts = [r["t"] for r in e_rows]
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
        xaxis_title="t",
        yaxis_title="y(t)",
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