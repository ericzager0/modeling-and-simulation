import streamlit as st
import sympy as sp


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

    st.title("Método de Punto Fijo")

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
    x0_str = st.text_input("x₀  (valor inicial)", value="1.5")
    x0 = None
    try:
        x0 = float(x0_str)
    except ValueError:
        st.error("Valor inválido para x₀")

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