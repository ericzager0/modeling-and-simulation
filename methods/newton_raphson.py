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