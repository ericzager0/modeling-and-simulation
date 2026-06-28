import streamlit as st
import sympy as sp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse(func_str: str):
    """Devuelve (callable, latex_str) o lanza excepción."""
    x = sp.Symbol("x")
    # Mapeo explícito: sympify trata 'e' como símbolo libre sin esto
    local_dict = {
        "x":    x,
        "e":    sp.E,       # e**x  →  número de Euler
        "E":    sp.E,
        "pi":   sp.pi,
        "ln":   sp.log,     # ln(x) →  logaritmo natural
        "log":  sp.log,
        "exp":  sp.exp,
        "sin":  sp.sin,
        "cos":  sp.cos,
        "tan":  sp.tan,
        "sqrt": sp.sqrt,
        "Abs":  sp.Abs,
        "abs":  sp.Abs,
    }
    expr = sp.sympify(func_str.replace("^", "**"), locals=local_dict)
    # numpy resuelve exp/sin/cos/log sin conflictos de tipos
    f = sp.lambdify(x, expr, modules=["numpy"])
    return f, sp.latex(expr)


def _bisect(f, a: float, b: float, tol: float, max_iter: int):
    rows = []
    fa = float(f(a))
    c = (a + b) / 2.0  # valor por defecto si max_iter llega a ser 0
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = float(f(c))
        rows.append({"i": i, "a": a, "b": b, "c": c, "fc": fc})
        if abs(fc) <= tol or (b - a) / 2.0 <= tol:
            return c, rows, True
        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
    # Importante: "c" ya quedó seteado en la última fila de la tabla (rows[-1]),
    # así el resultado final siempre coincide con lo que se muestra en pantalla.
    return c, rows, False


# ─────────────────────────────────────────────────────────────────────────────
# Página
# ─────────────────────────────────────────────────────────────────────────────

def run():
    st.markdown("""
    <style>
    /* ── Tamaño base de letra (Streamlit default: 16px) ── */
    html { font-size: 20px; }

    /* ── Botón rojo a ancho completo ── */
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

    /* ── Tabla centrada ── */
    table { margin: 0.5rem auto 0 auto; }
    table th, table td { text-align: center !important; }

    /* ── Tarjetas de resultado ── */
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
    with st.expander("📘 Teoría: ¿Cómo funciona el Método de Bisección?", expanded=False):
        st.markdown(r"""
### ¿Qué es?

El **método de bisección** es un método numérico para encontrar raíces (ceros) de una
función continua $f(x)$, es decir, valores $x^{*}$ tales que $f(x^{*}) = 0$.

Se basa en el **Teorema de Bolzano** (teorema del valor intermedio):

> Si $f$ es continua en $[a, b]$ y $f(a) \cdot f(b) < 0$ (signos opuestos en los extremos),
> entonces existe **al menos una raíz** $x^{*} \in (a, b)$.

Por eso, antes de empezar a iterar, siempre se verifica esta condición sobre el intervalo elegido.
""")

        st.markdown("### Paso a paso del algoritmo")
        st.markdown(r"""
1. **Elegir un intervalo $[a, b]$** en el que se cumpla Bolzano: $f(a) \cdot f(b) < 0$.
2. **Calcular el punto medio** del intervalo:
""")
        st.latex(r"c = \frac{a+b}{2}")
        st.markdown(r"""
3. **Evaluar $f(c)$.**
4. **Verificar el criterio de parada.** Se detiene si:
   - $|f(c)| \le \varepsilon$ (la función ya vale ≈ 0 en $c$), **o**
   - $\dfrac{b-a}{2} \le \varepsilon$ (el intervalo ya es lo bastante chico)

   En ese caso, $c$ es la raíz aproximada buscada.
5. **Si todavía no converge, decidir en qué mitad del intervalo sigue estando la raíz**,
   comparando el signo de $f(a)$ con el de $f(c)$:
   - Si $f(a) \cdot f(c) < 0$ → la raíz está en $[a, c]$ → se actualiza $b = c$ (queda $a$ igual).
   - Si $f(a) \cdot f(c) > 0$ → la raíz está en $[c, b]$ → se actualiza $a = c$ (queda $b$ igual).
6. **Repetir** desde el paso 2 con el nuevo intervalo $[a,b]$ ya reducido, hasta cumplir
   el criterio de parada o alcanzar el número máximo de iteraciones permitido.
""")

        st.markdown("### ¿Por qué funciona?")
        st.markdown(r"""
En cada iteración el intervalo que contiene a la raíz se reduce exactamente a la mitad.
Esto significa que, si Bolzano se cumple al principio, el método **siempre converge**
(no hace falta que $f$ sea derivable, solo continua). Es decir, es muy robusto, aunque
no el más rápido.

La cota de error después de $n$ iteraciones es:
""")
        st.latex(r"|x^{*} - c_n| \;\le\; \frac{b-a}{2^{\,n+1}}")

        st.markdown(r"""
### Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Convergencia **garantizada** si se cumple Bolzano | Convergencia **lineal**: relativamente lenta frente a Newton-Raphson o secante |
| No necesita calcular derivadas | Requiere un intervalo inicial válido (con cambio de signo) |
| Muy simple, estable y fácil de implementar | No detecta raíces de multiplicidad par, donde $f$ no cambia de signo |
""")

    st.title("Método de Bisección")

    # ── f(x) ─────────────────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)",
        value="x**3 - x - 2",
        placeholder="Ej: x**2 - 4,  sin(x) - x/2,  exp(x) - 3",
    )

    f = latex_f = None
    if func_str:
        try:
            f, latex_f = _parse(func_str)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Intervalo [a, b] ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    a_str = col_a.text_input("a", value="1")
    b_str = col_b.text_input("b", value="2")
    try:
        a = float(a_str)
    except ValueError:
        a = None
        col_a.error("Valor inválido")
    try:
        b = float(b_str)
    except ValueError:
        b = None
        col_b.error("Valor inválido")
    if a is not None and b is not None:
        st.latex(rf"[a,\; b] = [{a},\; {b}]")

    # ── Bolzano ───────────────────────────────────────────────────────────────
    bolzano_ok = False
    if f and a is not None and b is not None:
        try:
            fa_val = float(f(a))
            fb_val = float(f(b))
            prod = fa_val * fb_val
            bolzano_ok = prod < 0
            sign = r"<" if bolzano_ok else r"\geq"
            latex_bolzano = (
                rf"$f(a) \cdot f(b) = {fa_val:.6g} \cdot {fb_val:.6g}"
                rf" = {prod:.6g} \; {sign} \; 0$"
            )
            if bolzano_ok:
                st.success(f"**Bolzano se cumple:** {latex_bolzano}")
            else:
                st.error(f"**Bolzano no se cumple:** {latex_bolzano}")
        except Exception as e:
            st.error(f"Error al evaluar en el intervalo: {e}")

    # ── Parámetros ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    tol_str  = c1.text_input("Tolerancia (ε)", value="0.000001")
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
        elif a is None or b is None:
            st.error("Ingresá valores válidos para a y b.")
        elif tol is None:
            st.error("Ingresá una tolerancia válida.")
        elif max_iter is None or decimals is None:
            st.error("Ingresá valores válidos para iteraciones y decimales.")
        elif max_iter < 1:
            st.error("El número de iteraciones debe ser al menos 1.")
        elif decimals < 0:
            st.error("Los decimales deben ser un número entero ≥ 0.")
        elif a >= b:
            st.error("Se requiere $a < b$.")
        elif not bolzano_ok:
            st.warning("Bolzano no se cumple: el intervalo no garantiza una raíz.")
        else:
            root, rows, converged = _bisect(f, a, b, tol, max_iter)
            n = len(rows)

            # Resultado
            if converged:
                st.success(f"**Convergencia alcanzada en {n} iteraciones**")
            else:
                st.warning(f"**Sin convergencia** tras {n} iteraciones. "
                           "Mostrando mejor aproximación.")

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

            # Tabla con encabezados LaTeX (markdown nativo de Streamlit)
            d = decimals
            lines = [
                r"| $i$ | $a$ | $b$ | $c = \dfrac{a+b}{2}$ | $f(c)$ |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ]
            for row in rows:
                lines.append(
                    f"| {row['i']} "
                    f"| {row['a']:.{d}f} "
                    f"| {row['b']:.{d}f} "
                    f"| {row['c']:.{d}f} "
                    f"| {row['fc']:.{d}f} |"
                )
            st.markdown("\n".join(lines))