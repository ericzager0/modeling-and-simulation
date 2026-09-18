import re
import streamlit as st
import sympy as sp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de parseo
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
    """Devuelve (expr_sympy, latex_str) o lanza excepción."""
    x, ld = _local_dict()
    expr = sp.sympify(func_str.replace("^", "**"), locals=ld)
    return expr, _to_latex(expr)


def _parse_number(num_str: str) -> float:
    """Evalúa expresiones numéricas fijas (pi, e, sqrt(2), etc.) a float."""
    _, ld = _local_dict()
    ld_no_x = dict(ld)
    ld_no_x.pop("x", None)
    expr = sp.sympify(num_str.replace("^", "**"), locals=ld_no_x)
    if expr.free_symbols:
        vars_str = ", ".join(str(s) for s in expr.free_symbols)
        raise ValueError(f"no puede depender de variables ({vars_str})")
    return float(expr)


# ─────────────────────────────────────────────────────────────────────────────
# Motor de Búsqueda y Evaluación de g(x)
# ─────────────────────────────────────────────────────────────────────────────

def _generar_candidatos_gx(f_expr):
    """Genera candidatos para g(x) aplicando diversas estrategias algebraicas."""
    x, _ = _local_dict()
    candidatos = []

    # Estrategia 1 y 2: Triviales x ± f(x)
    candidatos.append(("Trivial: x + f(x)", f_expr + x))
    candidatos.append(("Trivial: x - f(x)", x - f_expr))

    # Estrategia 3: Extracción algebraica de términos polinómicos a * x**n
    f_expandida = sp.expand(f_expr)
    terminos = sp.Add.make_args(f_expandida)

    a = sp.Wild("a", exclude=[x])
    n = sp.Wild("n", exclude=[x])

    for termino in terminos:
        match = termino.match(a * x**n)
        if match and match[a] != 0 and match[n] != 0:
            val_a = match[a]
            val_n = match[n]
            resto = f_expandida - termino
            base = -resto / val_a

            if val_n == 1:
                candidatos.append((f"Despeje lineal (término {termino})", base))
            else:
                # Raíz general
                candidatos.append(
                    (f"Despeje raíz {val_n} (término {termino})", base ** (1 / val_n))
                )
                # Si la potencia es par, consideramos también la rama negativa
                if val_n.is_integer and val_n % 2 == 0:
                    candidatos.append(
                        (f"Despeje raíz -{val_n} (término {termino})", -(base ** (1 / val_n)))
                    )

                # Estrategia 4: División por x (ej: x^3 = x + 1 => x^2 = 1 + 1/x => x = sqrt(1 + 1/x))
                if val_n.is_integer and val_n > 1:
                    n_red = int(val_n) - 1
                    base_div = sp.cancel(base / x)
                    if n_red == 1:
                        candidatos.append(
                            (f"Despeje por división por x (término {termino})", base_div)
                        )
                    else:
                        candidatos.append(
                            (f"Despeje por división raíz {n_red} (término {termino})", base_div ** (1 / n_red))
                        )

    # Estrategia 5: Factorización de x con término independiente (si f(0) != 0)
    try:
        c = f_expr.subs(x, 0)
        if c != 0 and c.is_finite:
            resto_sin_c = sp.simplify((f_expr - c) / x)
            if resto_sin_c != 0:
                g_fact = -c / resto_sin_c
                candidatos.append(("Factorización de x: x = -c / [(f(x)-c)/x]", g_fact))
    except Exception:
        pass

    # Estrategia 6: Iterador de Newton-Raphson g(x) = x - f(x)/f'(x)
    f_der = sp.diff(f_expr, x)
    if f_der != 0:
        candidatos.append(("Iterador de Newton-Raphson: x - f(x)/f'(x)", x - (f_expr / f_der)))

    return candidatos


def _evaluar_candidatos(candidatos, x0: float, decimals: int = 6):
    """Evalúa y filtra cada g(x) según el Criterio de Lipschitz en x0."""
    x, _ = _local_dict()
    vistos = []
    resultados = []

    for estrategia, g in candidatos:
        # Simplificación para agrupar términos y detectar duplicados
        try:
            g_simp = sp.simplify(g)
        except Exception:
            g_simp = g

        # Evitar candidatos idénticos
        ya_visto = False
        for v in vistos:
            try:
                if sp.simplify(g_simp - v) == 0:
                    ya_visto = True
                    break
            except Exception:
                if str(g_simp) == str(v):
                    ya_visto = True
                    break
        if ya_visto:
            continue
        vistos.append(g_simp)

        # Cálculo de derivada y evaluación
        try:
            gp = sp.diff(g_simp, x)
            g_eval = g_simp.subs(x, x0).evalf()
            gp_eval = gp.subs(x, x0).evalf()

            # Descartar si el resultado tiene parte imaginaria no trivial
            im_g = float(sp.im(g_eval))
            im_gp = float(sp.im(gp_eval))
            if abs(im_g) > 1e-9 or abs(im_gp) > 1e-9:
                resultados.append({
                    "estrategia": estrategia,
                    "g": g_simp,
                    "gp": gp,
                    "latex_g": _to_latex(g_simp),
                    "latex_gp": _to_latex(gp),
                    "py_str": str(g_simp),
                    "gp_val": None,
                    "abs_gp": None,
                    "cumple_lipschitz": False,
                    "estado": "Fuera del dominio real (raíz compleja en x₀)",
                    "simulacion": [],
                    "error": "Valor no real en x₀",
                })
                continue

            g_val = float(sp.re(g_eval))
            gp_val = float(sp.re(gp_eval))

            if np.isnan(g_val) or np.isinf(g_val) or np.isnan(gp_val) or np.isinf(gp_val):
                cumple_lipschitz = False
                abs_gp = None
                estado = "Indeterminación matemática en x₀ (división por cero)"
                err_msg = "Valor indefinido en x₀"
                simulacion = []
            else:
                abs_gp = abs(gp_val)
                cumple_lipschitz = abs_gp < 1.0
                estado = "¡CONVERGE!" if cumple_lipschitz else "DIVERGE (o no garantizado)"
                err_msg = None

                # Simulación de las primeras 5 iteraciones
                simulacion = []
                curr = x0
                sim_ok = True
                for i in range(5):
                    try:
                        nxt_eval = g_simp.subs(x, curr).evalf()
                        if abs(float(sp.im(nxt_eval))) > 1e-9:
                            sim_ok = False
                            break
                        nxt = float(sp.re(nxt_eval))
                        if not np.isfinite(nxt):
                            sim_ok = False
                            break
                        simulacion.append({"n": i, "xn": curr, "xn1": nxt})
                        curr = nxt
                    except Exception:
                        sim_ok = False
                        break

            resultados.append({
                "estrategia": estrategia,
                "g": g_simp,
                "gp": gp,
                "latex_g": _to_latex(g_simp),
                "latex_gp": _to_latex(gp),
                "py_str": str(g_simp),
                "gp_val": gp_val if err_msg is None else None,
                "abs_gp": abs_gp,
                "cumple_lipschitz": cumple_lipschitz,
                "estado": estado,
                "simulacion": simulacion if err_msg is None and sim_ok else [],
                "error": err_msg,
            })

        except Exception as e:
            resultados.append({
                "estrategia": estrategia,
                "g": g_simp,
                "gp": None,
                "latex_g": _to_latex(g_simp),
                "latex_gp": "N/A",
                "py_str": str(g_simp),
                "gp_val": None,
                "abs_gp": None,
                "cumple_lipschitz": False,
                "estado": "Indeterminación matemática en x₀",
                "simulacion": [],
                "error": str(e),
            })

    return resultados


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

    .candidate-box {
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.04);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Teoría (desplegable) ──────────────────────────────────────────────────
    with st.expander("📘 Teoría: ¿Cómo despejar g(x) y Criterio de Lipschitz?", expanded=False):
        st.markdown(r"""
### ¿Por qué necesitamos $g(x)$?

En el **Método de Punto Fijo**, para resolver una ecuación $f(x) = 0$, transformamos algebraicamente
el problema en una ecuación equivalente de la forma:

$$x = g(x)$$

Un punto $x^*$ que satisface $x^* = g(x^*)$ es un **punto fijo** de $g$, y coincide exactamente con una
raíz de $f(x) = 0$.

Sin embargo, a partir de una misma $f(x) = 0$ se pueden obtener **muchas funciones $g(x)$ distintas**,
y **no todas convergen**. Algunas divergen rápidamente o entran en ciclos oscilatorios.

---

### Criterio de Lipschitz / Condición de Contracción

De acuerdo con el **Teorema del Punto Fijo de Banach**, la iteración $x_{n+1} = g(x_n)$ converge hacia
la raíz $x^*$ en un intervalo que contenga a la raíz si $g(x)$ es una función diferenciable y cumple:

$$|g'(x)| < 1$$

- **Si $|g'(x_0)| < 1$**: la función $g$ actúa como una **contracción** cerca de $x_0$. La sucesión
  de aproximaciones se acercará progresivamente al punto fijo.
- **Velocidad de convergencia**: cuanto más cercano a $0$ sea $|g'(x_0)|$, más rápida será la
  convergencia. Si $g'(x^*) = 0$, la convergencia es cuadrática (de segundo orden).
- **Si $|g'(x_0)| \ge 1$**: la iteración diverge o no tiene garantía de convergencia desde ese punto.

---

### Estrategias de despeje automático en esta herramienta

1. **Despejes triviales**:
   $$g(x) = x + f(x) \quad \text{o} \quad g(x) = x - f(x)$$
2. **Extracción algebraica de potencias ($a \cdot x^n$)**:
   Se aísla el monomio $a \cdot x^n = -\text{resto} \implies x = \left(-\frac{\text{resto}}{a}\right)^{1/n}$.
   *(Para potencias pares, se evalúan tanto la rama positiva como la negativa).*
3. **Despeje fraccionario / división por $x$**:
   Se divide la relación de potencias por $x$ para obtener formas como $x = \sqrt{1 + 1/x}$.
4. **Factorización de $x$**:
   Si existe término independiente $c \neq 0$, se factoriza $x$ del resto: $x = \frac{-c}{(f(x)-c)/x}$.
5. **Iterador tipo Newton-Raphson**:
   $$g(x) = x - \frac{f(x)}{f'(x)}$$
   Esta forma particular satisface $g'(x^*) = 0$ en la raíz, asegurando convergencia cuadrática.
        """)

    st.title("Buscador de Funciones de Iteración g(x)")
    st.markdown(
        "Ingresá $f(x)$ y un punto inicial $x_0$. La herramienta despejará automáticamente "
        "candidatos a $g(x)$ y evaluará el **criterio de Lipschitz** $|g'(x_0)| < 1$ para "
        "identificar cuáles garantizan convergencia."
    )

    # ── Entradas ─────────────────────────────────────────────────────────────
    col_fn, col_x0 = st.columns([3, 1])

    func_str = col_fn.text_input(
        "f(x) — Función objetivo",
        value="x**3 - x - 1",
        placeholder="Ej: x**3 - x - 1,  2*exp(x**2) - 5*x,  ln(x) - 1,  log(x)",
        help="ln(x) = logaritmo natural (base e) · log(x) = logaritmo decimal (base 10) · log(x, b) = base b",
    )

    x0_str = col_x0.text_input(
        "x₀ (valor inicial)",
        value="1.0",
        placeholder="Ej: 1.0, 0, pi/4, sqrt(2)",
    )

    f_expr = None
    if func_str:
        try:
            f_expr, latex_f = _parse_f(func_str)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar f(x): {e}")

    x0 = None
    if x0_str:
        try:
            x0 = _parse_number(x0_str)
        except Exception as e:
            st.error(f"Valor inválido para x₀: {e}")

    col_dec, col_chk = st.columns([1, 2])
    decimals_str = col_dec.text_input("Decimales a mostrar", value="6")
    try:
        decimals = int(decimals_str)
        if decimals < 0:
            decimals = 6
    except ValueError:
        decimals = 6

    solo_convergentes = col_chk.checkbox("Mostrar únicamente opciones que convergen", value=False)

    # ── Botón de Acción ───────────────────────────────────────────────────────
    if st.button("Buscar funciones g(x)", use_container_width=True):
        if not f_expr:
            st.error("Ingresá una función f(x) válida antes de continuar.")
        elif x0 is None:
            st.error("Ingresá un valor inicial x₀ válido.")
        else:
            with st.spinner("Despejando candidatos y evaluando criterio de Lipschitz..."):
                candidatos = _generar_candidatos_gx(f_expr)
                resultados = _evaluar_candidatos(candidatos, x0, decimals)

            convergentes = [r for r in resultados if r["cumple_lipschitz"]]
            divergentes = [r for r in resultados if not r["cumple_lipschitz"]]

            # Tarjetas de resumen métrico
            st.markdown(f"""
            <div class="result-cards">
                <div class="result-card" style="border-top: 3px solid #2563eb;">
                    <div class="rc-label">Total analizadas</div>
                    <div class="rc-value">{len(resultados)}</div>
                    <div class="rc-sub">candidatos g(x)</div>
                </div>
                <div class="result-card" style="border-top: 3px solid #16a34a;">
                    <div class="rc-label">Convergen (|g'| &lt; 1)</div>
                    <div class="rc-value">{len(convergentes)}</div>
                    <div class="rc-sub">opciones viables</div>
                </div>
                <div class="result-card" style="border-top: 3px solid #dc2626;">
                    <div class="rc-label">Divergen o no garantizadas</div>
                    <div class="rc-value">{len(divergentes)}</div>
                    <div class="rc-sub">|g'| &ge; 1 o fuera de dominio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Opciones que Convergen ────────────────────────────────────────
            if convergentes:
                st.success(
                    f"✅ **Se encontraron {len(convergentes)} opción(es) que cumplen el "
                    f"criterio de Lipschitz (|g'({x0})| < 1):**"
                )

                # Ordenar por |g'(x0)| ascendente (el más cercano a 0 converge más rápido)
                convergentes_ordenados = sorted(
                    convergentes,
                    key=lambda r: r["abs_gp"] if r["abs_gp"] is not None else 999.0,
                )

                for idx, item in enumerate(convergentes_ordenados, start=1):
                    es_mejor = idx == 1
                    distintivo_mejor = " 🚀 *(Mayor velocidad de convergencia)*" if es_mejor else ""
                    border_color = "#16a34a" if not es_mejor else "#2563eb"

                    st.markdown(f"""
                    <div class="candidate-box" style="border-left: 5px solid {border_color};">
                        <strong style="font-size: 1.15rem;">Opción {idx}: {item['estrategia']}</strong>{distintivo_mejor}
                    </div>
                    """, unsafe_allow_html=True)

                    st.latex(rf"g(x) = {item['latex_g']}")
                    st.latex(rf"g'(x) = {item['latex_gp']}")

                    gp_num = item['gp_val']
                    abs_gp_num = item['abs_gp']
                    st.info(
                        rf"**Evaluación en $x_0 = {x0}$:** "
                        rf"$g'({x0}) = {gp_num:.{decimals}f} \implies "
                        rf"|g'({x0})| = {abs_gp_num:.{decimals}f} < 1$  —  **¡Lipschitz OK!**"
                    )

                    st.caption("Fórmula lista para copiar y usar en la pestaña de **Punto Fijo**:")
                    st.code(item["py_str"], language="python")

                    # Simulación rápida si está disponible
                    if item["simulacion"]:
                        with st.expander(f"Ver primeras {len(item['simulacion'])} iteraciones con esta g(x)", expanded=False):
                            sim_lines = [
                                r"| $n$ | $x_n$ | $x_{n+1} = g(x_n)$ | $|x_{n+1} - x_n|$ |",
                                "|:---:|:---:|:---:|:---:|",
                            ]
                            for row in item["simulacion"]:
                                diff = abs(row["xn1"] - row["xn"])
                                sim_lines.append(
                                    f"| {row['n']} | {row['xn']:.{decimals}f} | {row['xn1']:.{decimals}f} | {diff:.{decimals}f} |"
                                )
                            st.markdown("\n".join(sim_lines))

            else:
                st.warning(
                    f"⚠️ No se encontraron opciones que cumplan estrictamente $|g'({x0})| < 1$. "
                    "Podés probar con otro valor inicial $x_0$ más cercano a la raíz esperada."
                )

            # ── Tabla Resumen Comparativa ────────────────────────────────────
            st.markdown("---")
            st.markdown("### Tabla Comparativa de Todas las Candidatas Analizadas")

            tabla_items = convergentes if solo_convergentes else resultados
            if tabla_items:
                headers = [
                    r"| # | Estrategia | $g(x)$ | $|g'(x_0)|$ | Estado Lipschitz |",
                    "|:---:|:---|:---:|:---:|:---:|",
                ]
                for idx, r in enumerate(tabla_items, start=1):
                    val_str = f"{r['abs_gp']:.{decimals}f}" if r["abs_gp"] is not None else "N/A"
                    status_icon = "✅ Cumple (< 1)" if r["cumple_lipschitz"] else "❌ No cumple (≥ 1)"
                    if r["error"]:
                        status_icon = f"⚠️ {r['estado']}"
                    # Truncar LaTeX si es excesivamente largo para la tabla
                    latex_disp = rf"${r['latex_g']}$"
                    headers.append(
                        f"| {idx} | {r['estrategia']} | {latex_disp} | {val_str} | {status_icon} |"
                    )
                st.markdown("\n".join(headers))

            # ── Desplegable con Candidatas que Divergen ───────────────────────
            if divergentes and not solo_convergentes:
                with st.expander(f"❌ Ver {len(divergentes)} candidatas descartadas o divergentes en x₀", expanded=False):
                    for idx, item in enumerate(divergentes, start=1):
                        st.markdown(f"**Candidata descartada #{idx}: {item['estrategia']}**")
                        st.latex(rf"g(x) = {item['latex_g']}")
                        if item["gp"] is not None:
                            st.latex(rf"g'(x) = {item['latex_gp']}")
                            if item["abs_gp"] is not None:
                                st.write(
                                    f"Valor en $x_0 = {x0}$: $|g'({x0})| = {item['abs_gp']:.{decimals}f} \\ge 1$ — **Diverge o no garantizado**"
                                )
                        if item["error"]:
                            st.write(f"Detalle: {item['estado']}")
                        st.divider()
