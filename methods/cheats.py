import streamlit as st
import sympy as sp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de parseo (mismas convenciones que en bisection.py)
# ─────────────────────────────────────────────────────────────────────────────

x = sp.Symbol("x")

_LOCAL_DICT = {
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

_X0_ALIASES_POS = {"oo", "inf", "+inf", "+oo", "infinito", "+infinito", "infinity"}
_X0_ALIASES_NEG = {"-oo", "-inf", "-infinito", "-infinity"}


def _parse(func_str: str):
    """Devuelve (expr_sympy, latex_str) o lanza excepción."""
    expr = sp.sympify(func_str.replace("^", "**"), locals=_LOCAL_DICT)
    return expr, sp.latex(expr)


def _parse_x0(x0_str: str):
    """Evalúa el punto al que tiende x. Acepta números, pi, e, sqrt(...),
    y también infinito bajo los alias 'oo', 'inf', 'infinito', '-oo', etc.
    Devuelve una expresión sympy (puede ser sp.oo / -sp.oo)."""
    s2 = x0_str.strip().lower().replace(" ", "")
    if s2 in _X0_ALIASES_POS:
        return sp.oo
    if s2 in _X0_ALIASES_NEG:
        return -sp.oo
    local_dict = dict(_LOCAL_DICT)
    local_dict.pop("x", None)  # x0 no puede depender de x
    expr = sp.sympify(x0_str.replace("^", "**"), locals=local_dict)
    if expr.free_symbols:
        vars_str = ", ".join(str(s) for s in expr.free_symbols)
        raise ValueError(f"no puede depender de variables ({vars_str})")
    return expr


# ─────────────────────────────────────────────────────────────────────────────
# Formato de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val):
    """Formatea un resultado simbólico en LaTeX legible."""
    try:
        if val is sp.nan or (hasattr(val, "is_nan") and val.is_nan):
            return r"\text{no determinado}"
        if val in (sp.oo, -sp.oo, sp.zoo):
            return sp.latex(val)
        return sp.latex(sp.nsimplify(sp.simplify(val)))
    except Exception:
        return sp.latex(val)


def _safe_limit(expr, x0, dir="+"):
    try:
        return sp.limit(expr, x, x0, dir=dir)
    except Exception:
        return sp.nan


# ─────────────────────────────────────────────────────────────────────────────
# Explicación paso a paso de derivadas (para mostrar CÓMO se deriva en L'Hôpital)
# ─────────────────────────────────────────────────────────────────────────────

_DERIV_MAX_DEPTH = 4

_FUNC_RULES = {
    sp.sin:  ("\\sin", r"\cos(u)",              lambda u: sp.cos(u)),
    sp.cos:  ("\\cos", r"-\sin(u)",              lambda u: -sp.sin(u)),
    sp.tan:  ("\\tan", r"\sec^{2}(u)",           lambda u: 1 / sp.cos(u) ** 2),
    sp.exp:  ("\\exp", r"e^{u}",                 lambda u: sp.exp(u)),
    sp.log:  ("\\ln",  r"\dfrac{1}{u}",          lambda u: 1 / u),
    sp.sqrt: ("\\sqrt", r"\dfrac{1}{2\sqrt{u}}", lambda u: 1 / (2 * sp.sqrt(u))),
    sp.Abs:  ("|\\cdot|", r"\operatorname{sign}(u)", lambda u: sp.sign(u)),
}


def _explain_derivative(expr, depth: int = 0, indent: int = 0):
    """Deriva `expr` respecto de x explicando la regla aplicada en cada paso.
    Devuelve (derivada_simplificada, lista_de_lineas_markdown)."""

    expr = sp.sympify(expr)
    pad = "  " * indent

    if not expr.has(x):
        return sp.Integer(0), [f"{pad}- $\\dfrac{{d}}{{dx}}\\big[{sp.latex(expr)}\\big] = 0$ (es constante)"]

    if expr == x:
        return sp.Integer(1), [f"{pad}- $\\dfrac{{d}}{{dx}}[x] = 1$"]

    if depth >= _DERIV_MAX_DEPTH:
        d = sp.diff(expr, x)
        return sp.simplify(d), [
            f"{pad}- $\\dfrac{{d}}{{dx}}\\big[{sp.latex(expr)}\\big] = {sp.latex(sp.simplify(d))}$"
        ]

    # ── Suma / resta → regla de la suma ─────────────────────────────────
    if expr.is_Add:
        lines = [f"{pad}- **Regla de la suma** $(f+g)' = f' + g'$ para ${sp.latex(expr)}$:"]
        total = sp.Integer(0)
        for term in expr.args:
            d_term, sub = _explain_derivative(term, depth + 1, indent + 1)
            lines.extend(sub)
            total += d_term
        result = sp.simplify(total)
        lines.append(f"{pad}  $\\Rightarrow$ derivada total $= {sp.latex(result)}$")
        return result, lines

    # ── División explícita → regla del cociente ─────────────────────────
    num, den = sp.fraction(expr)
    if den != 1 and den.has(x):
        lines = [
            f"{pad}- **Regla del cociente** "
            f"$\\left(\\dfrac{{f}}{{g}}\\right)' = \\dfrac{{f'g - fg'}}{{g^2}}$, "
            f"con $f={sp.latex(num)}$, $g={sp.latex(den)}$:"
        ]
        d_num, num_lines = _explain_derivative(num, depth + 1, indent + 1)
        d_den, den_lines = _explain_derivative(den, depth + 1, indent + 1)
        lines.extend(num_lines)
        lines.extend(den_lines)
        result = sp.simplify((d_num * den - num * d_den) / den ** 2)
        lines.append(f"{pad}  $\\Rightarrow {sp.latex(result)}$")
        return result, lines

    # ── Producto de factores que dependen de x → regla del producto ─────
    if expr.is_Mul:
        x_factors = [f for f in expr.args if f.has(x)]
        const_factors = [f for f in expr.args if not f.has(x)]
        const = sp.Mul(*const_factors) if const_factors else sp.Integer(1)

        if len(x_factors) >= 2:
            f_ = x_factors[0]
            g_ = sp.Mul(*x_factors[1:])
            lines = [
                f"{pad}- **Regla del producto** $(fg)' = f'g + fg'$, "
                f"con $f={sp.latex(f_)}$, $g={sp.latex(g_)}$:"
            ]
            d_f, f_lines = _explain_derivative(f_, depth + 1, indent + 1)
            d_g, g_lines = _explain_derivative(g_, depth + 1, indent + 1)
            lines.extend(f_lines)
            lines.extend(g_lines)
            result = sp.simplify(const * (d_f * g_ + f_ * d_g))
            lines.append(f"{pad}  $\\Rightarrow {sp.latex(result)}$")
            return result, lines

        elif len(x_factors) == 1:
            d_f, f_lines = _explain_derivative(x_factors[0], depth + 1, indent)
            result = sp.simplify(const * d_f)
            if const != 1:
                f_lines.append(
                    f"{pad}  (multiplicada por la constante ${sp.latex(const)}$) "
                    f"$\\Rightarrow {sp.latex(result)}$"
                )
            return result, f_lines

    # ── Potencia u(x)^n, n constante → regla de la potencia + cadena ────
    if expr.is_Pow:
        base, exponent = expr.args
        if base.has(x) and not exponent.has(x):
            lines = [
                f"{pad}- **Regla de la potencia + cadena** "
                f"$(u^{{n}})' = n\\,u^{{n-1}}\\,u'$, con $u={sp.latex(base)}$, $n={sp.latex(exponent)}$:"
            ]
            d_base, base_lines = _explain_derivative(base, depth + 1, indent + 1)
            lines.extend(base_lines)
            result = sp.simplify(exponent * base ** (exponent - 1) * d_base)
            lines.append(f"{pad}  $\\Rightarrow {sp.latex(result)}$")
            return result, lines

        elif exponent.has(x):
            lines = [
                f"{pad}- **Regla exponencial + cadena** "
                f"$(a^{{u}})' = a^{{u}}\\ln(a)\\,u'$, con $a={sp.latex(base)}$, $u={sp.latex(exponent)}$:"
            ]
            d_exp, exp_lines = _explain_derivative(exponent, depth + 1, indent + 1)
            lines.extend(exp_lines)
            result = sp.simplify(expr * sp.log(base) * d_exp)
            lines.append(f"{pad}  $\\Rightarrow {sp.latex(result)}$")
            return result, lines

    # ── Funciones elementales con argumento no trivial → regla de la cadena
    if expr.func in _FUNC_RULES and len(expr.args) == 1:
        u = expr.args[0]
        name, template, deriv_fn = _FUNC_RULES[expr.func]
        outer_d = deriv_fn(u)

        if u == x:
            return sp.simplify(outer_d), [
                f"{pad}- **Derivada básica** $\\dfrac{{d}}{{dx}}[{name}(x)] = {sp.latex(outer_d)}$"
            ]

        lines = [
            f"{pad}- **Regla de la cadena**: $[{name}(u)]' = {template} \\cdot u'$, "
            f"con $u={sp.latex(u)}$:"
        ]
        d_u, u_lines = _explain_derivative(u, depth + 1, indent + 1)
        lines.extend(u_lines)
        result = sp.simplify(outer_d * d_u)
        lines.append(f"{pad}  $\\Rightarrow {sp.latex(result)}$")
        return result, lines

    # ── Caso base: usar sympy directamente ───────────────────────────────
    d = sp.diff(expr, x)
    return sp.simplify(d), [
        f"{pad}- $\\dfrac{{d}}{{dx}}\\big[{sp.latex(expr)}\\big] = {sp.latex(sp.simplify(d))}$"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Motor de resolución de indeterminaciones (paso a paso)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_power(b, e):
    if b == 1 and e in (sp.oo, -sp.oo):
        return r"1^{\infty}"
    if b == 0 and e == 0:
        return r"0^{0}"
    if b == sp.oo and e == 0:
        return r"\infty^{0}"
    return None


def _classify_fraction(n, d):
    if n == 0 and d == 0:
        return r"\dfrac{0}{0}"
    if n in (sp.oo, -sp.oo, sp.zoo) and d in (sp.oo, -sp.oo, sp.zoo):
        return r"\dfrac{\infty}{\infty}"
    return None


def _resolve(expr, x0, steps: list, depth: int = 0, max_depth: int = 8, hopital_count: list = None):
    """Resuelve el límite de `expr` en x -> x0, agregando explicaciones a
    `steps`. Devuelve el valor final (expresión sympy) o None si no se pudo
    determinar."""

    if hopital_count is None:
        hopital_count = [0]

    if depth >= max_depth:
        val = _safe_limit(expr, x0)
        steps.append(
            f"Tras varias transformaciones sucesivas, se calcula el límite "
            f"restante de forma directa: $= {_fmt(val)}$"
        )
        return val

    try:
        expr = sp.together(sp.powsimp(sp.simplify(expr)))
    except Exception:
        pass

    # ── Potencia con exponente que depende de x → 1^∞, 0^0, ∞^0 ────────────
    if expr.is_Pow and expr.exp.has(x):
        base, exponent = expr.base, expr.exp
        b_lim = _safe_limit(base, x0)
        e_lim = _safe_limit(exponent, x0)
        kind = _classify_power(b_lim, e_lim)
        if kind:
            steps.append(
                f"Forma indeterminada **${kind}$** "
                f"(la base tiende a ${_fmt(b_lim)}$ y el exponente a ${_fmt(e_lim)}$). "
                f"Para resolverla tomamos logaritmo natural: si "
                f"$L = \\displaystyle\\lim_{{x \\to {sp.latex(x0)}}} {sp.latex(expr)}$, entonces "
                f"$\\ln L = \\displaystyle\\lim_{{x \\to {sp.latex(x0)}}} "
                f"{sp.latex(exponent)} \\cdot \\ln\\!\\left({sp.latex(base)}\\right)$"
            )
            log_expr = sp.expand_log(exponent * sp.log(base), force=True)
            inner = _resolve(log_expr, x0, steps, depth + 1, max_depth, hopital_count)
            if inner is None:
                return None
            final = sp.exp(inner)
            steps.append(
                f"Como $\\ln L = {_fmt(inner)}$, entonces "
                f"$L = e^{{{_fmt(inner)}}} = {_fmt(final)}$"
            )
            return final
        else:
            try:
                val = b_lim ** e_lim
            except Exception:
                val = sp.nan
            steps.append(
                f"No hay indeterminación en la potencia: sustituyendo, la base "
                f"tiende a ${_fmt(b_lim)}$ y el exponente a ${_fmt(e_lim)}$ "
                f"$\\Rightarrow {_fmt(val)}$"
            )
            return val

    # ── Suma/resta → ∞ − ∞ ───────────────────────────────────────────────
    if expr.is_Add:
        terms = expr.args
        lims = [_safe_limit(t, x0) for t in terms]
        if sp.oo in lims and -sp.oo in lims:
            steps.append(
                "Forma indeterminada **$\\infty - \\infty$** "
                "(hay términos que tienden a $+\\infty$ y otros a $-\\infty$). "
                "Combinamos toda la expresión en una sola fracción con común "
                "denominador para poder aplicar L'Hôpital."
            )
            combined = sp.together(expr)
            return _resolve(combined, x0, steps, depth + 1, max_depth, hopital_count)
        else:
            val = sum(lims)
            steps.append(
                f"No hay indeterminación: sustituyendo término a término, "
                f"la suma de los límites da ${_fmt(val)}$"
            )
            return val

    # ── Fracción real (numerador / denominador no trivial) ─────────────────
    num, den = sp.fraction(expr)
    if den != 1:
        n_lim = _safe_limit(num, x0)
        d_lim = _safe_limit(den, x0)
        kind = _classify_fraction(n_lim, d_lim)
        if kind:
            hopital_count[0] += 1
            aplicacion = (
                f" (aplicación sucesiva n.° {hopital_count[0]})" if hopital_count[0] > 1 else ""
            )
            steps.append(
                f"Forma indeterminada **${kind}$** "
                f"(numerador $\\to {_fmt(n_lim)}$, denominador $\\to {_fmt(d_lim)}$). "
                f"Aplicamos la **Regla de L'Hôpital**{aplicacion}: derivamos numerador "
                f"y denominador **por separado** (no el cociente completo)."
            )

            d_num, num_expl = _explain_derivative(num)
            num_block = (
                f"**Derivada del numerador** $f(x) = {sp.latex(num)}$:\n\n" + "\n".join(num_expl)
            )
            steps.append(num_block)

            d_den, den_expl = _explain_derivative(den)
            den_block = (
                f"**Derivada del denominador** $g(x) = {sp.latex(den)}$:\n\n" + "\n".join(den_expl)
            )
            steps.append(den_block)

            steps.append(
                f"Por lo tanto: $f'(x) = {sp.latex(d_num)}$, $\\; g'(x) = {sp.latex(d_den)}$. "
                f"El nuevo límite a evaluar es "
                f"$\\displaystyle\\lim_{{x \\to {sp.latex(x0)}}} "
                f"\\dfrac{{{sp.latex(d_num)}}}{{{sp.latex(d_den)}}}$"
            )
            return _resolve(d_num / d_den, x0, steps, depth + 1, max_depth, hopital_count)
        else:
            if d_lim == 0:
                val = sp.zoo if n_lim != 0 else sp.nan
            else:
                val = n_lim / d_lim
            steps.append(
                f"No hay indeterminación: sustituyendo, numerador $\\to {_fmt(n_lim)}$, "
                f"denominador $\\to {_fmt(d_lim)}$ $\\Rightarrow {_fmt(val)}$"
            )
            return val

    # ── Producto sin denominador explícito → 0 · ∞ ──────────────────────────
    if expr.is_Mul:
        factors = list(expr.args)
        flims = [_safe_limit(f, x0) for f in factors]
        zero_idx = next((i for i, l in enumerate(flims) if l == 0), None)
        inf_idx = next((i for i, l in enumerate(flims) if l in (sp.oo, -sp.oo)), None)
        if zero_idx is not None and inf_idx is not None:
            zero_factor = factors[zero_idx]
            inf_factor = factors[inf_idx]
            rest_factors = [f for i, f in enumerate(factors) if i not in (zero_idx, inf_idx)]
            rest = sp.Mul(*rest_factors) if rest_factors else sp.Integer(1)

            numerator = inf_factor * rest
            denominator = 1 / zero_factor

            steps.append(
                "Forma indeterminada **$0 \\cdot \\infty$** "
                f"(el factor ${sp.latex(zero_factor)}$ tiende a $0$ y "
                f"${sp.latex(inf_factor)}$ tiende a $\\infty$). "
                f"Reescribimos el producto como cociente, invirtiendo el factor "
                f"que tiende a $0$: "
                f"$\\;{sp.latex(expr)} = \\dfrac{{{sp.latex(numerator)}}}{{{sp.latex(denominator)}}}$"
            )
            return _resolve(numerator / denominator, x0, steps, depth + 1, max_depth, hopital_count)

    # ── Caso base: sustitución directa, sin indeterminación ─────────────────
    val = _safe_limit(expr, x0)
    steps.append(f"Sustitución directa: no hay indeterminación $\\Rightarrow {_fmt(val)}$")
    return val


# ─────────────────────────────────────────────────────────────────────────────
# UI: Cheat "Resolver indeterminaciones"
# ─────────────────────────────────────────────────────────────────────────────

def _run_indeterminate_forms():
    with st.expander("📘 Teoría: Formas indeterminadas y cómo se resuelven", expanded=False):
        st.markdown(r"""
### ¿Qué es una forma indeterminada?

Al calcular $\lim_{x \to a} f(x)$, a veces la sustitución directa de $x=a$ no da un
número definido, sino una de estas **7 formas indeterminadas**:

$$\dfrac{0}{0}, \quad \dfrac{\infty}{\infty}, \quad 0 \cdot \infty, \quad
\infty - \infty, \quad 0^{0}, \quad 1^{\infty}, \quad \infty^{0}$$

En estos casos el resultado **no** se puede determinar por sustitución directa;
hay que transformar la expresión.
""")
        st.markdown("### Cómo se resuelve cada una")
        st.markdown(r"""
| Forma | Estrategia |
|---|---|
| $\dfrac{0}{0}$ y $\dfrac{\infty}{\infty}$ | **Regla de L'Hôpital** directamente: $\displaystyle\lim \frac{f}{g} = \lim \frac{f'}{g'}$ |
| $0 \cdot \infty$ | Reescribir como cociente ($f \cdot g = \frac{f}{1/g}$) y aplicar L'Hôpital |
| $\infty - \infty$ | Combinar en una sola fracción (común denominador) y aplicar L'Hôpital |
| $0^{0},\; 1^{\infty},\; \infty^{0}$ | Tomar logaritmo natural: si $L=\lim f^g$, entonces $\ln L = \lim g\ln f$ (que es $0\cdot\infty$), resolver ese límite y luego $L=e^{\ln L}$ |
""")
        st.markdown(r"""
**Regla de L'Hôpital:** si $\lim f(x) = \lim g(x) = 0$ (o ambos $\pm\infty$), y
$g'(x) \neq 0$ cerca de $a$, entonces:
""")
        st.latex(r"\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}")
        st.markdown("Puede aplicarse **más de una vez** si el resultado sigue siendo indeterminado.")

    st.subheader("Resolver indeterminaciones")

    func_str = st.text_input(
        "f(x)",
        value="sin(x)/x",
        placeholder="Ej: sin(x)/x,  x*ln(x),  (1+1/x)**x,  1/x - 1/sin(x)",
        key="cheat_indet_fx",
    )
    x0_str = st.text_input(
        "x tiende a",
        value="0",
        placeholder="Ej: 0,  pi/2,  oo,  -oo,  1",
        key="cheat_indet_x0",
    )

    expr = latex_expr = x0 = None
    if func_str:
        try:
            expr, latex_expr = _parse(func_str)
            st.latex(rf"f(x) = {latex_expr}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")
    if x0_str:
        try:
            x0 = _parse_x0(x0_str)
            st.latex(rf"x \to {sp.latex(x0)}")
        except Exception as e:
            st.error(f"No se pudo interpretar el punto: {e}")

    if st.button("Resolver", use_container_width=True, key="cheat_indet_btn"):
        if expr is None:
            st.error("Ingresá una función válida antes de resolver.")
        elif x0 is None:
            st.error("Ingresá un valor válido para x.")
        else:
            steps: list = []
            result = None
            try:
                result = _resolve(expr, x0, steps)
            except Exception as e:
                st.error(f"No se pudo resolver el límite: {e}")

            if result is not None:
                # Chequeo bilateral (solo informativo, para puntos finitos)
                bilateral_note = None
                if x0.is_finite:
                    l_izq = _safe_limit(expr, x0, dir="-")
                    l_der = _safe_limit(expr, x0, dir="+")
                    if l_izq != l_der:
                        bilateral_note = (l_izq, l_der)

                st.markdown("### Resolución paso a paso")
                for i, s in enumerate(steps, start=1):
                    st.markdown(f"**Paso {i}.**")
                    st.markdown(s)

                st.markdown("### Resultado")
                with st.container(border=True):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.caption("LÍMITE")
                        st.latex(rf"{_fmt(result)}")
                    with c2:
                        st.caption("PUNTO")
                        st.latex(rf"x \to {sp.latex(x0)}")

                if bilateral_note:
                    l_izq, l_der = bilateral_note
                    st.warning(
                        f"⚠️ Los límites laterales son distintos: "
                        f"por izquierda $\\to {_fmt(l_izq)}$, por derecha $\\to {_fmt(l_der)}$. "
                        f"El límite bilateral **no existe**; el resultado de arriba "
                        f"corresponde al límite por derecha."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Página
# ─────────────────────────────────────────────────────────────────────────────

CHEATS = [
    "Resolver indeterminaciones",
]


def run():
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)

    st.title("Cheats")

    cheat = st.selectbox("Elegí un cheat", CHEATS, key="cheat_selector")

    st.markdown("---")

    if cheat == "Resolver indeterminaciones":
        _run_indeterminate_forms()