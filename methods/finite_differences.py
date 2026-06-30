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
    if len(set(round(v, 10) for v in xs)) != len(xs):
        raise ValueError("Los valores de x deben ser distintos (nodos únicos).")
    return xs, ys


def _find_node(xs: list, target: float, tol: float = 1e-7):
    """Busca el índice de un nodo en xs cuyo valor sea ≈ target. None si no existe."""
    for i, xv in enumerate(xs):
        if abs(xv - target) <= tol:
            return i
    return None


def _nearest_left_right(xs: list, x0: float):
    """
    Devuelve (idx_izq, idx_der): los índices del nodo inmediatamente a la
    izquierda y a la derecha de x0 (estrictos, sin contar un posible nodo
    igual a x0). None en el lugar que corresponda si no existe ese vecino
    (x0 antes del primer nodo o después del último).
    """
    idx_izq = None
    idx_der = None
    mejor_dist_izq = None
    mejor_dist_der = None
    for i, xv in enumerate(xs):
        if xv < x0 - 1e-9:
            d = x0 - xv
            if mejor_dist_izq is None or d < mejor_dist_izq:
                mejor_dist_izq, idx_izq = d, i
        elif xv > x0 + 1e-9:
            d = xv - x0
            if mejor_dist_der is None or d < mejor_dist_der:
                mejor_dist_der, idx_der = d, i
    return idx_izq, idx_der


def _find_symmetric_pair(xs: list, x0: float, tol: float = 1e-7):
    """
    Busca, entre TODOS los nodos de la tabla, el par (x_-, x_+) tal que
    x_- < x0 < x_+ y además x0 - x_- == x_+ - x0 (es decir, ambos a la
    MISMA distancia h de x0 — un par genuinamente simétrico).

    Si hay varios pares simétricos posibles, se queda con el de h más chico
    (igual criterio que el resto de la función: h chico = mejor aproximación).

    Returns
    -------
    (h, idx_izq, idx_der) si existe al menos un par simétrico.
    (None, None, None) si no existe ninguno.
    """
    n = len(xs)
    mejor = None  # (h, idx_izq, idx_der)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, xj = xs[i], xs[j]
            if xi < x0 - 1e-9 and xj > x0 + 1e-9:
                a = x0 - xi
                b = xj - x0
                if abs(a - b) <= tol:
                    h = a
                    if mejor is None or h < mejor[0]:
                        mejor = (h, i, j)
    if mejor is None:
        return None, None, None
    return mejor


def _auto_h(xs: list, x0: float, kind: str):
    """
    Busca automáticamente un paso h > 0 válido para la tabla de puntos.

    Cubre DOS escenarios, según si x0 coincide con un nodo de la tabla o cae
    estrictamente entre dos nodos (lo cual también es un caso legítimo: se
    puede derivar en un punto no tabulado usando sus vecinos más cercanos):

    CASO A — x0 ES un nodo de la tabla:
        - "progresiva" necesita  x0  y  x0 + h   (usa f(x0) real)
        - "regresiva"  necesita  x0  y  x0 - h   (usa f(x0) real)
        - "centrada"   necesita  x0 - h  y  x0 + h  (no usa f(x0))
      Estrategia: se consideran todos los espaciados |xi - xj| posibles
      entre nodos de la tabla, de menor a mayor, y se toma el primer h que
      conecte x0 con el/los vecinos exactos que la fórmula necesita.
      En este caso el par x0-h / x0+h usa siempre el MISMO h de cada lado,
      así que la centrada resultante es, por construcción, simétrica.

    CASO B — x0 NO es un nodo (cae estrictamente entre dos nodos):
        - "progresiva" y "regresiva" NO se pueden calcular: ambas requieren
          f(x0) tabulado, que no existe. Se devuelve (None, None, True).
        - "centrada": se intenta, en este orden:
            1) Buscar entre TODOS los nodos de la tabla si existe un par
               (x_-, x_+) EXACTAMENTE equidistante de x0 (misma distancia h
               a cada lado, aunque no sean los vecinos inmediatos). Si existe
               más de uno, se usa el de h más chico. Este caso conserva la
               precisión O(h²) propia de la fórmula centrada, porque es
               matemáticamente una centrada clásica con paso h.
            2) Si NINGÚN par es exactamente simétrico, se recurre como último
               recurso al vecino más próximo a la izquierda (x_-) y a la
               derecha (x_+) de x0 (que en general NO son equidistantes), y
               se define h como el promedio de ambas distancias:
                   h = ((x0 - x_-) + (x_+ - x0)) / 2
               aproximando la derivada con la pendiente entre esos dos puntos:
                   f'(x0) ≈ (f(x_+) - f(x_-)) / (x_+ - x_-)
               OJO: al no ser simétrico, este caso pierde la ventaja de orden
               O(h²) de la centrada y queda en O(h) (igual que una progresiva
               o regresiva). El llamador debe avisar esto al usuario.

    Returns
    -------
    (h, idx_dict, simetrica) si se encontró un h válido, donde idx_dict mapea
    los nombres de los nodos usados ("x0", "x0+h", "x0-h") a su índice en xs,
    y `simetrica` (bool) indica si el par usado es genuinamente equidistante
    de x0 (True) o es el atajo asimétrico de último recurso (False). Para
    progresiva/regresiva, `simetrica` no aplica y siempre vale True.
    (None, None, True) si no existe ningún h que permita aplicar esa fórmula.
    """
    n = len(xs)
    idx0 = _find_node(xs, x0)

    # ── CASO A: x0 es un nodo exacto de la tabla ────────────────────────────
    if idx0 is not None:
        candidatos = sorted({
            round(abs(xs[i] - xs[j]), 10)
            for i in range(n) for j in range(n) if i != j
        })
        candidatos = [h for h in candidatos if h > 1e-9]

        for h in candidatos:
            if kind == "progresiva":
                i1 = _find_node(xs, x0 + h)
                if i1 is not None:
                    return h, {"x0": idx0, "x0+h": i1}, True
            elif kind == "regresiva":
                i1 = _find_node(xs, x0 - h)
                if i1 is not None:
                    return h, {"x0": idx0, "x0-h": i1}, True
            elif kind == "centrada":
                i1 = _find_node(xs, x0 + h)
                i2 = _find_node(xs, x0 - h)
                if i1 is not None and i2 is not None:
                    return h, {"x0": idx0, "x0+h": i1, "x0-h": i2}, True
        return None, None, True

    # ── CASO B: x0 no es nodo, cae entre dos nodos de la tabla ──────────────
    if kind in ("progresiva", "regresiva"):
        # Requieren f(x0) tabulado, que no existe en este caso.
        return None, None, True

    # kind == "centrada"

    # 1) Preferir un par EXACTAMENTE simétrico, aunque no sean los vecinos
    #    inmediatos: conserva la precisión O(h²) real de la centrada.
    h_sim, i_izq_sim, i_der_sim = _find_symmetric_pair(xs, x0)
    if h_sim is not None:
        return h_sim, {"x0+h": i_der_sim, "x0-h": i_izq_sim}, True

    # 2) Último recurso: vecinos más próximos de cada lado, aunque no sean
    #    simétricos. Sigue siendo una estimación válida, pero de orden O(h).
    idx_izq, idx_der = _nearest_left_right(xs, x0)
    if idx_izq is None or idx_der is None:
        # x0 queda fuera del rango de la tabla (antes del primer nodo o
        # después del último): no hay vecinos de ambos lados.
        return None, None, True

    x_izq, x_der = xs[idx_izq], xs[idx_der]
    h = ((x0 - x_izq) + (x_der - x0)) / 2.0
    return h, {"x0+h": idx_der, "x0-h": idx_izq}, False


def _deriv_progresiva(ys, idx):
    y0, y1 = ys[idx["x0"]], ys[idx["x0+h"]]
    return y1 - y0, y0, y1, None


def _deriv_regresiva(ys, idx):
    y0, y1 = ys[idx["x0"]], ys[idx["x0-h"]]
    return y0 - y1, y1, y0, None


def _deriv_centrada(ys, idx):
    y_menos, y_mas = ys[idx["x0-h"]], ys[idx["x0+h"]]
    return y_mas - y_menos, y_menos, y_mas, None


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
    with st.expander("📘 Teoría: ¿Qué son las Diferencias Finitas?", expanded=False):
        st.markdown(r"""
### ¿Qué son?

Las **diferencias finitas** son un método numérico para **aproximar la derivada**
de una función $f$ en un punto $x_0$, usando únicamente **valores conocidos de $f$**
en puntos cercanos a $x_0$, en lugar de derivar $f$ de forma analítica.

Surgen directamente de la definición de derivada:
""")
        st.latex(r"f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}")
        st.markdown(r"""
Si en vez de tomar el límite $h \to 0$ usamos un valor de $h$ **pequeño pero finito**
(de ahí el nombre "diferencias finitas"), obtenemos una **aproximación** de $f'(x_0)$
calculada solo con un puñado de evaluaciones de $f$.
""")

        st.markdown("### ¿Por qué son útiles?")
        st.markdown(r"""
En la práctica, muchas veces **no conviene o no se puede derivar $f$ analíticamente**:

- $f$ puede ser **desconocida**: solo se cuenta con una tabla de datos experimentales
  o medidos (por ejemplo, temperatura cada hora, posición de un objeto cada segundo),
  sin una fórmula explícita detrás.
- $f$ puede ser **muy difícil o costosa de derivar a mano**, o no tener una expresión
  cerrada simple (funciones definidas por partes, soluciones numéricas de otra
  ecuación, salidas de una simulación, etc.).
- Incluso conociendo $f$, evaluarla en varios puntos y combinar los valores puede
  ser **mucho más rápido** que obtener y evaluar su derivada simbólica.

En todos estos casos, las diferencias finitas permiten estimar $f'(x_0)$ con buena
precisión **trabajando solo con los valores $f(x_i)$ ya disponibles en una tabla**,
sin necesidad de conocer la expresión algebraica de $f$.
""")

        st.markdown("### Los tres tipos de diferencias finitas")
        st.markdown(r"""
Dado un paso $h > 0$, existen tres formas básicas de aproximar $f'(x_0)$, según qué
nodos vecinos a $x_0$ se usen:
""")
        st.latex(r"\text{Progresiva (adelante):}\quad f'(x_0) \approx \frac{f(x_0+h) - f(x_0)}{h}")
        st.latex(r"\text{Regresiva (atrás):}\quad\;\;\, f'(x_0) \approx \frac{f(x_0) - f(x_0-h)}{h}")
        st.latex(r"\text{Centrada:}\quad\quad\quad\;\; f'(x_0) \approx \frac{f(x_0+h) - f(x_0-h)}{2h}")
        st.markdown(r"""
- La **progresiva** usa $x_0$ y un punto a la derecha ($x_0 + h$). Es la única opción
  si $x_0$ es el primer nodo de la tabla (no hay datos a su izquierda).
- La **regresiva** usa $x_0$ y un punto a la izquierda ($x_0 - h$). Es la única opción
  si $x_0$ es el último nodo de la tabla (no hay datos a su derecha).
- La **centrada** usa un punto a cada lado ($x_0 - h$ y $x_0 + h$), sin necesitar
  $f(x_0)$. Es la más precisa de las tres (su error es de orden $h^2$, mientras que
  el de las otras dos es de orden $h$), por eso se prefiere siempre que $x_0$ tenga
  vecinos de ambos lados disponibles — **siempre que esos dos vecinos sean
  realmente equidistantes de $x_0$** (ver más abajo qué pasa si no lo son).
""")

        st.markdown("### ¿Cómo se halla $h$?")
        st.markdown(r"""
El paso $h$ no es un valor que se elige libremente: tiene que ser una **distancia que
realmente exista entre nodos de la tabla**, porque la fórmula necesita evaluar $f$
(o, en este caso, leer $y$) exactamente en esos puntos — no se inventan valores
intermedios.

Por eso, para hallar $h$ de forma automática, este programa:

1. Calcula **todas las distancias posibles** entre pares de nodos $x_i$ de la tabla.
2. Las ordena de **menor a mayor** (un $h$ más chico da, en general, una mejor
   aproximación, ya que el error de truncamiento depende de $h$).
3. Para cada tipo de diferencia (progresiva, regresiva, centrada), recorre esas
   distancias y se queda con la **primera (más chica) que efectivamente conecte
   a $x_0$ con el o los nodos vecinos que esa fórmula necesita**.
4. Si **ningún** valor de $h$ permite armar una fórmula dada — por ejemplo, si $x_0$
   es el último nodo y no existe ningún nodo a su derecha — el programa lo informa
   con un error en lugar de mostrar un resultado inválido.

De esta manera, cada una de las tres fórmulas puede terminar usando un $h$ distinto
(el mejor disponible para esa fórmula en particular), y es perfectamente normal que
alguna de las tres no pueda calcularse si la tabla no tiene los nodos necesarios.

**¿Y si $x_0$ no es exactamente un nodo de la tabla?** Esto también es un caso válido
y bastante común: querer derivar en un punto intermedio, del cual no se tiene $f(x_0)$
tabulado, pero sí se tienen valores cercanos a ambos lados. En esa situación:

- Las fórmulas **progresiva** y **regresiva** no se pueden aplicar, porque ambas
  necesitan el valor exacto $f(x_0)$ en el numerador, y ese valor no está disponible.
- La fórmula **centrada** sí puede aplicarse, pero el programa primero busca, entre
  **todos** los nodos de la tabla (no solo los vecinos inmediatos), si existe algún
  par $x_-$, $x_+$ que esté **exactamente a la misma distancia** $h$ de $x_0$ a cada
  lado. Si existe (y puede haber más de uno, en cuyo caso se usa el de $h$ más
  chico), se aplica la centrada clásica con ese $h$, conservando su error $O(h^2)$.
- Solo si **ningún** par de la tabla es exactamente simétrico respecto a $x_0$, el
  programa recurre como último recurso al vecino más cercano a la izquierda ($x_-$)
  y a la derecha ($x_+$) — que en general **no** son equidistantes — y aproxima:
""")
        st.latex(r"f'(x_0) \approx \frac{f(x_+) - f(x_-)}{x_+ - x_-}")
        st.markdown(r"""
Esta expresión coincide con la centrada clásica cuando $x_+$ y $x_-$ están a la misma
distancia de $x_0$, pero si no lo están (que es justamente el caso en que se la usa,
como último recurso), **pierde la propiedad de error $O(h^2)$** y pasa a tener error
de orden $h$ — la misma precisión que una progresiva o regresiva. El programa avisa
explícitamente cada vez que esto ocurre.
""")

        st.markdown("### Ventajas y desventajas")
        st.markdown(r"""
| Ventajas | Desventajas |
|---|---|
| No requiere conocer la expresión analítica de $f$ | El resultado es una **aproximación**, no un valor exacto |
| Muy simple de calcular a partir de una tabla de datos | La precisión depende fuertemente de qué tan chico sea $h$ |
| Aplicable a datos experimentales o medidos | Necesita que existan nodos vecinos adecuados en la tabla |
| La fórmula centrada es más precisa (error de orden $h^2$) | Las fórmulas progresiva/regresiva son menos precisas (error de orden $h$) |
| | La centrada solo logra $O(h^2)$ si el par de nodos usado es realmente simétrico respecto a $x_0$ |
""")

    st.title("Diferencias Finitas")

    # ── f(x) — opcional ───────────────────────────────────────────────────────
    func_str = st.text_input(
        "f(x)  —  función original (opcional)",
        value="",
        placeholder="Ej: sin(x),  exp(x),  x**3 - 2*x + 1",
    )

    f_func = latex_f = None
    if func_str.strip():
        try:
            f_func, latex_f = _parse_f(func_str)
            st.latex(rf"f(x) = {latex_f}")
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")

    # ── Puntos ────────────────────────────────────────────────────────────────
    xs_str = st.text_input(
        "Valores de x  (separados por comas)",
        value="0,0.5,1,1.5,2",
        placeholder="Ej: 0, 0.5, 1, 1.5, 2  —  o también: 0, pi/4, pi/2",
    )
    ys_str = st.text_input(
        "Valores de y  (separados por comas)",
        value="1,1.6487,2.7183,4.4817,7.3891",
        placeholder="Ej: 1, 1.6487, 2.7183, 4.4817, 7.3891",
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

    # ── x0 — punto donde derivar ──────────────────────────────────────────────
    x0_str = st.text_input(
        "x₀  —  punto donde derivar",
        value="1",
        placeholder="Ej: 1  —  o también: pi/2, e",
    )

    x0_val = None
    if x0_str.strip():
        try:
            x0_val = _parse_val(x0_str)
        except Exception:
            st.error("Valor inválido para x₀.")

    # ── Botón ─────────────────────────────────────────────────────────────────
    if st.button("Calcular derivada", use_container_width=True):
        if xs is None or ys is None:
            st.error("Ingresá puntos válidos antes de calcular.")
            return
        if len(xs) < 2:
            st.warning("Necesitás al menos 2 puntos para poder derivar.")
            return
        if x0_val is None:
            st.error("Ingresá un valor válido para x₀.")
            return

        es_nodo = _find_node(xs, x0_val) is not None
        if not es_nodo:
            idx_izq_chk, idx_der_chk = _nearest_left_right(xs, x0_val)
            if idx_izq_chk is None or idx_der_chk is None:
                st.error(
                    f"x₀ = {_fmt(x0_val)} queda fuera del rango de la tabla "
                    f"(que va de {_fmt(min(xs))} a {_fmt(max(xs))}). "
                    "No hay nodos a ambos lados de x₀, así que ninguna de las "
                    "tres fórmulas puede aplicarse."
                )
                return

        st.markdown("---")
        st.markdown(f"### Punto de derivación: $x_0 = {_fmt(x0_val)}$")

        if not es_nodo:
            st.info(
                f"x₀ = {_fmt(x0_val)} **no es un nodo de la tabla**, cae entre dos "
                "valores tabulados. Por eso solo se puede calcular la **diferencia "
                "centrada** (no requiere $f(x_0)$ tabulado); las fórmulas "
                "progresiva y regresiva necesitan ese valor exacto y no aplican aquí."
            )

        if f_func is not None:
            try:
                st.latex(rf"f({_fmt(x0_val)}) = {float(f_func(x0_val)):.6g}")
            except Exception:
                pass

        # ── Las 3 fórmulas a resolver ───────────────────────────────────────
        formulas = [
            {
                "kind":  "progresiva",
                "title": "Diferencia Progresiva",
                "icon":  "▶",
                "color": "#2563eb",
                "formula_latex": r"f'(x_0) \approx \dfrac{f(x_0+h) - f(x_0)}{h}",
                "compute": _deriv_progresiva,
            },
            {
                "kind":  "regresiva",
                "title": "Diferencia Regresiva",
                "icon":  "◀",
                "color": "#9333ea",
                "formula_latex": r"f'(x_0) \approx \dfrac{f(x_0) - f(x_0-h)}{h}",
                "compute": _deriv_regresiva,
            },
            {
                "kind":  "centrada",
                "title": "Diferencia Centrada",
                "icon":  "◆",
                "color": "#16a34a",
                "formula_latex": r"f'(x_0) \approx \dfrac{f(x_0+h) - f(x_0-h)}{2h}",
                "compute": _deriv_centrada,
            },
        ]

        resultados = {}  # kind -> dict con resultado, para usar en el resumen final

        for fdata in formulas:
            kind  = fdata["kind"]
            color = fdata["color"]

            st.markdown("---")
            st.markdown(f"### {fdata['icon']} {fdata['title']}")
            st.latex(fdata["formula_latex"])

            h, idx, simetrica = _auto_h(xs, x0_val, kind)

            if h is None:
                # ── No se pudo hallar h: mostrar error explicativo ──────────
                if kind == "progresiva":
                    if not es_nodo:
                        motivo = (
                            f"$x_0 = {_fmt(x0_val)}$ no es un nodo de la tabla, "
                            "y esta fórmula necesita el valor exacto $f(x_0)$ tabulado."
                        )
                    else:
                        motivo = (
                            f"no existe ningún nodo a la derecha de $x_0 = {_fmt(x0_val)}$ "
                            "en la tabla (¿es el último punto?)."
                        )
                elif kind == "regresiva":
                    if not es_nodo:
                        motivo = (
                            f"$x_0 = {_fmt(x0_val)}$ no es un nodo de la tabla, "
                            "y esta fórmula necesita el valor exacto $f(x_0)$ tabulado."
                        )
                    else:
                        motivo = (
                            f"no existe ningún nodo a la izquierda de $x_0 = {_fmt(x0_val)}$ "
                            "en la tabla (¿es el primer punto?)."
                        )
                else:
                    motivo = (
                        f"no existe un nodo a cada lado de $x_0 = {_fmt(x0_val)}$ "
                        "en la tabla (faltan vecinos en alguno de los dos lados)."
                    )
                st.error(
                    f"**No se pudo hallar un valor de $h$ válido para la {fdata['title'].lower()}:** "
                    f"{motivo}"
                )
                resultados[kind] = None
                continue

            # ── h hallado: mostrar de dónde sale y resolver ─────────────────
            st.success(f"**h hallado automáticamente:** $h = {_fmt(h)}$")

            if kind == "centrada" and not simetrica:
                x_menos_w = xs[idx["x0-h"]]
                x_mas_w   = xs[idx["x0+h"]]
                st.warning(
                    "⚠️ **Vecinos asimétricos:** no existe en la tabla ningún par de "
                    f"nodos exactamente equidistante de $x_0 = {_fmt(x0_val)}$. Se usó "
                    f"el par más próximo disponible a cada lado "
                    f"($x_- = {_fmt(x_menos_w)}$, a distancia {_fmt(x0_val - x_menos_w)}; "
                    f"$x_+ = {_fmt(x_mas_w)}$, a distancia {_fmt(x_mas_w - x0_val)}). "
                    "Sigue siendo una estimación válida de $f'(x_0)$, pero al no ser "
                    "simétrica respecto a $x_0$ **pierde la precisión $O(h^2)$** propia "
                    "de la diferencia centrada: el error queda en $O(h)$, igual que en "
                    "una progresiva o regresiva."
                )

            num_val, y_izq, y_der, _ = fdata["compute"](ys, idx)
            deriv_val = num_val / h if kind != "centrada" else num_val / (2 * h)

            if kind == "progresiva":
                x1v = xs[idx["x0+h"]]
                st.latex(
                    rf"h = x_1 - x_0 = {_fmt(x1v)} - {_fmt(x0_val)} = {_fmt(h)}"
                )
                st.latex(
                    rf"f'(x_0) \approx \dfrac{{f({_fmt(x1v)}) - f({_fmt(x0_val)})}}{{{_fmt(h)}}}"
                    rf"= \dfrac{{{_fmt(y_der)} - {_fmt(y_izq)}}}{{{_fmt(h)}}}"
                )
            elif kind == "regresiva":
                x1v = xs[idx["x0-h"]]
                st.latex(
                    rf"h = x_0 - x_{{-1}} = {_fmt(x0_val)} - {_fmt(x1v)} = {_fmt(h)}"
                )
                st.latex(
                    rf"f'(x_0) \approx \dfrac{{f({_fmt(x0_val)}) - f({_fmt(x1v)})}}{{{_fmt(h)}}}"
                    rf"= \dfrac{{{_fmt(y_der)} - {_fmt(y_izq)}}}{{{_fmt(h)}}}"
                )
            else:
                x_mas  = xs[idx["x0+h"]]
                x_menos = xs[idx["x0-h"]]
                if simetrica:
                    st.latex(
                        rf"h = \dfrac{{({_fmt(x_mas)}) - ({_fmt(x_menos)})}}{{2}} = {_fmt(h)}"
                    )
                else:
                    st.latex(
                        rf"h = \dfrac{{(x_0 - x_-) + (x_+ - x_0)}}{{2}} = "
                        rf"\dfrac{{({_fmt(x0_val)} - {_fmt(x_menos)}) + ({_fmt(x_mas)} - {_fmt(x0_val)})}}{{2}} "
                        rf"= {_fmt(h)} \quad \text{{(asimétrico)}}"
                    )
                st.latex(
                    rf"f'(x_0) \approx \dfrac{{f({_fmt(x_mas)}) - f({_fmt(x_menos)})}}{{2 \cdot {_fmt(h)}}}"
                    rf"= \dfrac{{{_fmt(y_der)} - {_fmt(y_izq)}}}{{{_fmt(2*h)}}}"
                )

            st.latex(rf"\boxed{{\ f'({_fmt(x0_val)}) \approx {deriv_val:.6g}\ }}")

            # Error frente a f' real, si f está disponible
            error_val = None
            if f_func is not None:
                try:
                    x_sym = sp.Symbol("x")
                    expr  = sp.sympify(func_str.replace("^", "**"), locals=_local_dict()[1])
                    fprime_expr = sp.diff(expr, x_sym)
                    fprime_val  = float(fprime_expr.subs(x_sym, x0_val))
                    error_val   = abs(fprime_val - deriv_val)
                    st.latex(rf"f'_{{exacta}}({_fmt(x0_val)}) = {fprime_val:.6g}")
                    st.latex(
                        rf"\text{{Error}} = \left| f'_{{exacta}} - f'_{{aprox}} \right| "
                        rf"= |{fprime_val:.6g} - {deriv_val:.6g}| = {error_val:.6g}"
                    )
                except Exception:
                    pass

            resultados[kind] = {
                "h": h, "idx": idx, "deriv": deriv_val, "error": error_val,
                "simetrica": simetrica,
            }

            # Result card de esta fórmula
            err_html = (
                f'<div class="rc-sub">error = {error_val:.4g}</div>'
                if error_val is not None else
                '<div class="rc-sub">f\' aproximada</div>'
            )
            asim_html = ""
            if kind == "centrada" and not simetrica:
                asim_html = (
                    '<div class="rc-sub" style="opacity:.8; color:#d97706; '
                    'font-style:normal; font-weight:600;">⚠ asimétrica · O(h)</div>'
                )
            # Importante: armar el HTML en una sola línea (sin saltos de línea).
            # Si se pasa un f-string multilínea indentado a st.markdown, una
            # línea en blanco (por ejemplo cuando asim_html == "") corta el
            # bloque HTML en dos, y Markdown interpreta el resto (el </div>
            # indentado) como un bloque de código en vez de HTML.
            card_html = (
                '<div class="result-cards">'
                f'<div class="result-card" style="border-top: 3px solid {color};">'
                f'<div class="rc-label">{fdata["title"]}</div>'
                f'<div class="rc-value">{deriv_val:.6g}</div>'
                f'{err_html}'
                f'{asim_html}'
                '</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # ── Gráfico individual de esta fórmula ──────────────────────────
            try:
                pts_x  = [xs[i] for i in idx.values()]
                span_x = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
                pad    = max(span_x * 0.25, 0.5)
                x_lo   = min(xs) - pad
                x_hi   = max(xs) + pad
                x_plot = np.linspace(x_lo, x_hi, 600)

                y_nodes = np.array(ys, dtype=float)
                y_lo_ref = y_nodes.min()
                y_hi_ref = y_nodes.max()

                fig = go.Figure()

                # f(x) real, si está disponible
                if f_func is not None:
                    try:
                        y_f_raw = np.array(f_func(x_plot), dtype=float)
                        y_f_raw = np.where(np.isfinite(y_f_raw), y_f_raw, np.nan)
                        f_finite = y_f_raw[np.isfinite(y_f_raw)]
                        if len(f_finite) > 1:
                            y_lo_ref = min(y_lo_ref, float(f_finite.min()))
                            y_hi_ref = max(y_hi_ref, float(f_finite.max()))
                        fig.add_trace(go.Scatter(
                            x=x_plot, y=y_f_raw,
                            mode="lines", name="f(x)",
                            line=dict(color="#94a3b8", width=2, dash="dash"),
                        ))
                    except Exception:
                        pass

                y_rng = max(y_hi_ref - y_lo_ref, 0.5)
                y_lo  = y_lo_ref - y_rng * 0.25
                y_hi  = y_hi_ref + y_rng * 0.25

                # Todos los nodos de la tabla, de fondo
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    name="Nodos de la tabla",
                    marker=dict(color="rgba(148,163,184,0.55)", size=8,
                                line=dict(color="white", width=1)),
                ))

                # Nodos usados por esta fórmula, resaltados
                used_x = [xs[i] for i in idx.values()]
                used_y = [ys[i] for i in idx.values()]
                used_labels = [f"({_fmt(xv)}, {_fmt(yv)})" for xv, yv in zip(used_x, used_y)]
                fig.add_trace(go.Scatter(
                    x=used_x, y=used_y,
                    mode="markers+text",
                    name="Nodos usados",
                    marker=dict(color=color, size=13, symbol="circle",
                                line=dict(color="white", width=2)),
                    text=used_labels,
                    textposition="top center",
                    textfont=dict(size=11),
                ))

                # Recta secante / tangente aproximada que define la pendiente
                x_line = np.array([min(used_x) - span_x * 0.15, max(used_x) + span_x * 0.15])
                if kind == "progresiva":
                    x_base, y_base = x0_val, ys[idx["x0"]]
                elif kind == "regresiva":
                    x_base, y_base = x0_val, ys[idx["x0"]]
                else:
                    x_base, y_base = x0_val, (ys[idx["x0+h"]] + ys[idx["x0-h"]]) / 2.0
                y_line = y_base + deriv_val * (x_line - x_base)
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    name=f"Pendiente ≈ {deriv_val:.4g}",
                    line=dict(color=color, width=2.5),
                ))

                # Línea vertical en x0
                fig.add_vline(
                    x=x0_val,
                    line=dict(color="rgba(120,120,120,0.5)", width=1.5, dash="dot"),
                    annotation_text=f"x₀ = {_fmt(x0_val)}",
                    annotation_position="top right",
                    annotation_font=dict(size=12),
                )

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
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{kind}")

            except Exception as _graph_err:
                st.warning(f"No se pudo generar el gráfico de la {fdata['title'].lower()}: {_graph_err}")

        # ══════════════════════════════════════════════════════════════════════
        # RESUMEN — comparación de las 3 fórmulas
        # ══════════════════════════════════════════════════════════════════════
        validos = {k: v for k, v in resultados.items() if v is not None}
        if len(validos) >= 2:
            st.markdown("---")
            st.markdown("### Resumen comparativo")

            lines = [
                r"| Método | $h$ | $f'(x_0)$ aproximada" +
                (r" | Error" if any(v["error"] is not None for v in validos.values()) else "") +
                r" |",
                "|:---:|:---:|:---:|" + (":---:|" if any(v["error"] is not None for v in validos.values()) else ""),
            ]
            nombres = {"progresiva": "Progresiva", "regresiva": "Regresiva", "centrada": "Centrada"}
            tiene_error_col = any(v["error"] is not None for v in validos.values())
            for kind in ["progresiva", "regresiva", "centrada"]:
                if kind not in validos:
                    continue
                v = validos[kind]
                nombre_fila = nombres[kind]
                if kind == "centrada" and not v.get("simetrica", True):
                    nombre_fila += " ⚠️"
                row = f"| {nombre_fila} | {_fmt(v['h'])} | {v['deriv']:.6g} |"
                if tiene_error_col:
                    err_str = f"{v['error']:.6g}" if v["error"] is not None else "—"
                    row += f" {err_str} |"
                lines.append(row)
            st.markdown("\n".join(lines))

            if "centrada" in validos and not validos["centrada"].get("simetrica", True):
                st.caption(
                    "⚠️ La centrada de esta tabla usó vecinos **asimétricos** (no hay "
                    "ningún par de nodos equidistante de $x_0$), por lo que en este caso "
                    "su error real es $O(h)$ y no $O(h^2)$ — no se le debe dar más "
                    "crédito del que se le daría a la progresiva o regresiva."
                )
            else:
                st.caption(
                    "La diferencia **centrada** suele ser la más precisa (error de orden "
                    "$h^2$), por lo que conviene priorizarla cuando $x_0$ tiene vecinos "
                    "disponibles a ambos lados en la tabla."
                )