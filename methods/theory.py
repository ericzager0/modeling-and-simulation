import streamlit as st
import sympy as sp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_str(expr, val, decimals=4):
    """Evalúa una expresión sympy en un punto y devuelve el float redondeado."""
    return float(expr.subs(sp.Symbol("x"), val))


def _signo_color(valor: float) -> str:
    return "#16a34a" if valor > 0 else "#dc2626"


# ─────────────────────────────────────────────────────────────────────────────
# Sección: Teorema de Bolzano
# ─────────────────────────────────────────────────────────────────────────────

def _bolzano():
    st.header("Teorema de Bolzano")

    st.markdown("""
    El **Teorema de Bolzano** permite garantizar la existencia de al menos
    una raíz de una función en un intervalo, sin necesidad de calcularla
    explícitamente.
    """)

    st.subheader("Enunciado")
    st.info("""
    Si $f$ es **continua** en $[a, b]$ y $f(a) \\cdot f(b) < 0$
    (es decir, $f(a)$ y $f(b)$ tienen **signos opuestos**), entonces existe
    al menos un punto $c \\in (a, b)$ tal que $f(c) = 0$.
    """)

    st.markdown("""
    En otras palabras: si una función continua cambia de signo entre dos
    puntos, en algún lugar entre ellos **debe** cruzar el cero. Es un
    resultado de **existencia**, no de unicidad: puede haber una o varias
    raíces dentro del intervalo, el teorema solo asegura que hay al menos una.
    """)

    st.markdown("**Condiciones necesarias:**")
    st.markdown("""
    1. $f$ continua en $[a, b]$.
    2. $f(a)$ y $f(b)$ con signos opuestos, es decir $f(a) \\cdot f(b) < 0$.
    """)

    st.divider()

    # ── Ejemplo paso a paso ──────────────────────────────────────────────────
    st.subheader("Ejemplo paso a paso")

    st.markdown("Sea la función:")
    x = sp.Symbol("x")
    expr = 2 * x * sp.cos(x) - (x - 2) ** 2
    st.latex(rf"f(x) = {sp.latex(expr)}")

    st.markdown("Queremos demostrar que existe al menos una raíz en el intervalo:")
    a, b = 0.8, 1.4
    st.latex(rf"[a,\;b] = [{a},\;{b}]")

    st.markdown("#### Paso 1 — Continuidad")
    st.markdown("""
    $f(x)$ es una combinación de un polinomio y la función $\\cos(x)$,
    ambas continuas en todo $\\mathbb{R}$. Por lo tanto $f$ es continua en
    $[0.8,\\,1.4]$. **Se cumple la primera condición.**
    """)

    st.markdown("#### Paso 2 — Evaluar en los extremos")

    fa = _eval_str(expr, a)
    fb = _eval_str(expr, b)

    st.latex(
        rf"f({a}) = 2({a})\cos({a}) - ({a}-2)^2 = {fa:.4f}"
    )
    st.latex(
        rf"f({b}) = 2({b})\cos({b}) - ({b}-2)^2 = {fb:.4f}"
    )

    st.markdown(f"""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid {_signo_color(fa)};">
            <div class="rc-label">f({a})</div>
            <div class="rc-value">{fa:.4f}</div>
            <div class="rc-sub">{"positivo" if fa > 0 else "negativo"}</div>
        </div>
        <div class="result-card" style="border-top: 3px solid {_signo_color(fb)};">
            <div class="rc-label">f({b})</div>
            <div class="rc-value">{fb:.4f}</div>
            <div class="rc-sub">{"positivo" if fb > 0 else "negativo"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Paso 3 — Producto de signos")
    prod = fa * fb
    st.latex(
        rf"f({a}) \cdot f({b}) = ({fa:.4f})({fb:.4f}) = {prod:.4f}"
    )

    if prod < 0:
        st.success(
            f"**Bolzano se cumple:** $f({a}) \\cdot f({b}) = {prod:.4f} < 0$ "
            "→ los valores tienen signos opuestos."
        )
    else:
        st.error(
            f"**Bolzano no se cumple:** $f({a}) \\cdot f({b}) = {prod:.4f} \\geq 0$"
        )

    st.markdown("#### Conclusión")
    st.markdown(f"""
    Como $f$ es continua en $[{a}, {b}]$ y $f({a}) \\cdot f({b}) < 0$,
    el **Teorema de Bolzano** garantiza que existe al menos un
    $c \\in ({a}, {b})$ tal que $f(c) = 0$.

    Es decir, la función tiene **al menos una raíz** en ese intervalo. ✅
    """)

    st.caption(
        "Nota: Bolzano solo garantiza existencia, no unicidad ni el valor "
        "exacto de la raíz. Para aproximarla, usá un método numérico como "
        "Bisección o Newton-Raphson."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sección: Radianes vs. Grados
# ─────────────────────────────────────────────────────────────────────────────

def _radianes_vs_grados():
    st.header("Radianes vs. Grados")

    st.markdown("""
    Cuando trabajamos con funciones trigonométricas como $\\sin(x)$ o
    $\\cos(x)$ dentro de un método numérico (Bolzano, Bisección,
    Newton-Raphson, etc.), es fundamental saber en qué **unidad angular**
    se está evaluando $x$, porque el resultado cambia por completo según
    la convención usada.
    """)

    st.subheader("¿Qué es un radián?")
    st.info("""
    Un **radián** es el ángulo central que abarca, sobre una circunferencia,
    un arco de longitud igual al radio. Es una unidad basada en la
    geometría del círculo, a diferencia del grado, que divide la vuelta
    en 360 partes arbitrarias.
    """)

    st.markdown("Como el perímetro de una circunferencia es $2\\pi r$, una vuelta completa equivale a:")
    st.latex(r"360^\circ = 2\pi \text{ rad}")
    st.markdown("De donde se obtienen las conversiones:")
    st.latex(r"180^\circ = \pi \text{ rad} \approx 3.1416 \text{ rad}")
    st.latex(r"1 \text{ rad} = \frac{180^\circ}{\pi} \approx 57.3^\circ")

    st.divider()

    st.subheader("¿Por qué importa al evaluar una función?")
    st.markdown("""
    El mismo número $x$ representa **ángulos distintos** según la unidad
    que se use, y por lo tanto $\\cos(x)$ da resultados diferentes:
    """)

    x = sp.Symbol("x")
    val = 0.8
    cos_rad = float(sp.cos(val))
    cos_deg = float(sp.cos(sp.rad(val)))  # 0.8° convertido a radianes

    st.latex(rf"\cos({val}\text{{ rad}}) \approx {cos_rad:.4f}")
    st.latex(rf"\cos({val}^\circ) \approx {cos_deg:.4f}")

    st.markdown(f"""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid #2563eb;">
            <div class="rc-label">cos({val} rad)</div>
            <div class="rc-value">{cos_rad:.4f}</div>
            <div class="rc-sub">convención matemática estándar</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #d97706;">
            <div class="rc-label">cos({val}°)</div>
            <div class="rc-value">{cos_deg:.4f}</div>
            <div class="rc-sub">ángulo casi nulo → coseno ≈ 1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Son números completamente distintos. Si en un ejercicio de Bolzano o
    Bisección se evaluara $f(x)$ usando la convención equivocada,
    el signo de $f(a)$ o $f(b)$ podría cambiar, y con él la validez de
    toda la demostración.
    """)

    st.divider()

    st.subheader("Convención usada en este simulador")
    st.success("""
    Tanto en **matemática y análisis** como en **Python** (`sympy`,
    `numpy`, `math`), las funciones trigonométricas asumen que el
    argumento está en **radianes** por defecto. Es el estándar universal:
    las fórmulas de derivadas, como $\\dfrac{d}{dx}\\sin(x) = \\cos(x)$,
    solo son válidas si $x$ está en radianes.
    """)

    st.markdown("""
    Por eso, en todos los métodos de este simulador (Bisección,
    Newton-Raphson, Punto Fijo, etc.), cuando ingresás una función con
    `sin(x)`, `cos(x)` o `tan(x)`, el valor de `x` siempre se interpreta
    en **radianes**, sin necesidad de hacer ninguna conversión manual.
    """)

    st.caption(
        "Tip: si alguna vez necesitás convertir un ángulo dado en grados "
        "a radianes para usarlo en una función, la fórmula es "
        "rad = grados · π / 180."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sección: Construcción de g(x) para Punto Fijo / Aitken-Steffensen
# ─────────────────────────────────────────────────────────────────────────────

def _construccion_g():
    st.header("Cómo hallar la función auxiliar g(x)")

    st.markdown("""
    Los métodos de **Punto Fijo** (y sus aceleraciones, como
    **Aitken** o **Steffensen**) no trabajan directamente con $f(x)=0$,
    sino con una reformulación:
    """)

    st.latex(r"f(x) = 0 \quad \Longleftrightarrow \quad x = g(x)")

    st.markdown("""
    Si encontramos una función $g$ tal que la raíz de $f$ sea un
    **punto fijo** de $g$ (es decir, $g(x^{*}) = x^{*}$), entonces podemos
    aproximar la raíz iterando:
    """)
    st.latex(r"x_{n+1} = g(x_n), \qquad n = 0, 1, 2, \dots")

    st.markdown("""
    El problema real de este método **no es iterar** — eso es mecánico —
    sino **encontrar una $g(x)$ que funcione**. No cualquier despeje
    converge. Esta sección es una receta para encontrarla.
    """)

    st.divider()

    # ── Receta general ───────────────────────────────────────────────────────
    st.subheader("La idea: despejar f(x) = 0 como x = algo")

    st.markdown("""
    La forma más natural de pensarlo (como seguramente lo ves en tu curso)
    es: tomar la ecuación $f(x) = 0$ y manipularla algebraicamente hasta
    dejar una $x$ sola de un lado. Ese "algo" del otro lado **es** $g(x)$.
    """)

    st.markdown("""
    El problema es que, salvo en ecuaciones muy simples, **no hay una
    única manera de despejar**. Una misma $f(x)=0$ admite varias
    reordenaciones algebraicas válidas, y cada una da una $g(x)$
    distinta. Lo que cambia entre ellas no es la raíz (todas comparten
    la misma raíz que $f$), sino **si la iteración converge o no** al
    iterar cerca de la semilla elegida.
    """)

    st.markdown("**Por eso la receta tiene siempre dos partes:**")
    st.markdown("""
    1. Generar **varias candidatas** $g_1(x), g_2(x), g_3(x), \\dots$
       despejando de formas distintas.
    2. **Probar cada una** cerca de la semilla $x_0$ hasta encontrar
       una que cumpla la condición de convergencia (Lipschitz / derivada
       acotada — lo vemos abajo).
    """)

    st.divider()

    # ── Formas tipicas de despejar ───────────────────────────────────────────
    st.subheader("Formas típicas de despejar x = g(x)")

    st.markdown("**1. Aislar el término que contiene la potencia más alta o el radical**")
    st.markdown("""
    Si $f(x)$ tiene un término como $x^2$, $\\sqrt{x}$, o similar,
    se despeja ese término y se aplica la operación inversa
    (raíz cuadrada, potencia, etc.) a ambos lados.
    """)

    st.markdown("**2. Aislar el término trascendente (trigonométrico, exponencial, logarítmico)**")
    st.markdown("""
    Si $f(x)$ mezcla un polinomio con $\\sin(x)$, $\\cos(x)$, $e^x$ o
    $\\ln(x)$, conviene despejar ese término y aplicar la función
    inversa correspondiente ($\\arcsin$, $\\arccos$, $\\ln$, $e^{(\\cdot)}$).
    """)

    st.markdown("**3. El truco general: sumar x a ambos lados**")
    st.markdown("""
    Cuando no hay una forma algebraica "limpia" de despejar (muy común
    cuando $f$ mezcla varios tipos de funciones), siempre existe esta
    alternativa universal:
    """)
    st.latex(r"x = x + f(x)")
    st.markdown("""
    o, más general, multiplicando $f(x)$ por una constante de relajación
    $\\lambda \\neq 0$ (suele ayudar a controlar la convergencia):
    """)
    st.latex(r"x = x + \lambda\, f(x) = g(x)")
    st.markdown("""
    Esta forma siempre es matemáticamente válida (si $f(x)=0$ entonces
    $g(x)=x$), pero **no garantiza convergencia** — hay que verificar
    Lipschitz igual que con cualquier otra candidata, y a veces ningún
    $\\lambda$ razonable funciona bien cerca de la semilla.
    """)

    st.markdown("**4. Cuidado al despejar raíces: el valor absoluto**")
    st.markdown("""
    Cuando en $f(x)=0$ aparece un término elevado al cuadrado, como
    $(x-2)^2$, y se aplica raíz cuadrada a ambos lados, hay que recordar
    que:
    """)
    st.latex(r"\sqrt{(x-2)^2} = |x-2|")
    st.markdown("""
    **no** simplemente $x-2$. El valor absoluto se abre de dos formas
    distintas según el signo de $x-2$ en la zona donde se busca la raíz:
    """)
    st.latex(r"""
    |x-2| =
    \begin{cases}
    x-2, & \text{si } x \geq 2 \\[4pt]
    2-x, & \text{si } x < 2
    \end{cases}
    """)
    st.markdown("""
    Por eso, si $h(x)^{1/2} = |x-2|$ y se quiere despejar $x$, conviene
    primero estimar (por ejemplo, con Bolzano) si la raíz buscada cae
    en una zona con $x<2$ o $x>2$, **antes** de elegir el signo:
    """)
    st.markdown("""
    - Si la raíz está en una zona con $x \\geq 2$: se usa $x = 2+\\sqrt{h(x)}$
    - Si la raíz está en una zona con $x < 2$: se usa $x = 2-\\sqrt{h(x)}$
    """)
    st.markdown("""
    Elegir el signo equivocado no es solo un detalle estético: produce
    una $g(x)$ que apunta hacia un punto fijo completamente distinto
    (o que ni siquiera existe en los reales), aun cuando la derivada en
    la semilla pueda dar engañosamente bien. Lo vemos en el ejemplo
    aplicado más abajo.
    """)

    st.divider()

    # ── Condicion de Lipschitz ───────────────────────────────────────────────
    st.subheader("Cómo verificar que una g(x) sirve")

    st.markdown("""
    No toda $g(x)$ que despejemos hace que la iteración converja, y
    **mirar solo la derivada no alcanza** — más abajo vemos un caso
    concreto donde una candidata cumple $|g'(x_0)|<1$ y sin embargo es
    inútil. El chequeo completo tiene **tres pasos**, en este orden:
    """)

    st.markdown("**1. Dominio — ¿$g(x_0)$ existe y es un número real?**")
    st.markdown("""
    Si $g$ tiene raíces, logaritmos o arcos, hay que evaluar $g(x_0)$ y
    confirmar que no aparece una raíz de un número negativo, un
    logaritmo de un número no positivo, o un argumento fuera del rango
    de $\\arcsin$/$\\arccos$ (que es $[-1,1]$).
    """)

    st.markdown("**2. Cercanía razonable — ¿$g(x_0)$ queda cerca de $x_0$?**")
    st.markdown("""
    La condición de Lipschitz garantiza convergencia *si* ya estás cerca
    de un punto fijo. Si el primer salto $g(x_0)$ te aleja muchísimo de
    $x_0$ (y de la zona donde Bolzano te dijo que está la raíz), es una
    señal de alarma, aunque la derivada salga bien — probablemente
    estés cerca del comportamiento de $g$ en una región sin relación
    con la raíz que buscás.
    """)

    st.markdown("**3. Derivada — ¿$|g'(x_0)| < 1$?**")
    st.success(r"""
    **Derivar $g(x)$ y evaluar su módulo en la semilla $x_0$:**
    $$|g'(x_0)| < 1$$
    Si se cumple (y los dos puntos anteriores también), $g$ es una
    contracción cerca de $x_0$ (condición de Lipschitz local) y la
    iteración de punto fijo **converge**.
    """)

    st.caption("""
    Nota: formalmente, Lipschitz es una condición sobre un conjunto
    compacto, no sobre un solo punto, y los pasos 1 y 2 son justamente
    lo que garantiza que ese conjunto compacto exista alrededor de
    $x_0$ y que $g$ se quede dentro de él. Si se quiere ser más
    riguroso, se puede extender el chequeo de la derivada a varios
    puntos de un entorno de $x_0$, no solo el punto exacto. Pero a
    fines prácticos del curso, los tres pasos sobre la semilla son
    el criterio que se usa para decidir si la candidata sirve.
    """)

    st.divider()

    # ── Ejemplo aplicado ──────────────────────────────────────────────────────
    st.subheader("Ejemplo aplicado")

    x = sp.Symbol("x", real=True)
    f = 2 * x * sp.cos(x) - (x - 2) ** 2

    st.markdown("Retomamos la función del ejemplo de Bolzano, con raíz cerca de $x_0 = 1$:")
    st.latex(rf"f(x) = {sp.latex(f)} = 0")

    st.markdown("#### Paso 1 — Generar candidatas despejando x")

    st.markdown("**Candidata A** — aislando el término cuadrático, con $x<2$ cerca de la semilla:")
    st.latex(r"2x\cos(x) = (x-2)^2 \;\Rightarrow\; \sqrt{2x\cos(x)} = |x-2| = 2-x \;\Rightarrow\; x = 2 - \sqrt{2x\cos(x)}")
    gA = 2 - sp.sqrt(2 * x * sp.cos(x))

    st.markdown("**Candidata B** — aislando el coseno y aplicando $\\arccos$:")
    st.latex(r"\cos(x) = \frac{(x-2)^2}{2x} \;\Rightarrow\; x = \arccos\!\left(\frac{(x-2)^2}{2x}\right)")
    gB = sp.acos((x - 2) ** 2 / (2 * x))

    st.markdown("**Candidata C** — truco general $x = x + \\lambda f(x)$, con $\\lambda = 1$:")
    st.latex(r"x = x + 2x\cos(x) - (x-2)^2")
    gC = x + f

    st.markdown("**Candidata D** — la misma raíz que A, pero con el signo equivocado ($+2$):")
    st.latex(r"\sqrt{2x\cos(x)} = x-2 \;\Rightarrow\; x = 2 + \sqrt{2x\cos(x)}")
    st.markdown("""
    Este sería el error típico de olvidar el valor absoluto al sacar la
    raíz: como cerca de $x_0=1$ se tiene $x<2$, el signo correcto es
    "$-$" (Candidata A), no "$+$". Lo incluimos a propósito para ver
    qué pasa si se usa el signo equivocado.
    """)
    gD = 2 + sp.sqrt(2 * x * sp.cos(x))

    st.markdown("#### Paso 2 — Derivar cada candidata y evaluar |g'(x₀)| en x₀ = 1")

    gAp, gBp, gCp, gDp = sp.diff(gA, x), sp.diff(gB, x), sp.diff(gC, x), sp.diff(gD, x)
    x0 = 1.0

    gAp0 = abs(float(gAp.subs(x, x0)))
    gBp0 = abs(float(gBp.subs(x, x0)))
    gCp0 = abs(float(gCp.subs(x, x0)))
    gDp0 = abs(float(gDp.subs(x, x0)))
    gD0 = float(gD.subs(x, x0))

    st.latex(rf"|g_A'(1)| \approx {gAp0:.4f} \;\Rightarrow\; < 1 \;\; ✅")
    st.latex(rf"|g_B'(1)| \approx {gBp0:.4f} \;\Rightarrow\; > 1 \;\; ❌")
    st.latex(rf"|g_C'(1)| \approx {gCp0:.4f} \;\Rightarrow\; > 1 \;\; ❌")
    st.latex(rf"|g_D'(1)| \approx {gDp0:.4f} \;\Rightarrow\; < 1 \;\; \text{{¿✅?}}")

    st.warning(rf"""
    **Atención con la Candidata D — acá se ve por qué no alcanza con
    la derivada.** Su derivada da $|g_D'(1)| \approx {gDp0:.4f} < 1$ —
    el mismo valor que A, porque $g_D$ y $g_A$ son la misma expresión
    con una constante sumada o restada ($+2$ vs. $-2$), y la derivada
    de una constante es cero: **el paso 3 (derivada) sale igual para
    las dos, así que por sí solo no distingue cuál es la correcta.**

    Por eso hace falta el paso 2: ¿$g_D(x_0)$ queda cerca de $x_0$?
    Acá $g_D(1) \\approx {gD0:.4f}$, lejísimos de $x_0=1$ y de la zona
    de la raíz real (que Bolzano ubicó entre 0.8 y 1.4). Ese primer
    salto ya es la señal de alarma. Y si se itera igual, en el paso 1
    se confirma el problema: el siguiente valor de $x$ cae en una zona
    donde $\\cos(x) < 0$ (a partir de $x \\approx 1.57$), así que
    $2x\\cos(x)$ es negativo, la raíz cuadrada deja de ser real, y la
    iteración **se rompe** directamente.
    """)

    st.success("""
    **La Candidata A es la que sirve.** Cumple los tres pasos: $g_A(1)$
    existe y es real, queda cerca de $x_0$, y además $|g_A'(1)| < 1$.
    """)

    st.markdown("#### Paso 3 — g(x) elegida")
    st.latex(r"g(x) = 2 - \sqrt{2x\cos(x)}")
    st.latex(rf"g'(1) \approx {gAp0:.4f} \quad (|g'(1)| < 1 \;\Rightarrow\; \text{{contracción en }} x_0)")

    st.caption("""
    Nota: que B y C no funcionen acá no significa que sean "candidatas
    inválidas" en general — podrían servir con otra semilla u otro
    intervalo. La Candidata D es distinta: no es que "sirva para otra
    raíz en otra zona" — su dominio real se corta antes de llegar a
    $x=2$ (porque $\\cos(x)$ ya es negativo desde $x\\approx 1.57$), así
    que esa rama de la raíz cuadrada no tiene ningún punto fijo real
    accesible. Por eso el paso de "probar varias y verificar" es
    indispensable, no opcional — y verificar implica los tres pasos:
    dominio, cercanía a la semilla, y derivada.
    """)

    st.caption("""
    Con g(x) ya validada, se puede aplicar directamente la iteración de
    Punto Fijo, o acelerarla con Aitken (extrapolación Δ²) o
    Steffensen (que combina Aitken con la propia iteración de punto
    fijo en cada paso, sin necesidad de calcular derivadas de f).
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Sección: Tolerancia y cifras de precisión
# ─────────────────────────────────────────────────────────────────────────────

def _tolerancia():
    st.header("Tolerancia (ε) y cifras de precisión")

    st.markdown("""
    Cuando un enunciado pide "n cifras de precisión", hay que traducirlo
    al valor de **Tolerancia (ε)** que se ingresa en el simulador.
    La relación es directa:
    """)

    st.latex(r"n \text{ cifras de precisión} \quad \Longrightarrow \quad \varepsilon = 10^{-n}")

    st.markdown("**Tabla de conversión rápida:**")
    st.markdown("""
    | Cifras pedidas | Potencia de 10 | Tolerancia ε (decimal) |
    |:---:|:---:|:---:|
    | 2 cifras | $10^{-2}$ | 0.01 |
    | 3 cifras | $10^{-3}$ | 0.001 |
    | 4 cifras | $10^{-4}$ | 0.0001 |
    | 5 cifras | $10^{-5}$ | 0.00001 |
    | 6 cifras | $10^{-6}$ | 0.000001 |
    """)

    st.markdown("""
    **Truco rápido:** contá **todos los dígitos** después del punto,
    hasta llegar al primer dígito distinto de cero (inclusive) — esa
    posición es $n$. *No es lo mismo que "contar ceros":* en `0.001`
    hay dos ceros, pero el 1 cae en la 3ª posición → $n=3$.
    """)
    st.latex(r"0.000001 \;\rightarrow\; \text{el primer 1 está en la 6ª posición} \;\rightarrow\; 10^{-6} \;\rightarrow\; 6 \text{ cifras}")

    st.success("""
    Para tu ejercicio (6 cifras de precisión, raíz cerca de 1):
    ingresá **ε = 0.000001** en el campo de Tolerancia del simulador.
    """)

    st.caption("""
    Nota: esto equivale a 6 cifras *significativas* solo cuando el
    resultado tiene magnitud cercana a 1 (como en este caso). Si la raíz
    fuera mucho más chica (ej. 0.003...), contar ceros decimales ya no
    es lo mismo que contar cifras significativas.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Sección: Comparativa de métodos
# ─────────────────────────────────────────────────────────────────────────────

def _comparativa_metodos():
    st.header("Bisección vs. Punto Fijo vs. Aitken vs. Newton-Raphson")

    st.markdown("""
    Los cuatro métodos, **si convergen**, llegan a la misma raíz. Lo que
    los diferencia es **qué tan rápido llegan**, **qué tan exigentes son
    para arrancar**, y **qué tan fácil es plantearlos**.
    """)

    st.divider()

    # ── Tabla comparativa ────────────────────────────────────────────────────
    st.subheader("Resumen")

    st.markdown("""
    | | Bisección | Punto Fijo | Aitken | Newton-Raphson |
    |---|:---:|:---:|:---:|:---:|
    | **Velocidad** | 🐢 lenta | 🐢 lenta-media | 🐇 media-rápida | 🚀 rápida |
    | **Orden de convergencia** | Lineal | Lineal | Superlineal | Cuadrática |
    | **Dificultad** | 🟢 baja | 🟠 media-alta | 🟠 media-alta | 🟡 media |
    | **Qué necesita para arrancar** | $[a,b]$ con Bolzano | $x_0$ + $g(x)$ válida | $x_0$ + $g(x)$ válida | $x_0$ + $f'(x_0)\\neq 0$ |
    | **Garantía de convergencia** | Sí, si hay Bolzano | Solo si $\\|g'(x_0)\\|<1$ | Igual que Punto Fijo | No garantizada |
    """)

    st.divider()

    # ── Velocidad de convergencia ────────────────────────────────────────────
    st.subheader("1. Velocidad de convergencia")

    st.markdown("""
    Es el criterio que más las distingue. De más lenta a más rápida:
    """)

    st.markdown("""
    <div class="result-cards">
        <div class="result-card" style="border-top: 3px solid #dc2626;">
            <div class="rc-label">Bisección</div>
            <div class="rc-value">Lineal</div>
            <div class="rc-sub">error ÷ 2 en cada paso, siempre igual</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #d97706;">
            <div class="rc-label">Punto Fijo</div>
            <div class="rc-value">Lineal</div>
            <div class="rc-sub">depende de |g'(x₀)|</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #2563eb;">
            <div class="rc-label">Aitken</div>
            <div class="rc-value">Superlineal</div>
            <div class="rc-sub">acelera a Punto Fijo</div>
        </div>
        <div class="result-card" style="border-top: 3px solid #16a34a;">
            <div class="rc-label">Newton-Raphson</div>
            <div class="rc-value">Cuadrática</div>
            <div class="rc-sub">cifras correctas se duplican</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    - **Bisección**: el error se parte a la mitad en cada iteración,
      sin importar la función. Ritmo fijo y predecible, pero conservador.
    - **Punto Fijo**: también lineal, pero la velocidad **no es fija**:
      cuanto más chico sea $|g'(x_0)|$, más rápido converge; cerca de 1,
      se vuelve casi tan lento como Bisección.
    - **Aitken**: toma la misma sucesión de Punto Fijo y la acelera por
      extrapolación (Δ²), sin pedir información extra.
    - **Newton-Raphson**: la más rápida cuando funciona — cerca de la
      raíz, la cantidad de cifras decimales correctas se **duplica**
      en cada paso.
    """)

    st.divider()

    # ── Precision ─────────────────────────────────────────────────────────────
    st.subheader("2. Precisión: cuántas iteraciones hacen falta")

    st.info("""
    **Ojo con la palabra "precisión":** los cuatro métodos, si convergen,
    llegan a la **misma raíz exacta** (dentro de la tolerancia pedida).
    Ninguno es "más preciso" que otro en el resultado final — la
    diferencia real es **cuántas iteraciones** consume cada uno para
    llegar a esa tolerancia.
    """)

    st.markdown("""
    - **Bisección** necesita, por lejos, más iteraciones que el resto
      para el mismo número de cifras.
    - **Punto Fijo** varía mucho según la $g(x)$ elegida: puede ser casi
      tan lento como Bisección, o razonablemente rápido.
    - **Aitken** típicamente recorta a la mitad (o menos) las
      iteraciones que Punto Fijo necesitaría solo.
    - **Newton-Raphson** suele llegar primero a una tolerancia exigente,
      salvo que la semilla esté mal elegida.
    """)

    st.divider()

    # ── Dificultad ────────────────────────────────────────────────────────────
    st.subheader("3. Dificultad de aplicación")

    st.markdown("""
    - **Bisección** — la más simple: solo hay que verificar Bolzano en
      un intervalo. No se deriva ni se despeja nada.
    - **Newton-Raphson** — dificultad media: no hay que despejar $x$,
      pero sí calcular $f'(x)$ y elegir una semilla con $f'(x_0)\\neq 0$.
    - **Punto Fijo y Aitken** — las más laboriosas para arrancar: hay
      que **encontrar** una $g(x)$ despejando $f(x)=0$ y **verificar**
      que sea una contracción ($|g'(x_0)|<1$) antes de poder confiar en
      la iteración. *(Ver la sección "Construcción de g(x)" para el
      procedimiento completo.)*
    """)

    st.divider()

    st.subheader("En una frase")
    st.success("""
    **Bisección** es la más segura pero la más lenta. **Newton-Raphson**
    es la más rápida pero no garantiza convergencia. **Punto Fijo** y
    **Aitken** quedan en el medio: exigen más trabajo previo (hallar y
    validar $g(x)$), y Aitken simplemente acelera lo que Punto Fijo ya
    hace.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Página
# ─────────────────────────────────────────────────────────────────────────────

def run():
    st.markdown("""
    <style>
    /* ── Tamaño base de letra (Streamlit default: 16px) ── */
    html { font-size: 20px; }

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

    st.title("Teoría")

    TEMAS = ["Teorema de Bolzano", "Radianes vs. Grados", "Construcción de g(x) para Punto Fijo", "Tolerancia y cifras de precisión", "Comparativa de métodos para hallar raíz"]
    tema = st.selectbox("Elegí un tema", TEMAS)

    st.divider()

    if tema == "Teorema de Bolzano":
        _bolzano()
    elif tema == "Radianes vs. Grados":
        _radianes_vs_grados()
    elif tema == "Construcción de g(x) para Punto Fijo":
        _construccion_g()
    elif tema == "Tolerancia y cifras de precisión":
        _tolerancia()
    elif tema == "Comparativa de métodos para hallar raíz":
        _comparativa_metodos()