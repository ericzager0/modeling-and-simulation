import streamlit as st

st.set_page_config(page_title="Simulador Numérico", layout="wide")

st.markdown("""
    <style>
    [data-baseweb="radio"] {
        padding: 8px 0px !important;
    }
    [data-baseweb="radio"] p {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

PAGINAS = ["Inicio", "Bisección", "Punto Fijo", "Aitken", "Newton-Raphson", "Lagrange", "Newton-Cotes", "Montecarlo"]

with st.sidebar:
    pagina = st.radio("Navegación", PAGINAS, label_visibility="collapsed")

if pagina == "Inicio":
    st.title("Simulador de Métodos Numéricos")

elif pagina == "Bisección":
    from methods.bisection import run
    run()

elif pagina == "Punto Fijo":
    from methods.fixed_point import run
    run()

elif pagina == "Aitken":
    from methods.aitken import run
    run()

elif pagina == "Newton-Raphson":
    from methods.newton_raphson import run
    run()

elif pagina == "Lagrange":
    from methods.lagrange import run
    run()

elif pagina == "Newton-Cotes":
    from methods.newton_cotes import run
    run()

# Corregir que en newton cotes al entrar x**x no me dice que uso indeterminacion. y al entrar cos(x)/x muestra inf y no el valor.

elif pagina == "Montecarlo":
    from methods.montecarlo import run
    run()