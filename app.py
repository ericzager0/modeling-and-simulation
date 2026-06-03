import streamlit as st

st.set_page_config(page_title="Simulador Numérico", layout="wide")

# Sidebar de navegación
with st.sidebar:
    st.title("📐 Métodos Numéricos")
    pagina = st.radio("Navegar a:", [
        "Inicio",
        "Bisección",
        "Punto Fijo",
        "Aceleración de Aitken",
        "Newton-Raphson",
        "Lagrange",
    ])

# Router
if pagina == "Inicio":
    st.title("Simulador de Métodos Numéricos")
    st.markdown("Seleccioná un método desde el panel izquierdo.")

elif pagina == "Bisección":
    from methods.bisection import run
    run()