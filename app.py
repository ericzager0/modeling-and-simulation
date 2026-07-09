from importlib import import_module

import streamlit as st

st.set_page_config(page_title="Simulador Numérico", layout="wide")

st.markdown(
    """
    <style>
    [data-baseweb="radio"] {
        padding: 8px 0px !important;
    }
    [data-baseweb="radio"] p {
        font-size: 18px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

PAGINAS = [
    "Teoría",
    "Bisección",
    "Punto Fijo",
    "Aitken",
    "Newton-Raphson",
    "Lagrange",
    "Diferencias Finitas",
    "Newton-Cotes",
    "Montecarlo",
    "EDOs",
    "Cheats",
]

MODULOS = {
    "Teoría": "methods.theory",
    "Bisección": "methods.bisection",
    "Punto Fijo": "methods.fixed_point",
    "Aitken": "methods.aitken",
    "Newton-Raphson": "methods.newton_raphson",
    "Lagrange": "methods.lagrange",
    "Diferencias Finitas": "methods.finite_differences",
    "Newton-Cotes": "methods.newton_cotes",
    "Montecarlo": "methods.montecarlo",
    "EDOs": "methods.edo",
    "Cheats": "methods.cheats",
}

with st.sidebar:
    pagina = st.radio("Navegación", PAGINAS, label_visibility="collapsed")

import_module(MODULOS[pagina]).run()
