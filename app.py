import streamlit as st

from methods.aitken import run as run_aitken
from methods.bisection import run as run_bisection
from methods.cheats import run as run_cheats
from methods.edo import run as run_edo
from methods.finite_differences import run as run_finite_differences
from methods.fixed_point import run as run_fixed_point
from methods.lagrange import run as run_lagrange
from methods.montecarlo import run as run_montecarlo
from methods.newton_cotes import run as run_newton_cotes
from methods.newton_raphson import run as run_newton_raphson
from methods.theory import run as run_theory

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
    "Teoría": run_theory,
    "Bisección": run_bisection,
    "Punto Fijo": run_fixed_point,
    "Aitken": run_aitken,
    "Newton-Raphson": run_newton_raphson,
    "Lagrange": run_lagrange,
    "Diferencias Finitas": run_finite_differences,
    "Newton-Cotes": run_newton_cotes,
    "Montecarlo": run_montecarlo,
    "EDOs": run_edo,
    "Cheats": run_cheats,
}

with st.sidebar:
    pagina = st.radio("Navegación", PAGINAS, label_visibility="collapsed")

MODULOS[pagina]()
