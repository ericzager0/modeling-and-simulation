import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("Método de Bisección")
    st.markdown("Encuentra raíces de f(x) = 0 dividiendo el intervalo a la mitad en cada paso.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Parámetros")
        expr = st.text_input("f(x)", value="x**3 - x - 2")
        a = st.number_input("Extremo a", value=1.0)
        b = st.number_input("Extremo b", value=2.0)
        tol = st.number_input("Tolerancia", value=1e-6, format="%.2e")
        max_iter = st.number_input("Máx. iteraciones", value=50, step=1)

    if st.button("Calcular"):
        try:
            f = lambda x: eval(expr, {"x": x, "np": np})
            if f(a) * f(b) >= 0:
                st.error("f(a) y f(b) deben tener signos opuestos.")
                return

            tabla = []
            ai, bi = a, b
            for i in range(int(max_iter)):
                mi = (ai + bi) / 2
                fmi = f(mi)
                error = abs(bi - ai) / 2
                tabla.append({"Iter": i+1, "a": ai, "b": bi, "m": mi, "f(m)": fmi, "Error": error})
                if error < tol or fmi == 0:
                    break
                if f(ai) * fmi < 0:
                    bi = mi
                else:
                    ai = mi

            raiz = tabla[-1]["m"]
            st.success(f"Raíz aproximada: **{raiz:.8f}** en {len(tabla)} iteraciones")

            with col2:
                st.subheader("Gráfico")
                xs = np.linspace(a - 0.5, b + 0.5, 400)
                ys = [f(x) for x in xs]
                fig, ax = plt.subplots()
                ax.plot(xs, ys, label="f(x)")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.axvline(raiz, color="red", linestyle="--", label=f"Raíz ≈ {raiz:.4f}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            st.subheader("Tabla de iteraciones")
            st.dataframe(tabla, use_container_width=True)

        except Exception as e:
            st.error(f"Error al evaluar: {e}")