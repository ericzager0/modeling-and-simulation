# Simulador de Métodos Numéricos

Aplicación web interactiva construida con Streamlit para explorar métodos numéricos: Bisección, Punto Fijo, Aitken, Newton-Raphson y Lagrange.

---

## Estructura del proyecto

```
simulador-numerico/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── methods/
    ├── __init__.py
    ├── bisection.py
    ├── fixed_point.py
    ├── aitken.py
    ├── newton_raphson.py
    └── lagrange.py
```

---

## Requisitos previos

- Python 3.9 o superior
- Git

---

## 🚀 Clonar el proyecto en una PC nueva

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/simulador-numerico.git
cd simulador-numerico

# 2. Crear el entorno virtual
python3 -m venv venv

# 3. Activar el entorno virtual
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 4. Instalar las dependencias
pip install -r requirements.txt

# 5. Ejecutar la aplicación
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`

---

## 💻 Abrir el proyecto y comenzar a trabajar

Cada vez que abras una nueva terminal, debés activar el entorno virtual primero:

```bash
# Desde la raíz del proyecto
source venv/bin/activate

# Verificar que está activo (aparece "(venv)" al inicio del prompt)
# Ejecutar la app
streamlit run app.py
```

Para desactivar el entorno cuando termines:

```bash
deactivate
```

---

## ➕ Agregar una nueva librería

```bash
# 1. Asegurarte de que el entorno virtual esté activo
source venv/bin/activate

# 2. Instalar la librería
pip install nombre-libreria

# 3. Actualizar requirements.txt
pip freeze > requirements.txt

# 4. Confirmar los cambios en git
git add requirements.txt
git commit -m "feat: agrega nombre-libreria"
```

> **Importante:** siempre actualizá `requirements.txt` después de instalar algo nuevo,
> para que otros puedan replicar el entorno exacto.

---

## 🔧 Agregar un nuevo método numérico

1. Creá un archivo en `methods/`, por ejemplo `methods/secant.py`
2. Definí la función `run()` en ese archivo (seguí el patrón de `bisection.py`)
3. Agregá la página en `app.py`:

```python
PAGINAS = [..., "Secante"]   # agregá al listado

elif pagina == "Secante":
    from methods.secant import run
    run()
```

---

## 📦 Dependencias principales

| Librería   | Uso                          |
| ---------- | ---------------------------- |
| streamlit  | Interfaz web interactiva     |
| numpy      | Cálculos numéricos           |
| matplotlib | Gráficos                     |
| sympy      | Álgebra simbólica (opcional) |

---

## Git — flujo de trabajo básico

```bash
# Ver estado de cambios
git status

# Agregar archivos modificados
git add .

# Hacer un commit
git commit -m "descripción del cambio"

# Subir al repositorio remoto
git push origin main
```
