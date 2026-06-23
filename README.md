# 📊 Predicción y análisis inteligente de mercados financieros mediante Machine Learning

¡Bienvenido al repositorio oficial del proyecto! Esta es una aplicación web interactiva diseñada para analizar y proyectar el comportamiento diario de las acciones tecnológicas más importantes de Wall Street (enfocado principalmente en **Apple - AAPL**), utilizando algoritmos de Inteligencia Artificial.

## 🧠 ¿Cómo funciona el proyecto?
El sistema utiliza un modelo de **Regresión Logística** entrenado con datos históricos. El modelo evalúa las variables del mercado para determinar la probabilidad de que el precio de la acción suba o baje al cierre del día, ayudando a tomar decisiones de *Swing Trading*.

---

## 📁 Estructura del Repositorio

* **`app.py`**: El código principal de la interfaz gráfica y frontend, desarrollado con **Streamlit**.
* **`Analisis_y_Entrenamiento.ipynb`**: El archivo de **Google Colab** que contiene toda la "magia tras bambalinas": la limpieza de datos, el Análisis Exploratorio de Datos (EDA) y el entrenamiento del modelo.
* **`modelo.pkl`**: El modelo de Machine Learning ya entrenado y exportado (serializado) listo para usarse.
* **Archivos `.csv`**: Los conjuntos de datos históricos extraídos del mercado financiero.

---

## 🛠️ Tecnologías y Herramientas Utilizadas

* **Lenguaje:** Python 3.10+
* **Ciencia de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression)
* **Despliegue e Interfaz:** Streamlit

---

## 🚀 Cómo ejecutar la aplicación localmente

Si deseas clonar este proyecto y ejecutarlo en tu computadora, sigue estos pasos en tu terminal:

1. **Instalar los requisitos:**
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```

2. **Correr la aplicación:**
   ```bash
   streamlit run app.py
   ```
*Proyecto desarrollado con fines académicos para la sustentación de la materia de Machine Learning.*
* **Autor:** [Escribe tu nombre y apellido aquí]
* **Institución:** [Nombre de tu Universidad]
