import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os

# =====================================================================
# 1. FUNCIÓN PARA CARGAR EL MODELO (PICKLE)
# =====================================================================
@st.cache_resource
def cargar_modelo(nombre_empresa):
    # Primero intenta buscar un modelo específico por empresa
    ruta_specifica = f"modelo_{nombre_empresa}.pkl"
    if os.path.exists(ruta_specifica):
        with open(ruta_specifica, "rb") as f:
            return pickle.load(f)
            
    # Como renombraste tu archivo, aquí buscará con éxito "modelo.pkl"
    ruta_general = "modelo.pkl"
    if os.path.exists(ruta_general):
        with open(ruta_general, "rb") as f:
            return pickle.load(f)
            
    return None

# =====================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="FAANG Predictor",
    layout="wide"
)

st.sidebar.title("⚙️ Configuración")

empresa = st.sidebar.selectbox(
    "Seleccionar empresa",
    ["Google", "Apple", "Amazon", "Netflix", "Facebook"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Análisis histórico, indicadores técnicos y modelos de Machine Learning para la toma de decisiones financieras."
)

st.caption(
    "Análisis histórico, indicadores técnicos y modelos de Machine Learning para la toma de decisiones financieras."
)

# Título Principal
st.title(f"📊 Predicción y análisis inteligente de mercados financieros mediante Machine Learning | {empresa}")

# =====================================================================
# 3. CARGAR Y PROCESAR LOS DATOS (CÁLCULO DE TU MATRIZ X)
# =====================================================================
archivos = {
    "Google": "Google.csv",
    "Apple": "Apple.csv",
    "Amazon": "Amazon.csv",
    "Netflix": "Netflix.csv",
    "Facebook": "Facebook.csv"
}

# Leer el archivo seleccionado por el usuario
df = pd.read_csv(archivos[empresa])
df["Date"] = pd.to_datetime(df["Date"])

# --- CÁLCULO MATEMÁTICO DE TUS 10 VARIABLES DE ENTRADA ---
# Medias Móviles Simples (MA)
df["MA7"] = df["Close"].rolling(7).mean()
df["MA30"] = df["Close"].rolling(30).mean()

# Medias Móviles Exponenciales (EMA)
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

# Retornos diarios (Return)
df["Return"] = df["Close"].pct_change()

# Indicador de Fuerza Relativa (RSI de 14 períodos)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Limpiamos las filas vacías generadas al inicio por los cálculos matemáticos
df_limpio = df.dropna().copy()


# =====================================================================
# 4. MOSTRAR INFORMACIÓN EN LA PANTALLA
# =====================================================================
st.subheader("Información financiera")

col1, col2, col3 = st.columns(3)
precio_actual = round(df["Close"].iloc[-1], 2)
precio_max = round(df["Close"].max(), 2)
precio_min = round(df["Close"].min(), 2)

with col1:
    st.metric("Precio Actual", f"${precio_actual}")
with col2:
    st.metric("Máximo Histórico", f"${precio_max}")
with col3:
    st.metric("Mínimo Histórico", f"${precio_min}")

# Tabla de datos recientes
st.subheader("Vista previa de los datos")
st.dataframe(df.tail(20), use_container_width=True)

# Gráfico interactivo de Plotly
st.subheader("📈 Evolución Histórica del Precio")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Precio"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA7"], mode="lines", name="MA7"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA30"], mode="lines", name="MA30"))

fig.update_layout(
    height=600,
    template="plotly_dark",
    xaxis_title="Fecha",
    yaxis_title="Precio",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# 5. BLOQUE MÁGICO: AQUÍ TRABAJA EL PICKLE Y DA LA ALZA/BAJA
# =====================================================================
st.subheader("🔮 Predicción del Modelo para Mañana")

# Llamamos a la función para abrir el archivo .pkl
modelo_ia = cargar_modelo(empresa)

if modelo_ia is not None:
    try:
        # Tus 10 columnas obligatorias en el orden exacto de tu matriz X
        columnas_entrenamiento = [
            'Open', 'High', 'Low', 'Volume', 
            'MA7', 'MA30', 'Return', 'RSI', 
            'EMA20', 'EMA50'
        ]
        
        # Extraemos el último día de datos que calculamos arriba para alimentar al modelo
        datos_hoy = df_limpio[columnas_entrenamiento].iloc[-1].values.reshape(1, -1)
        
        # El modelo hace la predicción
        prediccion = modelo_ia.predict(datos_hoy)[0]
        
        # Evaluamos si dio 1 (Alza) o 0 (Baja)
        es_alza = (prediccion == 1) or (str(prediccion).strip().lower() in ["alza", "sube", "up", "1"])

        # Mostramos la respuesta bonita en la pantalla
        if es_alza:
            st.success("### 🚀 PROYECCIÓN: ALZA (BULLISH)")
            st.write("El modelo analiza los indicadores técnicos y proyecta una tendencia **alcista** para la próxima jornada.")
        else:
            st.error("### 📉 PROYECCIÓN: BAJA (BEARISH)")
            st.write("El modelo analiza los indicadores técnicos y proyecta una tendencia **bajista** para la próxima jornada.")

        # Métricas de control abajo de la predicción
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Algoritmo Utilizado", "Logistic Regression")
        with col_res2:
            try:
                # Intentamos calcular qué tan seguro está el modelo (probabilidad)
                probabilidad = max(modelo_ia.predict_proba(datos_hoy)[0]) * 100
                st.metric("Confianza del Modelo", f"{probabilidad:.1f}%")
            except AttributeError:
                st.metric("Estado del Modelo", "Conectado exitosamente")

    except Exception as e:
        st.error(f"❌ Error al procesar los datos para el modelo: {e}")
else:
    st.warning(f"⚠️ No se encontró el archivo del modelo. Asegúrate de que `modelo.pkl` esté guardado en la carpeta raíz de tu proyecto.")