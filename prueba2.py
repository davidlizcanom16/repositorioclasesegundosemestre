import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import pickle
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer


st.set_page_config(page_title="Gestión de Cargas", layout="wide")

# --- Pestaña 1: Introducción ---
def pagina_introduccion():
    st.title("Introducción")
    st.write("Esta aplicación ha sido desarrollada en Google Colab y desplegada usando Streamlit.")
    st.write("Proporciona una herramienta para la gestión de cargas en logística, permitiendo visualizar rutas, estimar costos y analizar datos de transporte.")
    st.write("### Funcionalidades:")
    st.write("- Generación de carga aleatoria con detalles del origen y destino.")
    st.write("- Visualización de la ruta en un mapa interactivo.")
    st.write("- Cálculo del costo estimado de transporte utilizando un modelo de Machine Learning.")

# --- Pestaña 2: Datos Utilizados ---
def pagina_datos():
    st.title("Datos Utilizados")
    st.write("Esta aplicación utiliza dos conjuntos de datos principales:")
    st.write("1. **dataset.parquet**: Contiene la información detallada de las cargas.")
    st.write("2. **Xtest_encoded.parquet**: Contiene la versión codificada de los datos utilizada para hacer predicciones.")
    st.write("El modelo de predicción ha sido entrenado previamente y guardado como **random_forest_model.pkl**.")

# --- Pestaña 3: Modelo de Predicción ---
def pagina_modelo():
    st.title("Modelo de Predicción")
    st.write("El modelo utilizado en esta aplicación es un Random Forest Regressor entrenado para estimar los costos de transporte.")
    st.write("Se ha calculado un **Mean Absolute Percentage Error (MAPE)** de **11.64%**, lo que indica un buen desempeño en la estimación de costos.")
    st.write("El modelo ha sido entrenado con datos reales de transporte y utiliza variables como el tipo de vehículo, la distancia y el peso para hacer las predicciones.")

# --- Cargar datos y modelo ---
@st.cache_data
def load_data():
    file_path = "dataset.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo dataset.parquet")
        return pd.DataFrame()

df = load_data()

@st.cache_data
def load_encoded_data():
    file_path = "Xtest_encoded.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo Xtest_encoded.parquet")
        return pd.DataFrame()

df_encoded = load_encoded_data()

@st.cache_data
def load_model():
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    else:
        st.error("⚠️ No se encontró el archivo random_forest_model.pkl")
        return None

model = load_model()

@st.cache_data
def load_encoder():
    encoder_path = "encoder.pkl"
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as enc_file:
            return pickle.load(enc_file)
    else:
        st.error("⚠️ No se encontró el archivo encoder.pkl")
        return None

encoder = load_encoder()

# --- Preprocesar los datos antes de predecir ---
def preprocess_data(carga, feature_columns):
    df_temp = pd.DataFrame([carga])
    cat_columns = ['CityOrigin', 'CityDestination', 'Equip', 'StateOrigin', 'StateDestination']
    
    if encoder is not None:
        df_temp[cat_columns] = encoder.transform(df_temp[cat_columns])
    else:
        st.error("⚠️ No se pudo aplicar Target Encoding porque el encoder no está disponible.")
        return None
    
    return df_temp[feature_columns]

# --- Generar una carga aleatoria ---
def generar_carga():
    if df.empty:
        st.error("⚠️ No hay datos disponibles.")
        return None
    return df.sample(1).iloc[0]

# --- Página 1: Generar Carga ---
def pagina_generar_carga():
    st.title("Generar Carga")
    if st.button("Generar Carga"):
        carga = generar_carga()
        if carga is not None:
            st.session_state["carga"] = carga
    
    if "carga" in st.session_state:
        carga = st.session_state["carga"]
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.subheader("Detalles de la Carga")
            st.write(f"**Origen:** {carga['CityOrigin']}")
            st.write(f"**Destino:** {carga['CityDestination']}")
            st.write(f"**Peso:** {carga['Weight']} lbs")
            st.write(f"**Tamaño:** {carga['Size']} cu ft")
            
            equip = carga['Equip'].lower()
            image_path = f"images/{equip}.png"
            if os.path.exists(image_path):
                st.image(image_path, caption=equip)
            else:
                st.warning(f"Imagen no encontrada: {image_path}")
        
        with col2:
            st.subheader("Ruta en Mapa")
            mapa = folium.Map(location=[carga['LatOrigin'], carga['LngOrigin']], zoom_start=6)
            folium.Marker([carga['LatOrigin'], carga['LngOrigin']], tooltip="Origen").add_to(mapa)
            folium.Marker([carga['LatDestination'], carga['LngDestination']], tooltip="Destino").add_to(mapa)
            folium_static(mapa)
        
        with col3:
            st.subheader("Distancia Estimada")
            distancia = np.random.randint(100, 500)
            st.write(f"**Distancia:** {distancia} km")
            st.session_state["distancia"] = distancia

# --- Página 2: Vista Dueño del Vehículo ---
def pagina_dueno():
    st.title("Vista Dueño del Vehículo")
    if "carga" in st.session_state:
        pagina_generar_carga()
        
        st.subheader("Estimación de Pago")
        if model is not None and "distancia" in st.session_state:
            feature_columns = model.feature_names_in_
            features = preprocess_data(st.session_state['carga'], feature_columns)
            if features is not None:
                pred = model.predict(features)[0]
                min_value = pred * 0.9
                max_value = pred * 1.1
                st.write(f"💰 **Valor mínimo:** ${min_value:.2f}")
                st.write(f"💰 **Valor máximo:** ${max_value:.2f}")
            else:
                st.warning("No se pudo calcular el pago debido a un problema con la transformación de datos.")
        else:
            st.warning("No se pudo calcular el pago. Asegúrate de que el modelo está cargado y los datos están correctamente procesados.")
    else:
        st.warning("Genera una carga primero en la otra página.")

# --- Menú de Navegación ---
pagina = st.sidebar.selectbox("Selecciona una página", ["Generar Carga", "Vista Dueño del Vehículo"])
if pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()



# --- Menú de Navegación ---
pagina = st.sidebar.selectbox("Selecciona una página", ["Introducción", "Datos Utilizados", "Modelo de Predicción", "Generar Carga", "Vista Dueño del Vehículo"])
if pagina == "Introducción":
    pagina_introduccion()
elif pagina == "Datos Utilizados":
    pagina_datos()
elif pagina == "Modelo de Predicción":
    pagina_modelo()
elif pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()
