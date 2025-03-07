import streamlit as st
import pandas as pd


# Configuración de la página
st.set_page_config(page_title="Proyecto Final Modelos Analíticos", layout="wide")

# Encabezado principal con estilo
st.markdown("<h1 style='text-align: center;'>Proyecto Final Modelos Analíticos</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Equipo: Karen Gomez, David Lizcano, Jason Barrios, Camilo Barriosnuevo</h3>", unsafe_allow_html=True)
st.markdown("---")

# Creación de pestañas para organizar la app
tab1, tab2, tab3 = st.tabs(["Carga y Limpieza", "Visualización", "Descarga CSV"])

# Pestaña 1: Carga y Limpieza de Datos
with tab1:
    st.header("Carga y Limpieza de Datos")
    
    @st.cache_data
    def load_data():
        # Asume que los archivos .parquet se encuentran en el directorio actual
        folder_base = os.getcwd()
        parquet_files = glob.glob(os.path.join(folder_base, "*.parquet"))
        dfs = [pd.read_parquet(file) for file in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
        cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
                'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
                'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
        return df[cols]

    df = load_data()
    st.write("**Dimensiones iniciales del DataFrame:**", df.shape)

    # Eliminación de duplicados
    duplicados = 18625 - df['ID'].nunique()
    st.write(f"**Duplicados detectados (estimado):** {duplicados}")
    df = df.drop_duplicates('ID', keep='first')
    st.write("**Dimensiones después de eliminar duplicados:**", df.shape)

    # Asignación de zonas según el estado
    def get_zone(state_code):
        zones = {
            "Z0": {"CT", "ME", "MA", "NH", "NJ", "RI", "VT"},
            "Z1": {"DE", "NY", "PA"},
            "Z2": {"MD", "NC", "SC", "VA", "WV"},
            "Z3": {"AL", "FL", "GA", "MS", "TN"},
            "Z4": {"IN", "KY", "MI", "OH"},
            "Z5": {"IA", "MN", "MT", "ND", "SD", "WI"},
            "Z6": {"IL", "KS", "MO", "NE"},
            "Z7": {"AR", "LA", "OK", "TX"},
            "Z8": {"AZ", "CO", "ID", "NV", "NM", "UT", "WY"},
            "Z9": {"CA", "OR", "WA", "AK"}
        }
        for zone, states in zones.items():
            if state_code in states:
                return zone
        return "Unknown"

    df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
    df['ZoneDestination'] = df['StateDestination'].apply(get_zone)
    st.write("**Zonas de origen identificadas:**", df['ZoneOrigin'].unique())

    # Filtrado y explosi\'on del campo Equip (camiones)
    def filter_and_explode_equip(data):
        desired_values = ['Van', 'Reefer', 'Flatbed']
        data['Equip'] = data['Equip'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        data = data[data['Equip'].apply(lambda x: isinstance(x, list))]
        data['Equip'] = data['Equip'].apply(lambda x: [item for item in x if item in desired_values])
        data = data[data['Equip'].map(len) > 0]
        return data.explode('Equip').reset_index(drop=True)
    
    df = filter_and_explode_equip(df)
    st.write("**Dimensiones tras filtrar equipos:**", df.shape)
    st.write("**Tipos de Equip disponibles:**", df['Equip'].unique())

    # Extracción del día de la semana
    df['Posted'] = pd.to_datetime(df['Posted'])
    df['weekday_name'] = df['Posted'].dt.day_name()
    pivot = df.pivot_table(index='weekday_name', values='ID', aggfunc='count')
    st.write("**Conteo de registros por día de la semana:**")
    st.dataframe(pivot)

    # Eliminación de registros de domingo
    df = df.drop(df.loc[df['weekday_name'] == 'Sunday'].index)
    st.write("**Dimensiones finales tras eliminar registros de domingo:**", df.shape)

# Pestaña 2: Visualización
with tab2:
    st.header("Visualización")
    st.markdown("### Distribución de Cargas por Día de la Semana")
    fig = px.bar(pivot.reset_index(), x='weekday_name', y='ID', 
                 title="Número de Cargas por Día", 
                 labels={"weekday_name": "Día de la Semana", "ID": "Cantidad de Cargas"})
    st.plotly_chart(fig, use_container_width=True)

# Pestaña 3: Descarga del Dataset Limpio
with tab3:
    st.header("Descarga de Datos")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos limpios (CSV)",
        data=csv,
        file_name='datos_limpios.csv',
        mime='text/csv'
    )
    st.success("¡Descarga completada!")
