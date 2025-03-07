import pandas as pd
import os
import glob
import ast
import streamlit as st


def load_data(folder_path="."):
    """Carga archivos parquet desde la carpeta especificada y los concatena en un DataFrame."""
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    return df

def clean_data(df):
    """Realiza la limpieza de datos según los criterios establecidos."""
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    df = df[cols]
    
    # Eliminar duplicados por ID
    df = df.drop_duplicates('ID', keep='first')
    
    # Aplicar zonas de Estados Unidos
    df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
    df['ZoneDestination'] = df['StateDestination'].apply(get_zone)
    
    # Filtrar por tipo de camión
    df = filter_and_explode_equip(df)
    
    # Extraer día de la semana
    df['Posted'] = pd.to_datetime(df['Posted'])
    df['weekday'] = df['Posted'].dt.weekday
    df['weekday_name'] = df['Posted'].dt.day_name()
    
    # Eliminar datos de domingo
    df = df[df['weekday_name'] != 'Sunday']
    
    return df

def get_zone(state_code):
    """Asigna una zona a cada estado de EE.UU."""
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

def filter_and_explode_equip(df):
    """Filtra y expande la columna de Equip para mantener solo los tipos deseados."""
    desired_values = {'Van', 'Reefer', 'Flatbed'}
    df['Equip'] = df['Equip'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df[df['Equip'].apply(lambda x: isinstance(x, list))]
    df['Equip'] = df['Equip'].apply(lambda x: [item for item in x if item in desired_values])
    df = df[df['Equip'].map(len) > 0]
    return df.explode('Equip').reset_index(drop=True)

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df.to_parquet("loads_cleaned.parquet")
    print("Datos procesados y guardados en 'loads_cleaned.parquet'")
