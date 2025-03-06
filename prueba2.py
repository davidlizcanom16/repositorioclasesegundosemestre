#**Proyecto Final Modelos Analíticos - Maestría Analítica Datos**
### **Miembros del Equipo:**
*  Karen Gomez
*  David Lizcano
*  Jason Barrios
*  Camilo Barriosnuevo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Úsalo si realmente lo necesitas
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import sweetviz as sv


"""##Contexto de Negocio: Situación problema
Empresa de bandera Norteamericana dedicada a conectar generadores de carga y transportistas con las necesidades de envío  a través de sus servicios tecnológicos, se encuentra en la búsqueda de predecir cúanto va pagar un determinado cliente por carga transportada, ya que por naturaleza de dinámica de mercado (oferta/demanda), algunos clientes no suelen publicar las tarifas de envío en el portal, generando incertidumbre en las condiciones de negociación con los transportistas.

### El objetivo es suministrar información oportuna a los transportistas dando la visibilidad de las ofertas de carga dependiendo la zona, día, cliente y estimación de las tarifas. Para ello se entrenará un modelo para encontrar la variable `RatePerMile` que anticipará la tarifa que ofertará un cliente y se comparará con la de mercado

## 1. Limpieza de información

### 1.1 Cargas Broker
Acciones:
- Filtrar broker
- Duplicados por ID
- Seleccionar sólo camiones [Vans]
- Filtrar por [RatePerMile] (por naturaleza de mercado muchas observaciones que publican sin tarifa)
- Cambio de Estados de Origen y destino por zonas (reducción)
- Eliminar lanes intrahub
- Eliminar outliers generales

Output:
Dataset para realizar el proyecto

#### Filtro broker
Para este análisis se analizarán solamente las cargas publicadas por una empresa cuyo nombre reservamos. En el siguiente comando se leen todas las bases de datos correspondientes y se concatenan en un solo archivo que tiene la información de las últimas 3 semanas de cargas publicadas exceptuando sábados y domingos (3-28 de febrero). En total obtenemos 18625 cargas.
"""

#Cargas

df = pd.DataFrame()

folder_base = os.getcwd()
parquet_files = glob.glob(folder_base+"/*.parquet")

dfs = []
for file in parquet_files:
    df = pd.read_parquet(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
 'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
 'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
df = df[cols]
df.to_parquet("loads.parquet")

df.shape

"""#### Eliminación duplicados
Es necesario borrar duplicados por ID dado que es posible que una carga haya permanecido varios días publicada antes de que algún conductor la haya reservado. Solo existen 53 registros duplicados, los eliminaremos de la base de datos.
"""

18625 - df['ID'].nunique()

df = df.drop_duplicates('ID',keep='first')
df.shape

"""#### Manejo de origen y destino por zona
En este punto nuestro origen destino depende de la ciudad y estado de la carga, sin embargo, dada la cantidad de ciudades distintas y dado que al final un comportamiento similar de cargas sucede a nivel de estado, manejaremos el análisis a nivel de estado.

Además dado que la cantidad de estados igual es alta sobre 45 posibles opciones de origen, destino y la combinación entre estos. Podemos hacer un segundo análisis por Zonas de Estados Unidos, las cuales son 10. Eventualmente podremos concluir a partir de esto si disminuir la variabilidad del origen destino de la carga reduce el error de la predicción.
"""

df['StateOrigin'].nunique() + df['StateDestination'].nunique()

df['CityOrigin'].nunique() + df['CityDestination'].nunique()


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

df['ZoneOrigin'] = df['StateOrigin'].apply(lambda x: get_zone(x))
df['ZoneDestination'] = df['StateDestination'].apply(lambda x: get_zone(x))
df['ZoneOrigin'].unique()

"""#### Manejo de camiones
El siguiente paso es limitar el tipo de camión de este análisis.
"""

df['Equip'].unique()

import ast
def filter_and_explode_equip(data):
    if data.empty:
        return pd.DataFrame()
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    desired_values_set = set(desired_values)

    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values_set])

    data = data[data[column_name].map(len) > 0]
    data = data.explode(column_name).reset_index(drop=True)

    return data

df = filter_and_explode_equip(df)
print(df.shape)
df['Equip'].unique()

"""#### Extracción día de la semana"""

df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()

df.pivot_table(index='weekday_name',values='ID',aggfunc='count')

"""Dado que el día "Domingo" no es incluído en los días de la base de datos, parece ser un día atípico e imputado incorrectamente, por este motivo lo borraremos de la base de datos."""

df = df.drop(df.loc[df['weekday_name']=='Sunday'].index)
df.shape
