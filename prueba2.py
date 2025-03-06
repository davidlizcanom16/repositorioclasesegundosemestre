import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import folium

# Cargar los datos
df = pd.read_excel("/louisville_traffic.xlsx")

## Análisis de valores nulos
df.info()
df.head(3)

null_counts = df.isnull().sum()

plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno')
plt.title('Heatmap de Valores Nulos')
plt.xlabel('Columnas')
plt.ylabel('Filas')
plt.show()

filtered_df = df[df['RatePerMile'].isnull()]
sns.histplot(data=filtered_df, x='Equip', hue='Equip')
plt.title('Histograma de valores nulos de RatePerMile por Equip')
plt.xlabel('Equip')
plt.ylabel('Cantidad de valores nulos')
plt.show()

summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size',
    nulos=lambda x: x.isnull().sum(),
    no_nulos='count'
)
summary['% nulos por Equip'] = (summary['nulos'] / summary['total']) * 100
summary['% nulos por Equip'] = summary['% nulos por Equip'].map("{:.2f}%".format)
print(summary)

df.describe()

## Mapa de cargas con y sin tarifa publicada
state_total_counts = df.groupby('StateOrigin')['RatePerMile'].size()
state_non_null_counts = df.groupby('StateOrigin')['RatePerMile'].count()
state_null_counts = df[df['RatePerMile'].isnull()].groupby('StateOrigin').size()

summary_df = pd.DataFrame({
    'Envíos sin Rate': state_null_counts,
    'Envíos con Rate': state_non_null_counts,
    'Total_Envíos': state_total_counts
})
summary_df = summary_df.fillna(0).astype(int)
summary_df['% Envíos null'] = (summary_df['Envíos sin Rate'] / summary_df['Total_Envíos']) * 100
summary_df['% Envíos null'] = summary_df['% Envíos null'].map("{:.2f}%".format)
summary_df = summary_df.sort_values(by=['Total_Envíos'], ascending=False)
display(summary_df)

m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["LatOrigin"], row["LngOrigin"]],
        color="Blue" if row["RatePerMile"] > 0 else "orange",
        fill=True,
    ).add_to(m)
m

## Eliminación de rates NaN
df = df.loc[df['RatePerMile'] > 0]
piv = df.pivot_table(index='Equip', values='ID', aggfunc='count')
display(piv)
print(piv['ID'].sum())

df = df.drop(df[df['HubOrigin'] == df['HubDestination']].index)

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    return df[mask]

df = remove_outliers_iqr(df, 'RatePerMile')

plt.hist(df['RatePerMile'], range=(0, 7), bins=100)
plt.xlabel('RatePerMile')
plt.ylabel('Frecuencia')
plt.title('Histograma de RatePerMile')
plt.show()

fig = px.box(df, y="RatePerMile")
fig.show()

fig = px.box(df, x="Equip", y="RatePerMile", color="Equip")
fig.show()

sns.boxplot(data=df, x='ZoneDestination', y='RatePerMile', hue='Equip')

model = smf.ols('RatePerMile ~ Equip', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

cargas_por_dia = df.groupby(df['Posted'].dt.date)['ID'].nunique().reset_index()
fig = px.bar(cargas_por_dia, x='Posted', y='ID',
             title='Cantidad de cargas publicadas por día',
             labels={'Posted': 'Día', 'ID': 'Cantidad de cargas'},
             color='ID',
             color_continuous_scale='Greys')
fig.show()

fig = px.scatter_mapbox(
    df,
    lat="LatOrigin",
    lon="LngOrigin",
    color="Equip",
    hover_name="ID",
    zoom=4,
    height=600,
    mapbox_style="open-street-map"
)
fig.show()

df_num = df.select_dtypes(include=['number'])
correlation_matrix = df_num.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de calor de la Matriz de Correlación")
plt.show()

df['ZoneCombination'] = df['ZoneOrigin'] + '-' + df['ZoneDestination']
combination_counts = df['ZoneCombination'].value_counts().reset_index()
combination_counts.columns = ['ZoneCombination', 'Count']

fig = px.bar(
    combination_counts.head(10),
    x='ZoneCombination',
    y='Count',
    title='Combinaciones de Zonas Más Frecuentes',
    color='ZoneCombination',
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig.show()

fig = px.box(
    df,
    x="ZoneCombination",
    y="RatePerMile",
    category_orders={"ZoneCombination": df.groupby("ZoneCombination")["RatePerMile"].mean().sort_values(ascending=False).index},
    title='BoxPlot de Tarifa por milla respecto al Trayecto',
    color='ZoneCombination',
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig.update_xaxes(tickangle=-45)
fig.show()

df['ZoneCombination'].nunique()

mapa = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
colores = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "black", "cyan", "magenta", "lime", "teal", "olive", "navy", "maroon", "aquamarine", "coral", "fuchsia", "silver", "gold", "indigo", "lavender"]
paleta_colores = {zona: colores[i % len(colores)] for i, zona in enumerate(df["ZoneCombination"].unique())}
for _, registro in df.iterrows():
    folium.CircleMarker(
        location=[registro["LatOrigin"], registro["LngOrigin"]],
        radius=5,
        color=paleta_colores.get(registro["ZoneCombination"], "gray"),
        fill=True,
        fill_color=paleta_colores.get(registro["ZoneCombination"], "gray"),
        popup=registro["ZoneCombination"],
        tooltip=registro["ZoneCombination"]
    ).add_to(mapa)
mapa
