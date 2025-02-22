import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Cargar dataset Iris
data = datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Título de la app
st.title("🌸 Clasificación de Especies en el Dataset Iris")

# Selección de variables para visualizar
x_var = st.selectbox("Selecciona la variable del eje X", df.columns[:-1])
y_var = st.selectbox("Selecciona la variable del eje Y", df.columns[:-1])

# Crear gráfico de dispersión
st.subheader("📊 Gráfico de Dispersión")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_var, y=y_var, hue='species', palette='coolwarm', ax=ax)
st.pyplot(fig)

# Mostrar pairplot completo
st.subheader("🔍 Pairplot de Todas las Variables")
pairplot_fig = sns.pairplot(df, hue='species', palette='coolwarm')
st.pyplot(pairplot_fig)
