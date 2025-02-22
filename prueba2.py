import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Título
st.title("🔗 Matriz de Correlación - Dataset Iris")

# Cargar dataset
df = sns.load_dataset("iris")

# Matriz de correlación
st.subheader("📊 Mapa de Calor de Correlaciones")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

# Mostrar gráfico en Streamlit
st.pyplot(fig)



