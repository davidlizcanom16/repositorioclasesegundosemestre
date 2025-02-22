

# Configuración del título
st.title("🔍 Predicción de Especie - Modelo de Machine Learning")

# Cargar el dataset
df = sns.load_dataset("iris")

# Entrenar modelo Random Forest
X = df.drop(columns=["species"])
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar para ingresar valores manualmente
st.sidebar.header("🔢 Ingresa las características")

sepal_length = st.sidebar.number_input("📏 Largo del sépalo", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
sepal_width = st.sidebar.number_input("📐 Ancho del sépalo", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.sidebar.number_input("🌿 Largo del pétalo", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
petal_width = st.sidebar.number_input("🍃 Ancho del pétalo", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

# Predicción
if st.sidebar.button("🔍 Predecir Especie"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]

    st.subheader("✨ Especie Predicha")
    st.success(f"🌸 La especie más probable es: **{prediction.capitalize()}**")

    # Mostrar ejemplo de registros similares en la tabla
    df_similar = df[
        (df["sepal_length"] >= sepal_length - 0.5) & (df["sepal_length"] <= sepal_length + 0.5) &
        (df["sepal_width"] >= sepal_width - 0.5) & (df["sepal_width"] <= sepal_width + 0.5) &
        (df["petal_length"] >= petal_length - 0.5) & (df["petal_length"] <= petal_length + 0.5) &
        (df["petal_width"] >= petal_width - 0.5) & (df["petal_width"] <= petal_width + 0.5)
    ]

    st.write("📄 Registros similares en el dataset:")
    st.dataframe(df_similar)

    # Gráfico interactivo
    st.subheader("📊 Comparación con Datos Reales")
    fig = px.scatter(df, x="petal_length", y="petal_width", color="species", 
                     size="sepal_length", hover_data=["sepal_width"], 
                     title="Distribución de Tamaño de los Pétalos")
    
    # Agregar la predicción al gráfico
    fig.add_scatter(x=[petal_length], y=[petal_width], mode="markers", marker=dict(size=10, color="red"), name="Predicción")

    st.plotly_chart(fig)


