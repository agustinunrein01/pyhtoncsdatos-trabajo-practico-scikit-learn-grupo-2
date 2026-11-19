from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RUTA = Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"

def cargar_datos():
    # Carga el CSV comprimido y crea una versión numérica del precio.
    try:
        df = pd.read_csv(RUTA, compression="gzip")
    except FileNotFoundError:
        print(f"Aviso: No se encontró el archivo en {RUTA}. Se requiere para la ejecución.")
        return None
    if "price" not in df.columns:
        raise ValueError("No se encontró la columna price.")
    df["price_num"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
        errors="coerce"
    )
    return df

def preparar(df, predictores):
    # Convierte columnas, elimina faltantes y recorta el 1% inferior/superior.
    faltan = set(predictores) - set(df.columns)
    assert not faltan, f"Faltan columnas: {faltan}"

    columnas = predictores + ["price_num"]
    datos = df[columnas].copy()
    for col in predictores:
        datos[col] = pd.to_numeric(datos[col], errors="coerce")

    datos = datos.dropna()

    # Eliminación de registros imposibles
    datos = datos[datos["price_num"] > 0]
    datos = datos[datos["bedrooms"] >= 0]
    datos = datos[datos["accommodates"] > 0]
    datos = datos[datos["latitude"].between(-90, 90)]
    datos = datos[datos["longitude"].between(-180, 180)]
    
    q01, q99 = datos["price_num"].quantile([0.01, 0.99])
    datos = datos[datos["price_num"].between(q01, q99)]
    
    print("Nota: Se eliminan precios atípicos (el 1% inferior y el 1% superior).")
    print("Justificación: Se eliminan para evitar que propiedades atípicas sesguen las distancias.")
    return datos

def evaluar(y_real, y_pred):
    # Calcula y muestra métricas de regresión.
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    return mae, rmse, r2

def graficar_resultados(y_test, y_pred):
    """Genera un gráfico de valores reales frente a valores predichos."""

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)

    lim = [y_test.min(), y_test.max()]
    plt.plot(
        lim,
        lim,
        color="crimson",
        label="Predicción perfecta"
    )

    plt.xlabel("Precio real")
    plt.ylabel("Precio predicho")
    plt.title("k-NN (k=7): real vs. predicho")
    plt.legend()

    plt.tight_layout()
    RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "reports" / "figures"
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        RUTA_FIGURAS / "ejercicio_5.png",
        dpi=300,
        bbox_inches="tight"
)
    plt.show()


if __name__ == "__main__":
    df = cargar_datos()
    
    if df is not None:
        predictores = ["latitude", "longitude", "accommodates", "bedrooms"]
        datos = preparar(df, predictores)
        X = datos[predictores]
        y = datos["price_num"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        print("\n--- MODELO SELECCIONADO ---")
        print(
            "Se utiliza k-NN para regresión porque se busca estimar una variable "
            "numérica continua (precio) comparando alojamientos con ubicación "
            "y características físicas similares.")


        modelo = Pipeline([
            ("escalador", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))
        ])
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        print("\n--- MÉTRICAS ---")
        evaluar(y_test, y_pred)
        
        nuevo = X_test.iloc[[0]]
        print("\nCaracterísticas del caso:\n", nuevo)
        print("Precio real:", y_test.iloc[0])
        print("Precio estimado:", modelo.predict(nuevo)[0])
        
        escalador = modelo.named_steps["escalador"]
        knn = modelo.named_steps["knn"]
        
        # Extracción de vecinos escalando adecuadamente
        nuevo_escalado = escalador.transform(nuevo)
        distancias, indices = knn.kneighbors(nuevo_escalado)
        
        vecinos = X_train.iloc[indices[0]].copy()
        vecinos["precio"] = y_train.iloc[indices[0]].values
        vecinos["distancia_escalada"] = distancias[0]
        print("\n--- VECINOS MÁS CERCANOS ---")
        print(vecinos)
        
        print("\n--- REAL VS PREDICHO ---")
        comparacion = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).round(0)
        print(comparacion.head(10))
        
        print("\n--- EXPLICACIÓN DEL ESCALADO E INTERPRETACIÓN ---")
        print(
            "Sin escalar, las variables con magnitudes numéricas mayores podrían dominar "
            "el cálculo de las distancias."
        )
        print(
            "StandardScaler coloca las variables en escalas comparables antes de aplicar k-NN, "
            "evitando que una variable tenga mayor influencia solamente por su escala."
        )
        print(
            "La distancia calculada por k-NN corresponde al espacio estandarizado y no debe "
            "interpretarse como una distancia física en kilómetros."
        )
        
        print("\n--- CONCLUSIÓN ---")
        print(
            "El modelo k-NN estima el precio comparando cada alojamiento con otros de "
            "ubicación y características físicas similares."
        )
        print(
            "La estandarización dentro del pipeline permite calcular las distancias sobre "
            "variables expresadas en escalas comparables."
        )
        print(
            "Los resultados muestran un desempeño moderado, por lo que el modelo logra "
            "captar parte de la variabilidad del precio, aunque no toda."
        )
        print(
            "La dispersión entre valores reales y predichos indica que existen otros "
            "factores relevantes que no están incluidos entre los predictores utilizados."
        )
        
        # Gráfico Real vs Predicho
        graficar_resultados(y_test, y_pred)