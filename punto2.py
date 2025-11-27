import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("dataset_location.csv")

# Definir combustibles verdes
verdes = ["AGUA", "RAD SOLAR", "BIOGAS", "BAGAZO"]

# Filtrar recursos verdes
df_verde = df[df["Combustible"].isin(verdes)]

# Agrupar por Recurso y Tipo de Generación
agrupado = (df_verde.groupby(["Recurso", "Tipo Generación"])
            ["produccion_diaria"].sum()
            .reset_index()
            .sort_values("produccion_diaria", ascending=False))

# Graficar top 10
plt.figure(figsize=(10,6))
plt.barh(agrupado["Recurso"] + " - " + agrupado["Tipo Generación"],
         agrupado["produccion_diaria"], color="green")
plt.xlabel("Producción acumulada")
plt.ylabel("Recurso - Tipo")
plt.title("Producción total de recursos verdes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
