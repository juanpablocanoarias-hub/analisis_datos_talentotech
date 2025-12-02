import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import squarify
import geopandas as gpd
import geopy
import folium
from plotly.subplots import make_subplots
from plotly.offline import plot
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time

df = pd.read_csv("dataset_final.csv")
columnas_horarias = ['0','1','2','3','4','5','6','7','8','9','10','11','12',
                    '13','14','15','16','17','18','19','20','21','22','23','produccion_diaria']
for columna in columnas_horarias:
    if columna in df.columns:
        df[f'{columna}_GWh'] = df[columna] / 1e6

df['Fecha'] = pd.to_datetime(df['Fecha'])

df['mes'] = df['Fecha'].dt.month
df['dia_semana'] = df['Fecha'].dt.day_name()
df['semana_año'] = df['Fecha'].dt.isocalendar().week
df['trimestre'] = df['Fecha'].dt.quarter
df['dia_del_mes'] = df['Fecha'].dt.day


i=2
if i == 0:
    plt.figure(figsize=(15, 6))
    df.groupby('Fecha')['produccion_diaria'].sum().plot()
    plt.title('Producción Eléctrica Diaria - Año Completo')
    plt.ylabel('Producción (kWh)')
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    df['mes'] = df['Fecha'].dt.month
    df.groupby('mes')['produccion_diaria'].sum().plot(kind='bar')
    plt.title('Producción por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Producción Total (kWh)')
    plt.show()

    df['dia_semana'] = df['Fecha'].dt.day_name()
    produccion_diaria = df.groupby('dia_semana')['produccion_diaria'].mean()
    orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    produccion_diaria.loc[orden_dias].plot(kind='bar')
    plt.title('Producción Promedio por Día de la Semana')
    plt.show()

    plt.figure(figsize=(12, 6))
    df.groupby('0')['produccion_diaria'].sum().plot(kind='hist')
    plt.title('Patrón Horario de Producción (Promedio)')
    plt.xlabel('Hora del Día')
    plt.ylabel('Producción Promedio (kWh)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    produccion_mensual = df.groupby('mes')['produccion_diaria'].sum()
    produccion_mensual.plot(kind='bar')
    plt.title('Producción Eléctrica Mensual')
    plt.ylabel('Producción Total (MWh)')
    plt.xlabel('Mes')
    plt.xticks(rotation=0)
    plt.show()

plt.figure(figsize=(12, 6))
produccion_tipo = df.groupby('Tipo Generación')['produccion_diaria_GWh'].sum().sort_values(ascending=False)
bars = plt.bar(produccion_tipo.index, produccion_tipo.values, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Producción Total por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Total (GWh)')
plt.xlabel('Tipo de Generación')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
produccion_diaria_total = df.groupby('Fecha')['produccion_diaria_GWh'].sum()
plt.plot(produccion_diaria_total.index, produccion_diaria_total.values, 
         linewidth=1, color='steelblue')
plt.title('Evolución Diaria de la Producción Eléctrica', fontsize=14, fontweight='bold')
plt.ylabel('Producción Diaria (GWh)')
plt.xlabel('Fecha')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

horas_columns = [f'{j}_GWh' for j in range(24)]
produccion_promedio_hora = df[horas_columns].mean()

plt.figure(figsize=(14, 8))
plt.imshow([produccion_promedio_hora.values], cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Producción Promedio (GWh)')
plt.title('Producción Promedio por Hora del Día', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día')
plt.yticks([])
plt.xticks(range(24))
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
tipos_generacion = df['Tipo Generación'].unique()
for i, tipo in enumerate(tipos_generacion):
    df_tipo = df[df['Tipo Generación'] == tipo]
    perfil_horario = df_tipo[horas_columns].sum()
    plt.plot(range(24), perfil_horario.values, marker='o', label=tipo, linewidth=2)
plt.title('Perfil Horario Promedio por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Hora del Día')
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
tipos_generacion = df['Tipo Generación'].unique()
for i, tipo in enumerate(tipos_generacion):
    df_tipo = df[df['Tipo Generación'] == tipo]
    perfil_horario = df_tipo[horas_columns].mean()
    plt.plot(range(24), perfil_horario.values, marker='o', label=tipo, linewidth=2)
plt.title('Perfil Horario Promedio por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Hora del Día')
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
produccion_dia_semana = df.groupby('dia_semana')['produccion_diaria_GWh'].mean()
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_esp = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
produccion_dia_semana = produccion_dia_semana.reindex(dias_orden)
plt.bar(dias_esp, produccion_dia_semana.values, color='lightcoral', alpha=0.8)
plt.title('Producción Promedio por Día de la Semana', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Día de la Semana')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

produccion_mes_combustible = df.groupby(['mes', 'Combustible'])['produccion_diaria_GWh'].sum().unstack().fillna(0)
top_combustibles = df.groupby('Combustible')['produccion_diaria_GWh'].sum().nlargest(13).index
produccion_mes_combustible = produccion_mes_combustible[top_combustibles]
produccion_mes_combustible.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('Producción Mensual por Combustible', fontsize=14, fontweight='bold')
plt.ylabel('Producción (GWh)')
plt.xlabel('Mes')
plt.xticks(range(12), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=0)
plt.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

fig = px.line(df.groupby(['Fecha', 'Tipo Generación'])['produccion_diaria_GWh'].sum().reset_index(),
              x='Fecha', y='produccion_diaria_GWh', color='Tipo Generación',
              title='Evolución de Producción por Tipo (Interactivo)')
plot(fig, filename='produccion_por_tipo.html', auto_open=True)

produccion_departamento = df.groupby('Departamento')['produccion_diaria_GWh'].sum().sort_values(ascending=False)
produccion_municipio = df.groupby(['Departamento', 'Municipio'])['produccion_diaria_GWh'].sum().reset_index()
top_departamentos = produccion_departamento.head(25)
top_municipios = produccion_municipio.nlargest(25, 'produccion_diaria_GWh')

plt.figure(figsize=(14, 8))
bars = plt.barh(top_departamentos.index, top_departamentos.values, color='teal')
plt.title('Top 15 Departamentos por Producción Eléctrica', fontsize=16, fontweight='bold')
plt.xlabel('Producción Total (GWh)')
plt.ylabel('Departamento')
plt.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:,.0f}', ha='left', va='center', fontweight='bold')
plt.tight_layout()
plt.show()

df_termica = df[df['Tipo Generación'] == 'TERMICA'].copy()
print(f"📈 ESTADÍSTICAS DE GENERACIÓN TÉRMICA")
print("=" * 50)
print(f"Registros de generación térmica: {len(df_termica)}")
print(f"Producción total térmica: {df_termica['produccion_diaria_GWh'].sum():,.1f} GWh")
print(f"Porcentaje del total nacional: {(df_termica['produccion_diaria_GWh'].sum() / df['produccion_diaria_GWh'].sum() * 100):.1f}%")
print(f"Combustibles utilizados: {df_termica['Combustible'].nunique()}")
print(f"Centrales térmicas: {df_termica['Recurso'].nunique()}")
print(f"Municipios con generación térmica: {df_termica['Municipio'].nunique()}")

# Producción mensual por combustible
produccion_mensual_combustible = df_termica.groupby(['mes', 'Combustible'])['produccion_diaria_GWh'].sum().unstack().fillna(0)

# Tomar los 8 combustibles principales
top_combustibles = df_termica.groupby('Combustible')['produccion_diaria_GWh'].sum().nlargest(8).index
produccion_mensual_combustible = produccion_mensual_combustible[top_combustibles]

# Gráfico de líneas
plt.figure(figsize=(16, 8))
for combustible in produccion_mensual_combustible.columns:
    plt.plot(produccion_mensual_combustible.index, 
             produccion_mensual_combustible[combustible], 
             marker='o', linewidth=2, markersize=6, label=combustible)

plt.title('Evolución Mensual de Producción Térmica por Combustible', fontsize=16, fontweight='bold')
plt.ylabel('Producción Mensual (GWh)')
plt.xlabel('Mes')
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Preparar datos para heatmap
heatmap_data = produccion_mensual_combustible.T

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.0f',
            cbar_kws={'label': 'Producción Mensual (GWh)'})

plt.title('Heatmap: Producción Térmica por Combustible y Mes', fontsize=16, fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Combustible')
plt.tight_layout()
plt.show()

# Producción por departamento y combustible
produccion_depto_combustible = df_termica.pivot_table(
    values='produccion_diaria_GWh',
    index='Departamento',
    columns='Combustible',
    aggfunc='sum'
).fillna(0)

# Top 10 departamentos en generación térmica
top_deptos_termica = df_termica.groupby('Departamento')['produccion_diaria_GWh'].sum().nlargest(10).index
produccion_depto_combustible = produccion_depto_combustible.loc[top_deptos_termica]

# Gráfico de barras apiladas
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Gráfico 1: Barras apiladas
produccion_depto_combustible.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Top 10 Departamentos - Producción Térmica por Combustible', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Producción Total (GWh)')
axes[0].set_xlabel('Departamento')
axes[0].legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].tick_params(axis='x', rotation=45)

# Gráfico 2: Heatmap departamento vs combustible
sns.heatmap(produccion_depto_combustible, 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.0f',
            cbar_kws={'label': 'Producción (GWh)'},
            ax=axes[1])
axes[1].set_title('Heatmap: Producción Térmica por Departamento y Combustible', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Combustible')
axes[1].set_ylabel('Departamento')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Para cada combustible, identificar las centrales más importantes
combustibles_principales = df_termica.groupby('Combustible')['produccion_diaria_GWh'].sum().nlargest(6).index

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, combustible in enumerate(combustibles_principales):
    # Filtrar datos del combustible
    df_combustible = df_termica[df_termica['Combustible'] == combustible]
    
    # Top 5 centrales para este combustible
    top_centrales = df_combustible.groupby('Recurso')['produccion_diaria_GWh'].sum().nlargest(5)
    
    # Gráfico de barras
    bars = axes[idx].barh(range(len(top_centrales)), top_centrales.values)
    axes[idx].set_yticks(range(len(top_centrales)))
    axes[idx].set_yticklabels(top_centrales.index, fontsize=9)
    axes[idx].set_title(f'{combustible}\nTotal: {df_combustible["produccion_diaria_GWh"].sum():,.0f} GWh', 
                       fontweight='bold', fontsize=10)
    axes[idx].set_xlabel('Producción Total (GWh)')
    
    # Añadir municipio como etiqueta
    for i, (central, _) in enumerate(top_centrales.items()):
        municipio = df_combustible[df_combustible['Recurso'] == central]['Municipio'].iloc[0]
        depto = df_combustible[df_combustible['Recurso'] == central]['Departamento'].iloc[0]
        axes[idx].text(5, i, f"{municipio}, {depto}", fontsize=8, va='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Crear serie mensual para los principales combustibles
mix_mensual_termico = produccion_mensual_combustible

mix_mensual_termico.plot(kind='area', stacked=True, alpha=0.8, figsize=(16, 8))

plt.title('Mix de Combustibles en Generación Térmica por Mes', fontsize=16, fontweight='bold')
plt.ylabel('Producción Mensual (GWh)')
plt.xlabel('Mes')
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Crear subplots para diferentes análisis
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Producción acumulada por combustible
for combustible in top_combustibles[:6]:
    df_comb = df_termica[df_termica['Combustible'] == combustible].sort_values('Fecha')
    df_comb['Acumulado'] = df_comb['produccion_diaria_GWh'].cumsum()
    axes[0, 0].plot(df_comb['Fecha'], df_comb['Acumulado'], label=combustible, linewidth=2)

axes[0, 0].set_title('Producción Acumulada por Combustible', fontweight='bold')
axes[0, 0].set_ylabel('Producción Acumulada (GWh)')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].legend(loc='upper left', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 2. Porcentaje de participación por mes
participacion_mensual = produccion_mensual_combustible.div(produccion_mensual_combustible.sum(axis=1), axis=0) * 100
for combustible in participacion_mensual.columns[:5]:
    axes[0, 1].plot(participacion_mensual.index, participacion_mensual[combustible], 
                   marker='s', linewidth=2, label=combustible)

axes[0, 1].set_title('Participación Mensual por Combustible (%)', fontweight='bold')
axes[0, 1].set_ylabel('Participación (%)')
axes[0, 1].set_xlabel('Mes')
axes[0, 1].legend(loc='upper left', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 3. Heatmap de días con generación por combustible
# Contar días con generación por mes y combustible
dias_generacion = df_termica.groupby(['mes', 'Combustible']).size().unstack().fillna(0)
dias_generacion = dias_generacion[top_combustibles]

sns.heatmap(dias_generacion, cmap='Blues', annot=True, fmt='.0f',
            cbar_kws={'label': 'Días con Generación'}, ax=axes[1, 0])
axes[1, 0].set_title('Días con Generación por Combustible y Mes', fontweight='bold')
axes[1, 0].set_xlabel('Combustible')
axes[1, 0].set_ylabel('Mes')

# 4. Eficiencia relativa (producción/días)
eficiencia = produccion_mensual_combustible.div(dias_generacion).replace([np.inf, -np.inf], np.nan).fillna(0)
sns.heatmap(eficiencia, cmap='Greens', annot=True, fmt='.1f',
            cbar_kws={'label': 'Producción Promedio por Día (GWh)'}, ax=axes[1, 1])
axes[1, 1].set_title('Producción Promedio por Día de Generación', fontweight='bold')
axes[1, 1].set_xlabel('Combustible')
axes[1, 1].set_ylabel('Mes')

plt.tight_layout()
plt.show()

# 1. Preparar datos de centrales únicas
centrales_unicas = df[['Recurso', 'Municipio', 'Departamento', 'Tipo Generación', 'Combustible']].drop_duplicates()

# 2. Función para geocodificar
def geocodificar_ubicacion(municipio, departamento, recurso):
    """Obtiene coordenadas de un municipio usando Nominatim"""
    geolocator = Nominatim(user_agent="centrales_electricas_colombia")
    
    try:
        # Primero intentar con municipio + departamento
        location = geolocator.geocode(f"{municipio}, {departamento}, Colombia")
        if location:
            return location.latitude, location.longitude
        else:
            # Si no encuentra, intentar solo con departamento
            location = geolocator.geocode(f"{departamento}, Colombia")
            return location.latitude, location.longitude if location else (None, None)
    except:
        return (None, None)

# 3. Aplicar geocodificación (esto puede tardar)
print("⏳ Geocodificando ubicaciones... Esto puede tomar varios minutos")
centrales_unicas['coordenadas'] = centrales_unicas.apply(
    lambda x: geocodificar_ubicacion(x['Municipio'], x['Departamento'], x['Recurso']), 
    axis=1
)

# 4. Separar latitud y longitud
centrales_unicas['latitud'] = centrales_unicas['coordenadas'].apply(lambda x: x[0] if x else None)
centrales_unicas['longitud'] = centrales_unicas['coordenadas'].apply(lambda x: x[1] if x else None)

print(f"✅ Geocodificación completada. {centrales_unicas['latitud'].notna().sum()}/{len(centrales_unicas)} ubicaciones encontradas")

# Crear mapa centrado en Colombia
mapa = folium.Map(location=[4.570868, -74.297333], zoom_start=6)

# Colores por tipo de generación
colores_tipo = {
    'HIDRÁULICA': 'blue',
    'TÉRMICA': 'red',
    'SOLAR': 'orange',
    'EÓLICA': 'green',
    'COGENERACIÓN': 'purple'
}

# Añadir marcadores para cada central
for _, central in centrales_unicas.dropna(subset=['latitud', 'longitud']).iterrows():
    # Calcular producción total de la central
    produccion_total = df[df['Recurso'] == central['Recurso']]['produccion_diaria_GWh'].sum()
    
    # Crear popup con información
    popup_content = f"""
    <b>{central['Recurso']}</b><br>
    <b>Tipo:</b> {central['Tipo Generación']}<br>
    <b>Combustible:</b> {central['Combustible']}<br>
    <b>Ubicación:</b> {central['Municipio']}, {central['Departamento']}<br>
    <b>Producción total:</b> {produccion_total:,.0f} GWh<br>
    """
    
    folium.CircleMarker(
        location=[central['latitud'], central['longitud']],
        radius=8 + (produccion_total / 1000),  # Tamaño proporcional a producción
        popup=folium.Popup(popup_content, max_width=300),
        color=colores_tipo.get(central['Tipo Generación'], 'gray'),
        fill=True,
        fill_color=colores_tipo.get(central['Tipo Generación'], 'gray'),
        fill_opacity=0.7
    ).add_to(mapa)

# Añadir leyenda
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 180px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;">
     <b>🎯 Leyenda</b><br>
     <i class="fa fa-circle" style="color:blue"></i> Hidráulica<br>
     <i class="fa fa-circle" style="color:red"></i> Térmica<br>
     <i class="fa fa-circle" style="color:orange"></i> Solar<br>
     <i class="fa fa-circle" style="color:green"></i> Eólica<br>
     <i class="fa fa-circle" style="color:purple"></i> Cogeneración<br>
     <br><b>Tamaño:</b><br>Producción total
</div>
'''
mapa.get_root().html.add_child(folium.Element(legend_html))

# Guardar mapa
mapa.save('mapa_centrales_electricas_colombia.html')
print("✅ Mapa guardado como 'mapa_centrales_electricas_colombia.html'")
print("📂 Ábrelo en tu navegador para verlo interactivo")

"""Genera un resumen del dataset con la estructura actual"""
print("RESUMEN - DATOS DIARIOS")
print("=" * 50)
print(f"Período: {df['Fecha'].min().strftime('%d/%m/%Y')} a {df['Fecha'].max().strftime('%d/%m/%Y')}")
print(f"Días analizados: {df['Fecha'].nunique()}")
print(f"Producción total: {df['produccion_diaria_GWh'].sum():,.1f} GWh")
print(f"Producción promedio diaria: {df['produccion_diaria_GWh'].mean():.1f} GWh")
print(f"Centrales eléctricas: {df['Recurso'].nunique()}")

print("\n Tipos de generación:")
for tipo, prod in df.groupby('Tipo Generación')['produccion_diaria_GWh'].sum().items():
    porcentaje = (prod / df['produccion_diaria_GWh'].sum()) * 100
    print(f"  - {tipo}: {prod:,.1f} GWh ({porcentaje:.1f}%)")

def resumen_municipal_completo(df, top_n=10):
    """Genera un resumen ejecutivo del análisis municipal"""
    
    print("🏙️  RESUMEN EJECUTIVO - ANÁLISIS MUNICIPAL")
    print("=" * 70)
    
    # Estadísticas generales
    num_municipios = df['Municipio'].nunique()
    num_departamentos = df['Departamento'].nunique()
    total_prod = df['produccion_diaria_GWh'].sum()
    
    print(f"📊 ESTADÍSTICAS GENERALES:")
    print(f"   • Municipios productores: {num_municipios}")
    print(f"   • Departamentos involucrados: {num_departamentos}")
    print(f"   • Producción total nacional: {total_prod:,.0f} GWh")
    
    # Top municipios
    top_municipios = df.groupby(['Departamento', 'Municipio'])['produccion_diaria_GWh'].sum().nlargest(top_n)
    
    print(f"\n🏆 TOP {top_n} MUNICIPIOS PRODUCTORES:")
    for i, ((depto, mun), prod) in enumerate(top_municipios.items(), 1):
        porcentaje = (prod / total_prod) * 100
        print(f"   {i:2d}. {mun} ({depto}): {prod:,.0f} GWh ({porcentaje:.1f}%)")
    
    # Concentración
    print(f"\n📈 ANÁLISIS DE CONCENTRACIÓN:")
    for n in [3, 5, 10]:
        top_n_sum = top_municipios.head(n).sum()
        porcentaje = (top_n_sum / total_prod) * 100
        print(f"   • Top {n} municipios: {porcentaje:.1f}% de la producción nacional")
    
    # Diversificación
    tipos_por_mun = df.groupby(['Departamento', 'Municipio'])['Tipo Generación'].nunique()
    mun_mas_diverso = tipos_por_mun.idxmax()
    num_tipos = tipos_por_mun.max()
    
    print(f"\n🌿 DIVERSIFICACIÓN:")
    print(f"   • Municipio más diversificado: {mun_mas_diverso[1]} ({mun_mas_diverso[0]})")
    print(f"     con {num_tipos} tipos diferentes de generación")
    
    # Departamentos con más municipios productores
    depto_con_mas_mun = df.groupby('Departamento')['Municipio'].nunique().idxmax()
    num_mun_depto = df.groupby('Departamento')['Municipio'].nunique().max()
    
    print(f"\n📍 DISTRIBUCIÓN GEOGRÁFICA:")
    print(f"   • Departamento con más municipios productores: {depto_con_mas_mun}")
    print(f"     con {num_mun_depto} municipios generando electricidad")

# Ejecutar resumen
resumen_municipal_completo(df, top_n=15)