# Análisis de Rendimiento en Fútbol con Machine Learning

## Descripción
Este proyecto implementa dos modelos de machine learning aplicados al análisis de rendimiento en fútbol utilizando datos de la Copa del Mundo. El objetivo es proporcionar insights objetivos sobre el rendimiento de los jugadores y sus estilos de juego mediante técnicas avanzadas de análisis de datos.

## Objetivos

1. **Modelo de Clasificación**: 
   - Predecir si el rendimiento de un jugador en un partido fue alto o bajo
   - Utilizar variables como pases, tiros, intercepciones, etc.
   - Generar etiquetas a partir de métricas combinadas
   - Evaluar usando accuracy, F1 y matriz de confusión

2. **Modelo de Clustering**: 
   - Agrupar jugadores por estilo de juego
   - Analizar patrones basados en estadísticas como tipos de pases, duelos ganados, etc.
   - Visualizar grupos mediante PCA o t-SNE

## Metodología
El proyecto sigue la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining):

1. **Comprensión del Negocio**: Contexto del fútbol y objetivos del análisis
2. **Comprensión de los Datos**: Exploración de estructura y variables disponibles
3. **Preparación de Datos**: Limpieza, creación de features y normalización
4. **Modelado**: Implementación de modelos de clasificación y clustering
5. **Evaluación**: Análisis de rendimiento de los modelos
6. **Despliegue**: Conclusiones prácticas y aplicaciones potenciales

## Estructura del Proyecto
```
Tarea_M10_MPAD_Carpenco_Agustin/
├── README.md                       # Este archivo
├── data/                           # Datos
│   ├── raw/                        # Datos originales
│   │   └── matches_World_Cup.json  # Datos de partidos de la Copa del Mundo
│   └── processed/                  # Datos procesados
├── notebooks/                      # Jupyter notebooks con análisis
├── src/                           # Código fuente modular
└── requirements.txt               # Dependencias del proyecto
```

## Tecnologías Utilizadas
- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- XGBoost
- Técnicas de reducción de dimensionalidad (PCA, t-SNE)

## Autor
Agustín Carpenco