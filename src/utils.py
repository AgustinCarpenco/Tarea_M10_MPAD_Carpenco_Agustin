"""
Módulo de utilidades generales para el proyecto.
Contiene funciones auxiliares que pueden ser utilizadas en distintos módulos.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dir(base_dir: str = '../outputs') -> str:
    """
    Crea un directorio de salida con fecha y hora para guardar resultados.
    
    Args:
        base_dir: Directorio base donde crear el directorio de salida
    
    Returns:
        Ruta al directorio creado
    """
    # Crear directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Crear directorio con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir)
    
    logger.info(f"Directorio de salida creado: {output_dir}")
    return output_dir

def save_results(results: Dict, filename: str, output_dir: Optional[str] = None) -> str:
    """
    Guarda resultados en formato JSON.
    
    Args:
        results: Diccionario con resultados a guardar
        filename: Nombre del archivo
        output_dir: Directorio donde guardar (si es None, se crea uno nuevo)
    
    Returns:
        Ruta al archivo guardado
    """
    # Crear directorio si no se especifica
    if output_dir is None:
        output_dir = create_output_dir()
    
    # Asegurar extensión .json
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    # Ruta completa
    filepath = os.path.join(output_dir, filename)
    
    # Convertir objetos no serializables (DataFrames, arrays, etc.)
    def convert_to_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    # Crear versión serializable de los resultados
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)
    
    # Guardar en JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en: {filepath}")
    return filepath

def load_results(filepath: str) -> Dict:
    """
    Carga resultados desde un archivo JSON.
    
    Args:
        filepath: Ruta al archivo
    
    Returns:
        Diccionario con resultados cargados
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Resultados cargados desde: {filepath}")
        return results
    
    except Exception as e:
        logger.error(f"Error al cargar resultados: {e}")
        return {}

def filter_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Filtra outliers en las columnas especificadas.
    
    Args:
        df: DataFrame con los datos
        columns: Lista de columnas a filtrar
        method: Método para detectar outliers ('iqr' o 'zscore')
        threshold: Umbral para considerar outlier (para IQR, típicamente 1.5)
    
    Returns:
        DataFrame sin outliers
    """
    df_filtered = df.copy()
    initial_rows = len(df_filtered)
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Columna {col} no encontrada en el DataFrame")
            continue
        
        if method == 'iqr':
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                      (df_filtered[col] <= upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_filtered[col]))
            df_filtered = df_filtered[z_scores <= threshold]
            
        else:
            logger.warning(f"Método {method} no reconocido. Usando 'iqr'")
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                      (df_filtered[col] <= upper_bound)]
    
    filtered_rows = initial_rows - len(df_filtered)
    if filtered_rows > 0:
        logger.info(f"Se filtraron {filtered_rows} filas ({filtered_rows/initial_rows:.1%}) con outliers")
    
    return df_filtered

def identify_missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica patrones de valores faltantes en el DataFrame.
    
    Args:
        df: DataFrame a analizar
    
    Returns:
        DataFrame con análisis de valores faltantes
    """
    # Total de valores faltantes por columna
    missing_counts = df.isnull().sum()
    missing_percent = 100 * missing_counts / len(df)
    
    # Crear DataFrame de análisis
    missing_df = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_percent': missing_percent
    })
    
    # Ordenar por cantidad de faltantes
    missing_df = missing_df.sort_values('missing_count', ascending=False)
    
    # Filtrar solo columnas con faltantes
    missing_df = missing_df[missing_df['missing_count'] > 0]
    
    if len(missing_df) > 0:
        logger.info(f"Se encontraron valores faltantes en {len(missing_df)} columnas")
        
        # Detectar patrones de valores faltantes (correlación entre columnas con faltantes)
        if len(missing_df) > 1:
            # Convertir valores faltantes a 1, no faltantes a 0
            missing_patterns = df[missing_df.index].isnull().astype(int)
            
            # Calcular correlación entre patrones de valores faltantes
            pattern_corr = missing_patterns.corr()
            
            # Identificar correlaciones fuertes (>0.75)
            strong_correlations = pattern_corr.unstack().reset_index()
            strong_correlations.columns = ['col1', 'col2', 'correlation']
            strong_correlations = strong_correlations[
                (strong_correlations['correlation'] > 0.75) & 
                (strong_correlations['col1'] != strong_correlations['col2'])
            ]
            
            if len(strong_correlations) > 0:
                logger.info("Patrones de valores faltantes correlacionados:")
                for _, row in strong_correlations.iterrows():
                    logger.info(f"  {row['col1']} y {row['col2']}: {row['correlation']:.2f}")
    else:
        logger.info("No se encontraron valores faltantes en el DataFrame")
    
    return missing_df

def encode_cyclical_features(df: pd.DataFrame, features: List[str], 
                           max_values: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Codifica características cíclicas (día de la semana, mes, etc.) usando seno y coseno.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características cíclicas a codificar
        max_values: Diccionario con valor máximo para cada característica (opcional)
                    Por defecto: día semana=7, mes=12, día mes=31, hora=24, minuto=60
    
    Returns:
        DataFrame con características cíclicas codificadas
    """
    df_encoded = df.copy()
    
    # Valores máximos predeterminados
    default_max = {
        'day_of_week': 7,
        'month': 12,
        'day_of_month': 31,
        'hour': 24,
        'minute': 60,
        'second': 60
    }
    
    # Actualizar con valores proporcionados
    if max_values:
        default_max.update(max_values)
    
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Característica {feature} no encontrada en el DataFrame")
            continue
        
        # Determinar valor máximo
        if feature in default_max:
            max_value = default_max[feature]
        else:
            max_value = df[feature].max()
            logger.info(f"Usando valor máximo detectado para {feature}: {max_value}")
        
        # Codificar usando seno y coseno
        df_encoded[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / max_value)
        df_encoded[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / max_value)
        
        logger.info(f"Característica cíclica {feature} codificada como {feature}_sin y {feature}_cos")
    
    return df_encoded

def split_data_by_match(df: pd.DataFrame, match_id_col: str, test_size: float = 0.25, 
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos asegurando que todos los registros de un mismo partido estén
    en el mismo conjunto (entrenamiento o prueba). Útil para evitar data leakage.
    
    Args:
        df: DataFrame con los datos
        match_id_col: Nombre de la columna con ID de partido
        test_size: Proporción de datos para prueba
        random_state: Semilla aleatoria para reproducibilidad
    
    Returns:
        Tupla con (df_train, df_test)
    """
    if match_id_col not in df.columns:
        logger.error(f"Columna {match_id_col} no encontrada en el DataFrame")
        return df, pd.DataFrame()
    
    # Obtener ids de partidos únicos
    match_ids = df[match_id_col].unique()
    
    # Dividir a nivel de partido
    from sklearn.model_selection import train_test_split
    train_matches, test_matches = train_test_split(
        match_ids, test_size=test_size, random_state=random_state
    )
    
    # Dividir DataFrame
    df_train = df[df[match_id_col].isin(train_matches)]
    df_test = df[df[match_id_col].isin(test_matches)]
    
    logger.info(f"División por partidos: {len(train_matches)} partidos de entrenamiento, " 
               f"{len(test_matches)} partidos de prueba")
    logger.info(f"Registros: {len(df_train)} de entrenamiento, {len(df_test)} de prueba")
    
    return df_train, df_test

def compare_train_test_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                   features: List[str], figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compara la distribución de características entre conjuntos de entrenamiento y prueba.
    
    Args:
        train_df: DataFrame de entrenamiento
        test_df: DataFrame de prueba
        features: Lista de características a comparar
        figsize: Tamaño de la figura
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Número de características a comparar
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature not in train_df.columns or feature not in test_df.columns:
            logger.warning(f"Característica {feature} no encontrada en ambos DataFrames")
            continue
        
        ax = axes[i]
        
        # Para variables numéricas
        if train_df[feature].dtype in ['int64', 'float64']:
            # Histogramas superpuestos
            sns.histplot(train_df[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Train')
            sns.histplot(test_df[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Test')
        else:
            # Para variables categóricas
            train_counts = train_df[feature].value_counts(normalize=True)
            test_counts = test_df[feature].value_counts(normalize=True)
            
            # Unir y rellenar con ceros
            counts = pd.DataFrame({'Train': train_counts, 'Test': test_counts}).fillna(0)
            counts.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Distribución de {feature}')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # Ocultar ejes sin usar
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()