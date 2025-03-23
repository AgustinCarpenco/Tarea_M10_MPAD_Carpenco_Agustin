"""
Módulo para ingeniería de características de datos de fútbol.
Contiene funciones para crear y transformar características para los modelos.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea métricas de eficiencia basadas en ratios de acciones exitosas vs. intentadas.
    
    Args:
        df: DataFrame con estadísticas de jugadores
        
    Returns:
        DataFrame con métricas de eficiencia añadidas
    """
    df_result = df.copy()
    
    # Pares de métricas (intentos, exitosos) que podrían estar en los datos
    # Se adaptará según las columnas reales disponibles
    metric_pairs = [
        ('passes_attempted', 'passes_completed'),
        ('shots', 'shots_on_target'),
        ('tackles_attempted', 'tackles_won'),
        ('duels_total', 'duels_won'),
        ('aerial_duels', 'aerial_won'),
        ('dribbles_attempted', 'dribbles_completed')
    ]
    
    # Crear ratios de eficiencia para cada par
    for attempt_col, success_col in metric_pairs:
        if attempt_col in df.columns and success_col in df.columns:
            ratio_col = f'ratio_{success_col.split("_")[0]}'
            
            # Evitar divisiones por cero
            df_result[ratio_col] = np.where(
                df[attempt_col] > 0,
                df[success_col] / df[attempt_col],
                0
            )
            
            logger.info(f"Creada métrica de eficiencia: {ratio_col}")
    
    return df_result

def normalize_by_minutes(df: pd.DataFrame, minutes_col: str = 'minutes_played') -> pd.DataFrame:
    """
    Normaliza estadísticas acumulativas por minutos jugados (por 90 minutos).
    
    Args:
        df: DataFrame con estadísticas de jugadores
        minutes_col: Nombre de la columna con minutos jugados
        
    Returns:
        DataFrame con métricas normalizadas por minutos
    """
    if minutes_col not in df.columns:
        logger.warning(f"Columna {minutes_col} no encontrada. No se pudo normalizar por minutos.")
        return df
    
    df_result = df.copy()
    
    # Columnas a normalizar (numéricas que no son porcentajes o ratios)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    exclude_patterns = ['ratio', 'percentage', 'perc', 'id', 'match_id', 'player_id', minutes_col]
    
    cols_to_normalize = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
    
    # Normalizar por 90 minutos
    for col in cols_to_normalize:
        per90_col = f"{col}_per90"
        
        df_result[per90_col] = np.where(
            df[minutes_col] > 0,
            df[col] * 90 / df[minutes_col],
            0
        )
    
    logger.info(f"{len(cols_to_normalize)} columnas normalizadas por 90 minutos")
    return df_result

def create_performance_score(df: pd.DataFrame, position_col: Optional[str] = None) -> pd.DataFrame:
    """
    Crea una puntuación de rendimiento compuesta para cada jugador.
    
    Args:
        df: DataFrame con estadísticas de jugadores
        position_col: Nombre de la columna con la posición del jugador (opcional)
        
    Returns:
        DataFrame con puntuación de rendimiento añadida
    """
    df_result = df.copy()
    
    # Métricas ofensivas
    offensive_patterns = ['goal', 'assist', 'shot', 'chance', 'pass', 'dribble', 'cross']
    offensive_cols = [col for col in df.columns 
                     if any(pattern in col.lower() for pattern in offensive_patterns)
                     and col.endswith('_per90')]
    
    # Métricas defensivas
    defensive_patterns = ['tackle', 'interception', 'clearance', 'block', 'recovery', 'duel_won']
    defensive_cols = [col for col in df.columns 
                     if any(pattern in col.lower() for pattern in defensive_patterns)
                     and col.endswith('_per90')]
    
    # Si hay posiciones disponibles, crear puntuación específica por posición
    if position_col and position_col in df.columns:
        # Inicializar columna de puntuación
        df_result['performance_score'] = 0
        
        # Definir posiciones
        positions = df[position_col].dropna().unique()
        
        for position in positions:
            mask = df[position_col] == position
            
            # Seleccionar métricas relevantes según la posición
            if any(pos in str(position).lower() for pos in ['goalkeeper', 'gk', 'portero']):
                # Métricas para porteros (a adaptar según datos reales)
                relevant_cols = [col for col in df.columns if any(s in col.lower() 
                                                                 for s in ['save', 'clean_sheet', 'goals_conceded'])]
                
                if relevant_cols:
                    # Normalizar y combinar
                    temp_df = df.loc[mask, relevant_cols].copy()
                    if len(temp_df) > 1:  # Necesitamos al menos 2 filas para escalar
                        scaler = MinMaxScaler()
                        temp_df = pd.DataFrame(scaler.fit_transform(temp_df), 
                                              columns=relevant_cols, 
                                              index=temp_df.index)
                        df_result.loc[mask, 'performance_score'] = temp_df.mean(axis=1)
            
            elif any(pos in str(position).lower() for pos in ['defend', 'back', 'cb', 'lb', 'rb']):
                # Métricas para defensas
                if defensive_cols:
                    temp_df = df.loc[mask, defensive_cols].copy()
                    if len(temp_df) > 1:
                        scaler = MinMaxScaler()
                        temp_df = pd.DataFrame(scaler.fit_transform(temp_df), 
                                              columns=defensive_cols, 
                                              index=temp_df.index)
                        df_result.loc[mask, 'performance_score'] = temp_df.mean(axis=1)
            
            elif any(pos in str(position).lower() for pos in ['midfield', 'cm', 'cdm', 'dm']):
                # Métricas para centrocampistas (balance defensivo-ofensivo)
                relevant_cols = defensive_cols + offensive_cols
                if relevant_cols:
                    temp_df = df.loc[mask, relevant_cols].copy()
                    if len(temp_df) > 1:
                        scaler = MinMaxScaler()
                        temp_df = pd.DataFrame(scaler.fit_transform(temp_df), 
                                              columns=relevant_cols, 
                                              index=temp_df.index)
                        df_result.loc[mask, 'performance_score'] = temp_df.mean(axis=1)
            
            elif any(pos in str(position).lower() for pos in ['attack', 'forward', 'wing', 'striker']):
                # Métricas para atacantes
                if offensive_cols:
                    temp_df = df.loc[mask, offensive_cols].copy()
                    if len(temp_df) > 1:
                        scaler = MinMaxScaler()
                        temp_df = pd.DataFrame(scaler.fit_transform(temp_df), 
                                              columns=offensive_cols, 
                                              index=temp_df.index)
                        df_result.loc[mask, 'performance_score'] = temp_df.mean(axis=1)
    
    else:
        # Si no hay información de posición, crear una puntuación general
        relevant_cols = offensive_cols + defensive_cols
        
        if relevant_cols:
            # Normalizar todas las columnas
            temp_df = df[relevant_cols].copy()
            if len(temp_df) > 1:
                scaler = MinMaxScaler()
                temp_df = pd.DataFrame(scaler.fit_transform(temp_df), 
                                      columns=relevant_cols, 
                                      index=temp_df.index)
                
                # Calcular puntuación como media de las métricas normalizadas
                df_result['performance_score'] = temp_df.mean(axis=1)
    
    # Crear etiqueta binaria de rendimiento alto/bajo
    if 'performance_score' in df_result.columns:
        # Punto de corte en el percentil 70
        threshold = df_result['performance_score'].quantile(0.7)
        df_result['high_performance'] = df_result['performance_score'] >= threshold
        
        logger.info("Creada puntuación de rendimiento y etiqueta binaria")
    
    return df_result

def prepare_features_for_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara características finales para modelos de machine learning.
    
    Args:
        df: DataFrame con estadísticas de jugadores
        
    Returns:
        DataFrame con características preparadas
    """
    # Copia para no modificar el original
    df_features = df.copy()
    
    # 1. Identificar columnas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Excluir columnas de identificación
    id_patterns = ['id', 'name', 'player_name', 'team_name']
    categorical_cols = [col for col in categorical_cols 
                       if not any(pattern in col.lower() for pattern in id_patterns)]
    
    # 2. Convertir columnas categóricas a variables dummy
    if categorical_cols:
        logger.info(f"Convirtiendo {len(categorical_cols)} columnas categóricas a variables dummy")
        df_dummies = pd.get_dummies(df_features[categorical_cols], drop_first=True)
        
        # Eliminar columnas originales y añadir dummies
        df_features = df_features.drop(columns=categorical_cols)
        df_features = pd.concat([df_features, df_dummies], axis=1)
    
    # 3. Manejar valores faltantes
    # Rellenar con 0 los valores faltantes en columnas numéricas
    numeric_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
    
    # 4. Eliminar columnas con todos valores nulos
    df_features = df_features.dropna(axis=1, how='all')
    
    logger.info(f"DataFrame de características preparado con {df_features.shape[1]} columnas")
    return df_features