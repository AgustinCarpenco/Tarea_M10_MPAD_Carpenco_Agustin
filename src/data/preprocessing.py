"""
Módulo de preprocesamiento para datos de fútbol.
Contiene funciones para cargar, limpiar y transformar datos de partidos.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_match_data(file_path: str) -> dict:
    """
    Carga datos de partidos desde un archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Diccionario con los datos cargados
    """
    logger.info(f"Cargando datos desde {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("Datos cargados correctamente")
        return data
    
    except Exception as e:
        logger.error(f"Error al cargar el archivo: {e}")
        raise

def extract_matches_dataframe(data: Union[dict, list]) -> pd.DataFrame:
    """
    Extrae información de partidos a un DataFrame.
    
    Args:
        data: Datos cargados del archivo JSON
        
    Returns:
        DataFrame con información de partidos
    """
    matches = []
    
    # Procesamiento según la estructura específica del JSON
    # Esta función se adaptará una vez que conozcamos la estructura exacta
    
    logger.info("Extrayendo información de partidos")
    
    # Ejemplo genérico para estructura común
    if isinstance(data, list):
        # Si es una lista de partidos
        for match in data:
            if isinstance(match, dict):
                # Extraer información básica del partido
                match_info = {}
                
                # Incluir campos simples (no listas ni diccionarios)
                for key, value in match.items():
                    if not isinstance(value, (dict, list)):
                        match_info[key] = value
                
                # Añadir información de equipos si está disponible
                if 'home_team' in match and isinstance(match['home_team'], dict):
                    home_team = match['home_team']
                    if 'name' in home_team:
                        match_info['home_team_name'] = home_team['name']
                    if 'score' in home_team:
                        match_info['home_score'] = home_team['score']
                
                if 'away_team' in match and isinstance(match['away_team'], dict):
                    away_team = match['away_team']
                    if 'name' in away_team:
                        match_info['away_team_name'] = away_team['name']
                    if 'score' in away_team:
                        match_info['away_score'] = away_team['score']
                
                matches.append(match_info)
    
    # Crear DataFrame
    if matches:
        matches_df = pd.DataFrame(matches)
        logger.info(f"DataFrame de partidos creado con {len(matches_df)} filas y {len(matches_df.columns)} columnas")
        return matches_df
    else:
        logger.warning("No se pudo extraer información de partidos")
        return pd.DataFrame()

def extract_players_dataframe(data: Union[dict, list]) -> pd.DataFrame:
    """
    Extrae información de jugadores a un DataFrame.
    
    Args:
        data: Datos cargados del archivo JSON
        
    Returns:
        DataFrame con información de jugadores por partido
    """
    players = []
    
    # Esta función se adaptará una vez que conozcamos la estructura exacta
    
    logger.info("Extrayendo información de jugadores")
    
    # Ejemplo genérico para estructura común
    if isinstance(data, list):
        # Si es una lista de partidos
        for match in data:
            match_id = match.get('id', None)
            
            # Buscar jugadores en diferentes ubicaciones posibles
            
            # Opción 1: Lista de jugadores directamente en el partido
            if 'players' in match and isinstance(match['players'], list):
                for player in match['players']:
                    if isinstance(player, dict):
                        player_info = player.copy()
                        player_info['match_id'] = match_id
                        players.append(player_info)
            
            # Opción 2: Jugadores dentro de equipos
            for team_type in ['home_team', 'away_team']:
                if team_type in match and isinstance(match[team_type], dict):
                    team = match[team_type]
                    team_name = team.get('name', team_type)
                    
                    if 'players' in team and isinstance(team['players'], list):
                        for player in team['players']:
                            if isinstance(player, dict):
                                player_info = player.copy()
                                player_info['match_id'] = match_id
                                player_info['team'] = team_name
                                players.append(player_info)
    
    # Crear DataFrame
    if players:
        players_df = pd.DataFrame(players)
        logger.info(f"DataFrame de jugadores creado con {len(players_df)} filas y {len(players_df.columns)} columnas")
        return players_df
    else:
        logger.warning("No se pudo extraer información de jugadores")
        return pd.DataFrame()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame eliminando duplicados y tratando valores nulos.
    
    Args:
        df: DataFrame a limpiar
        
    Returns:
        DataFrame limpio
    """
    # Copiar el DataFrame para no modificar el original
    df_clean = df.copy()
    
    # Eliminar duplicados
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    
    if len(df_clean) < initial_rows:
        logger.info(f"Se eliminaron {initial_rows - len(df_clean)} filas duplicadas")
    
    # Contar valores nulos
    null_counts = df_clean.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if not columns_with_nulls.empty:
        logger.info(f"Columnas con valores nulos:\n{columns_with_nulls}")
    
    return df_clean

def process_match_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa el archivo de datos completo y retorna DataFrames limpios.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Tupla con (DataFrame de partidos, DataFrame de jugadores)
    """
    # Cargar datos
    data = load_match_data(file_path)
    
    # Extraer DataFrames
    matches_df = extract_matches_dataframe(data)
    players_df = extract_players_dataframe(data)
    
    # Limpiar DataFrames
    matches_df = clean_dataframe(matches_df)
    players_df = clean_dataframe(players_df)
    
    return matches_df, players_df