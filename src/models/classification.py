"""
Módulo para el modelo de clasificación de rendimiento de jugadores.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara la división de datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Características
        y: Variable objetivo
        test_size: Proporción de datos para prueba
        random_state: Semilla aleatoria para reproducibilidad
        
    Returns:
        Tupla con (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"División de datos: {X_train.shape[0]} muestras de entrenamiento, "
               f"{X_test.shape[0]} muestras de prueba")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       param_grid: Optional[Dict] = None) -> Tuple[RandomForestClassifier, Dict]:
    """
    Entrena un modelo Random Forest con búsqueda de hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        param_grid: Rejilla de hiperparámetros para búsqueda
        
    Returns:
        Tupla con (mejor modelo, mejores parámetros)
    """
    # Valores predeterminados para la rejilla de parámetros
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Crear pipeline con escalado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Ajustar nombres de parámetros para el pipeline
    param_grid_pipe = {f'rf__{key}': value for key, value in param_grid.items()}
    
    # Búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        pipeline, param_grid_pipe, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    logger.info("Iniciando entrenamiento de Random Forest con búsqueda de hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Mejor puntuación F1: {grid_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    
    # Extraer el modelo RF del pipeline
    best_model = grid_search.best_estimator_.named_steps['rf']
    best_params = {key.replace('rf__', ''): value 
                   for key, value in grid_search.best_params_.items()}
    
    return best_model, best_params

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                 param_grid: Optional[Dict] = None) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Entrena un modelo XGBoost con búsqueda de hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        param_grid: Rejilla de hiperparámetros para búsqueda
        
    Returns:
        Tupla con (mejor modelo, mejores parámetros)
    """
    # Valores predeterminados para la rejilla de parámetros
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    # Crear pipeline con escalado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    
    # Ajustar nombres de parámetros para el pipeline
    param_grid_pipe = {f'xgb__{key}': value for key, value in param_grid.items()}
    
    # Búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        pipeline, param_grid_pipe, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    logger.info("Iniciando entrenamiento de XGBoost con búsqueda de hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Mejor puntuación F1: {grid_search.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    
    # Extraer el modelo XGB del pipeline
    best_model = grid_search.best_estimator_.named_steps['xgb']
    best_params = {key.replace('xgb__', ''): value 
                   for key, value in grid_search.best_params_.items()}
    
    return best_model, best_params

def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evalúa un modelo de clasificación con métricas estándar.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Target de prueba
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    
    # Mostrar resultados
    logger.info(f"Métricas de evaluación:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    logger.info(f"Matriz de confusión:\n{metrics['confusion_matrix']}")
    
    return metrics

def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 20, 
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza la importancia de características de un modelo.
    
    Args:
        model: Modelo entrenado con atributo feature_importances_
        feature_names: Nombres de las características
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("El modelo no tiene atributo feature_importances_")
        return
    
    # Obtener importancia de características
    importances = model.feature_importances_
    
    # Crear DataFrame con importancias
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Ordenar por importancia
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Seleccionar top_n características
    top_features = feature_importance.head(top_n)
    
    # Visualizar
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Características más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str] = ['Bajo', 'Alto'],
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualiza una matriz de confusión.
    
    Args:
        conf_matrix: Matriz de confusión
        class_names: Nombres de las clases
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualiza la curva ROC.
    
    Args:
        y_test: Valores reales
        y_prob: Probabilidades predichas
        figsize: Tamaño de la figura
    """
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    # Visualizar
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.tight_layout()
    plt.show()