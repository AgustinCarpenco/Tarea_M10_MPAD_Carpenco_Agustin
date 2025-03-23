"""
Módulo para visualización de datos y resultados de modelos.
Contiene funciones para crear diferentes tipos de gráficos.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración global de visualización
plt.style.use('seaborn-whitegrid')
sns.set_palette('viridis')

def plot_feature_distributions(df: pd.DataFrame, features: List[str], 
                              by_target: Optional[str] = None,
                              figsize: Tuple[int, int] = (16, 12),
                              n_cols: int = 3) -> None:
    """
    Visualiza la distribución de características, opcionalmente agrupadas por una variable objetivo.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características a visualizar
        by_target: Nombre de la columna objetivo para colorear (opcional)
        figsize: Tamaño de la figura
        n_cols: Número de columnas en la cuadrícula
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if by_target and by_target in df.columns:
            # Gráfico con color según variable objetivo
            if df[by_target].dtype in ['bool', 'int64', 'int32'] and df[by_target].nunique() <= 5:
                # Para variables categóricas/binarias
                for target_val in sorted(df[by_target].unique()):
                    subset = df[df[by_target] == target_val]
                    sns.kdeplot(subset[feature], shade=True, alpha=0.5, 
                               label=f'{by_target}={target_val}', ax=ax)
            else:
                # Para variables continuas, usar un scatterplot
                sns.scatterplot(x=feature, y=by_target, data=df, ax=ax)
        else:
            # Gráfico de distribución simple
            if df[feature].dtype in ['int64', 'float64']:
                sns.histplot(df[feature], kde=True, ax=ax)
            else:
                # Para variables categóricas
                sns.countplot(y=feature, data=df, ax=ax)
        
        ax.set_title(f'Distribución de {feature}')
        ax.tick_params(axis='x', rotation=45)
    
    # Ocultar ejes sin usar
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, features: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (12, 10),
                          cmap: str = 'coolwarm') -> None:
    """
    Visualiza la matriz de correlación entre características.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características a incluir (si es None, usa todas las numéricas)
        figsize: Tamaño de la figura
        cmap: Mapa de colores para la visualización
    """
    # Si no se especifican características, usar todas las numéricas
    if features is None:
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calcular correlaciones
    corr_matrix = df[features].corr()
    
    # Configurar figura
    plt.figure(figsize=figsize)
    
    # Crear mapa de calor
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=0.5,
        fmt='.2f'
    )
    
    plt.title('Matriz de Correlación')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame, features: List[str], by_category: Optional[str] = None,
                figsize: Tuple[int, int] = (16, 12), n_cols: int = 2) -> None:
    """
    Crea boxplots para visualizar la distribución de características, opcionalmente agrupadas por categoría.
    
    Args:
        df: DataFrame con los datos
        features: Lista de características a visualizar
        by_category: Nombre de la columna categórica para agrupar (opcional)
        figsize: Tamaño de la figura
        n_cols: Número de columnas en la cuadrícula
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if by_category and by_category in df.columns:
            # Boxplot agrupado por categoría
            sns.boxplot(x=by_category, y=feature, data=df, ax=ax)
            ax.set_title(f'Distribución de {feature} por {by_category}')
            ax.set_xlabel(by_category)
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)
        else:
            # Boxplot simple
            sns.boxplot(y=feature, data=df, ax=ax)
            ax.set_title(f'Distribución de {feature}')
            ax.set_xlabel('')
            ax.set_ylabel(feature)
        
    # Ocultar ejes sin usar
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza la importancia de características de un modelo.
    
    Args:
        importance_df: DataFrame con columnas 'feature' e 'importance'
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
    """
    # Ordenar por importancia
    sorted_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=sorted_df)
    plt.title(f'Top {top_n} Características más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: Union[List, np.ndarray, pd.Series], 
                         y_pred: Union[List, np.ndarray, pd.Series],
                         class_names: List[str] = ['Bajo', 'Alto'],
                         figsize: Tuple[int, int] = (8, 6),
                         normalize: bool = False) -> None:
    """
    Visualiza una matriz de confusión.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        class_names: Nombres de las clases
        figsize: Tamaño de la figura
        normalize: Si se normaliza la matriz (proporciones en lugar de conteos)
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar si se solicita
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: Union[List, np.ndarray, pd.Series], 
                  y_prob: Union[List, np.ndarray, pd.Series],
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualiza la curva ROC.
    
    Args:
        y_true: Valores reales
        y_prob: Probabilidades de la clase positiva
        figsize: Tamaño de la figura
    """
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    # Calcular AUC
    auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true: Union[List, np.ndarray, pd.Series],
                              y_prob: Union[List, np.ndarray, pd.Series],
                              figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualiza la curva de precisión-recall.
    
    Args:
        y_true: Valores reales
        y_prob: Probabilidades de la clase positiva
        figsize: Tamaño de la figura
    """
    # Calcular curva precision-recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    # Calcular AUC
    auc = np.trapz(precision, recall)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'AUC = {auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_2d(X_reduced: pd.DataFrame, cluster_labels: pd.Series, 
                   title: str = 'Visualización de Clusters',
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualiza clusters en un espacio bidimensional.
    
    Args:
        X_reduced: DataFrame con datos reducidos a 2 dimensiones
        cluster_labels: Etiquetas de cluster
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    if X_reduced.shape[1] != 2:
        logger.warning("Esta función espera datos reducidos a 2 dimensiones")
        return
    
    plt.figure(figsize=figsize)
    
    # Obtener nombres de columnas para ejes
    x_col, y_col = X_reduced.columns
    
    # Crear scatter plot con colores por cluster
    sns.scatterplot(
        x=x_col, 
        y=y_col,
        hue=cluster_labels,
        palette='viridis',
        data=pd.concat([X_reduced, cluster_labels], axis=1),
        legend='full',
        alpha=0.8
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cluster_sizes(cluster_labels: pd.Series, 
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualiza el tamaño de cada cluster.
    
    Args:
        cluster_labels: Serie con etiquetas de cluster
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Contar elementos por cluster
    cluster_counts = cluster_labels.value_counts().sort_index()
    
    # Crear gráfico de barras
    ax = cluster_counts.plot(kind='bar', color='skyblue')
    
    # Añadir etiquetas con el número exacto y porcentaje
    total = len(cluster_labels)
    for i, count in enumerate(cluster_counts):
        percentage = count / total * 100
        ax.text(i, count + 0.1, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom')
    
    plt.title('Distribución de elementos por cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Número de elementos')
    plt.tight_layout()
    plt.show()

def plot_radar_chart(stats_df: pd.DataFrame, categories: List[str], 
                    group_col: str = 'cluster',
                    figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Crea un gráfico de radar para comparar grupos (clusters) en diferentes categorías.
    
    Args:
        stats_df: DataFrame con estadísticas
        categories: Lista de categorías a incluir en el radar
        group_col: Nombre de la columna para agrupar (cluster, posición, etc.)
        figsize: Tamaño de la figura
    """
    # Verificar que todas las categorías existan
    for cat in categories:
        if cat not in stats_df.columns:
            logger.warning(f"Categoría {cat} no encontrada en el DataFrame")
            return
    
    # Número de categorías
    N = len(categories)
    
    # Obtener grupos únicos
    groups = sorted(stats_df[group_col].unique())
    
    # Crear figura
    fig = plt.figure(figsize=figsize)
    
    # Calcular ángulos para el radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el círculo
    
    # Crear un subplot con coordenadas polares
    ax = fig.add_subplot(111, polar=True)
    
    # Añadir categorías como etiquetas
    plt.xticks(angles[:-1], categories, size=12)
    
    # Colores para diferentes grupos
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    
    # Dibujar el radar para cada grupo
    for i, group in enumerate(groups):
        # Filtrar datos para el grupo
        group_data = stats_df[stats_df[group_col] == group]
        
        # Verificar que hay datos
        if len(group_data) == 0:
            continue
        
        # Obtener valores
        values = group_data[categories].mean().values.flatten().tolist()
        values += values[:1]  # Cerrar el círculo
        
        # Dibujar
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
               label=f'{group_col}={group}', color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Añadir leyenda
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Comparación de perfiles por categorías', size=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()