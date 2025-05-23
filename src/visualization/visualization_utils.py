import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuration du style
plt.style.use('default')
sns.set_palette("husl")

class EnergyVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialise le visualiseur avec les données.
        
        Args:
            data (pd.DataFrame): DataFrame contenant les données énergétiques
        """
        self.data = data
        self.colors = sns.color_palette("husl", 8)
        
    def plot_temporal_evolution(self, 
                              date_col: str,
                              value_col: str,
                              title: str = "Évolution de la Consommation Énergétique",
                              save_path: Optional[str] = None) -> None:
        """
        Trace l'évolution temporelle de la consommation énergétique.
        
        Args:
            date_col (str): Nom de la colonne de dates
            value_col (str): Nom de la colonne de valeurs
            title (str): Titre du graphique
            save_path (str, optional): Chemin pour sauvegarder le graphique
        """
        plt.figure(figsize=(15, 8))
        
        # Tracer la série temporelle
        plt.plot(self.data[date_col], self.data[value_col], 
                color=self.colors[0], linewidth=2)
        
        # Ajouter une tendance
        z = np.polyfit(range(len(self.data)), self.data[value_col], 1)
        p = np.poly1d(z)
        plt.plot(self.data[date_col], p(range(len(self.data))), 
                "--", color=self.colors[1], alpha=0.8, 
                label="Tendance")
        
        # Personnalisation
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Consommation (kWh)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotation des dates pour une meilleure lisibilité
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_correlation_matrix(self,
                              columns: List[str],
                              title: str = "Matrice de Corrélation",
                              save_path: Optional[str] = None) -> None:
        """
        Génère une heatmap de corrélation.
        
        Args:
            columns (List[str]): Liste des colonnes à inclure
            title (str): Titre du graphique
            save_path (str, optional): Chemin pour sauvegarder le graphique
        """
        plt.figure(figsize=(12, 10))
        
        # Calcul de la matrice de corrélation
        corr_matrix = self.data[columns].corr()
        
        # Création de la heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   square=True,
                   linewidths=.5)
        
        plt.title(title, fontsize=14, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_feature_importance(self,
                              feature_names: List[str],
                              importance_values: np.ndarray,
                              title: str = "Importance des Variables",
                              save_path: Optional[str] = None) -> None:
        """
        Trace l'importance des variables.
        
        Args:
            feature_names (List[str]): Noms des variables
            importance_values (np.ndarray): Valeurs d'importance
            title (str): Titre du graphique
            save_path (str, optional): Chemin pour sauvegarder le graphique
        """
        plt.figure(figsize=(12, 8))
        
        # Création du graphique à barres
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Feature'], 
                importance_df['Importance'],
                color=self.colors)
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel("Importance (%)", fontsize=12)
        plt.ylabel("Variables", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_model_performance(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str,
                             save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Trace et calcule les métriques de performance du modèle.
        
        Args:
            y_true (np.ndarray): Valeurs réelles
            y_pred (np.ndarray): Valeurs prédites
            model_name (str): Nom du modèle
            save_path (str, optional): Chemin pour sauvegarder le graphique
            
        Returns:
            Dict[str, float]: Dictionnaire des métriques de performance
        """
        plt.figure(figsize=(10, 8))
        
        # Calcul des métriques
        metrics = {
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        }
        
        # Tracer les prédictions vs valeurs réelles
        plt.scatter(y_true, y_pred, alpha=0.5, color=self.colors[0])
        
        # Ligne de référence (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                '--', color='red', alpha=0.8)
        
        # Personnalisation
        plt.title(f"Performance du Modèle {model_name}", fontsize=14, pad=20)
        plt.xlabel("Valeurs Réelles", fontsize=12)
        plt.ylabel("Valeurs Prédites", fontsize=12)
        
        # Ajouter les métriques sur le graphique
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        plt.text(0.05, 0.95, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return metrics
        
    def plot_prediction_intervals(self,
                                dates: np.ndarray,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_lower: np.ndarray,
                                y_upper: np.ndarray,
                                title: str = "Intervalles de Prédiction",
                                save_path: Optional[str] = None) -> None:
        """
        Trace les intervalles de prédiction.
        
        Args:
            dates (np.ndarray): Dates des prédictions
            y_true (np.ndarray): Valeurs réelles
            y_pred (np.ndarray): Valeurs prédites
            y_lower (np.ndarray): Bornes inférieures
            y_upper (np.ndarray): Bornes supérieures
            title (str): Titre du graphique
            save_path (str, optional): Chemin pour sauvegarder le graphique
        """
        plt.figure(figsize=(15, 8))
        
        # Tracer les intervalles de confiance
        plt.fill_between(dates, y_lower, y_upper,
                        color=self.colors[0], alpha=0.2,
                        label="Intervalle de confiance")
        
        # Tracer les prédictions
        plt.plot(dates, y_pred, color=self.colors[0],
                linewidth=2, label="Prédictions")
        
        # Tracer les valeurs réelles
        plt.scatter(dates, y_true, color=self.colors[1],
                   alpha=0.5, label="Valeurs réelles")
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Consommation (kWh)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotation des dates
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def create_interactive_map(self,
                             lat_col: str,
                             lon_col: str,
                             value_col: str,
                             title: str = "Carte Interactive de la Performance Énergétique",
                             save_path: Optional[str] = None) -> None:
        """
        Crée une carte interactive avec Plotly.
        
        Args:
            lat_col (str): Nom de la colonne de latitude
            lon_col (str): Nom de la colonne de longitude
            value_col (str): Nom de la colonne de valeurs
            title (str): Titre de la carte
            save_path (str, optional): Chemin pour sauvegarder la carte
        """
        fig = px.scatter_mapbox(self.data,
                              lat=lat_col,
                              lon=lon_col,
                              color=value_col,
                              size=value_col,
                              hover_name=value_col,
                              zoom=5,
                              mapbox_style="carto-positron",
                              title=title)
        
        if save_path:
            fig.write_html(save_path)
        return fig 