#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la visualisation des résultats du modèle de prédiction.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import joblib
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, output_dir: str = "visualizations/model"):
        """
        Initialise le visualiseur de modèle.
        
        Args:
            output_dir (str): Répertoire de sortie pour les visualisations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration du style des graphiques
        sns.set_theme()
        
    def plot_predictions_vs_actual(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 title: str = "Prédictions vs Valeurs Réelles") -> None:
        """
        Crée un graphique comparant les prédictions aux valeurs réelles.
        
        Args:
            y_true (np.ndarray): Valeurs réelles
            y_pred (np.ndarray): Prédictions du modèle
            title (str): Titre du graphique
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        
        plt.xlabel('Valeurs Réelles (TWh)')
        plt.ylabel('Prédictions (TWh)')
        plt.title(title)
        
        # Calculer et afficher R²
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"predictions_vs_actual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
    def plot_feature_importance(self, 
                              model,
                              feature_names: List[str],
                              title: str = "Importance des Variables") -> None:
        """
        Crée un graphique de l'importance des features.
        
        Args:
            model: Modèle entraîné avec attribut feature_importances_
            feature_names (List[str]): Noms des features
            title (str): Titre du graphique
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Le modèle n'a pas d'attribut feature_importances_")
            return
            
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, 
                  ha='right')
        
        plt.xlabel('Variables')
        plt.ylabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
    def plot_residuals(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      title: str = "Analyse des Résidus") -> None:
        """
        Crée un graphique d'analyse des résidus.
        
        Args:
            y_true (np.ndarray): Valeurs réelles
            y_pred (np.ndarray): Prédictions du modèle
            title (str): Titre du graphique
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Graphique des résidus vs prédictions
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Prédictions')
        ax1.set_ylabel('Résidus')
        ax1.set_title('Résidus vs Prédictions')
        
        # Histogramme des résidus
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Résidus')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des Résidus')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"residuals_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
    def plot_predictions_by_country(self, 
                                  predictions_df: pd.DataFrame,
                                  title: str = "Évolution des Prédictions par Pays") -> None:
        """
        Crée un graphique de l'évolution des prédictions par pays.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame contenant les prédictions
            title (str): Titre du graphique
        """
        plt.figure(figsize=(12, 6))
        
        for country in predictions_df['Country'].unique():
            country_data = predictions_df[predictions_df['Country'] == country]
            plt.plot(country_data['Year'], 
                    country_data['Predicted_Total_Energy_Consumption'],
                    label=country,
                    marker='o',
                    markersize=4)
        
        plt.xlabel('Année')
        plt.ylabel('Consommation Énergétique Prédite (TWh)')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"predictions_by_country_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def generate_all_visualizations(model_path: str,
                              predictions_path: str,
                              historical_data_path: str) -> None:
    """
    Génère toutes les visualisations pour le modèle.
    
    Args:
        model_path (str): Chemin vers le modèle sauvegardé
        predictions_path (str): Chemin vers les prédictions
        historical_data_path (str): Chemin vers les données historiques
    """
    try:
        # Charger le modèle et les données
        model = joblib.load(model_path)
        predictions_df = pd.read_csv(predictions_path)
        historical_df = pd.read_csv(historical_data_path)
        
        # Initialiser le visualiseur
        visualizer = ModelVisualizer()
        
        # Charger les features
        features_path = Path("models/features.joblib")
        if features_path.exists():
            features = joblib.load(features_path)
        else:
            logger.warning("Fichier des features non trouvé")
            features = None
        
        # Générer les visualisations
        if features is not None:
            visualizer.plot_feature_importance(model, features)
        
        # Pour les résidus et prédictions vs réelles, nous avons besoin des données historiques
        if 'Total Energy Consumption (TWh)' in historical_df.columns:
            y_true = historical_df['Total Energy Consumption (TWh)'].values
            # Calculer les prédictions sur les données historiques
            X_historical = historical_df.drop(columns=['Total Energy Consumption (TWh)'])
            if features is not None:
                X_historical = X_historical[features]
            y_pred_historical = model.predict(X_historical)
            
            visualizer.plot_predictions_vs_actual(y_true, y_pred_historical)
            visualizer.plot_residuals(y_true, y_pred_historical)
        
        # Visualiser les prédictions futures par pays
        visualizer.plot_predictions_by_country(predictions_df)
        
        logger.info("Toutes les visualisations ont été générées avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations : {str(e)}")
        raise

if __name__ == "__main__":
    # Exemple d'utilisation
    model_path = "models/best_model.joblib"
    predictions_path = "predictions/predictions_2024_2039_latest.csv"
    historical_data_path = "data/processed/global_energy_consumption.csv"
    
    generate_all_visualizations(model_path, predictions_path, historical_data_path) 