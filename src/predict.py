#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour prédire la consommation énergétique sur les 15 prochaines années.
"""

import sys
from pathlib import Path
# Ajout du répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from src.data import DataLoader, DataPreprocessor
from src.visualization.model_visualizations import generate_all_visualizations

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_future_data(df: pd.DataFrame, years_ahead: int = 15) -> pd.DataFrame:
    """
    Prépare les données pour les années futures en extrapolant les tendances.
    
    Args:
        df (pd.DataFrame): DataFrame original
        years_ahead (int): Nombre d'années à prédire
        
    Returns:
        pd.DataFrame: DataFrame avec les données extrapolées
    """
    # Obtenir la dernière année dans les données
    last_year = df['Year'].max()
    
    # Créer un DataFrame pour les années futures
    future_years = pd.DataFrame({
        'Year': range(last_year + 1, last_year + years_ahead + 1)
    })
    
    # Pour chaque pays, extrapoler les tendances
    future_data = []
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].copy()
        
        # Calculer les tendances pour chaque variable numérique
        trends = {}
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col != 'Year':
                # Calculer la moyenne mobile sur 3 ans pour lisser les tendances
                country_data[f'{col}_trend'] = country_data[col].rolling(window=3, min_periods=1).mean()
                # Calculer la pente de la tendance
                x = np.array(range(len(country_data)))
                y = country_data[f'{col}_trend'].values
                slope = np.polyfit(x, y, 1)[0]
                trends[col] = slope
        
        # Créer les données futures pour ce pays
        country_future = future_years.copy()
        country_future['Country'] = country
        
        # Extrapoler chaque variable en utilisant les tendances
        for col, trend in trends.items():
            last_value = country_data[col].iloc[-1]
            years_diff = country_future['Year'] - last_year
            country_future[col] = last_value + (trend * years_diff)
        
        future_data.append(country_future)
    
    # Combiner toutes les données futures
    future_df = pd.concat(future_data, ignore_index=True)
    
    # S'assurer que les valeurs restent dans des limites raisonnables
    for col in future_df.select_dtypes(include=['float64']).columns:
        if 'Share' in col or 'Dependency' in col:
            future_df[col] = future_df[col].clip(0, 100)
        elif 'Price' in col:
            future_df[col] = future_df[col].clip(0)
    
    return future_df

def main():
    """Fonction principale pour générer les prédictions."""
    try:
        logger.info("Démarrage de la génération des prédictions")
        
        # 1. Charger le modèle entraîné
        model_path = Path("models/best_model.joblib")
        if not model_path.exists():
            logger.error("Le modèle entraîné n'existe pas. Veuillez d'abord entraîner le modèle.")
            return
        
        model = joblib.load(model_path)
        logger.info("Modèle chargé avec succès")
        
        # 2. Charger les données originales
        data_loader = DataLoader()
        df = data_loader.load_data("global_energy_consumption.csv")
        if df is None:
            logger.error("Impossible de charger les données originales")
            return
        
        # 3. Préparer les données futures
        future_data = prepare_future_data(df, years_ahead=15)
        logger.info("Données futures préparées")
        
        # 4. Prétraiter les données futures
        if 'Total Energy Consumption (TWh)' in future_data.columns:
            future_data = future_data.drop(columns=['Total Energy Consumption (TWh)'])
        preprocessor = DataPreprocessor()
        X_future, _ = preprocessor.preprocess_data(future_data)

        # Charger la liste des features sauvegardée et réordonner X_future
        features_path = Path("models/features.joblib")
        if features_path.exists():
            features = joblib.load(features_path)
            X_future = X_future[features]
        else:
            logger.warning("Fichier des features non trouvé. Les prédictions pourraient échouer si l'ordre diffère.")
        
        # 5. Générer les prédictions
        predictions = model.predict(X_future)
        
        # 6. Ajouter les prédictions aux données futures
        future_data['Predicted_Total_Energy_Consumption'] = predictions
        
        # 7. Sauvegarder les prédictions
        output_path = Path("predictions") / f"predictions_2024_2039_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(exist_ok=True)
        future_data.to_csv(output_path, index=False)
        
        # 8. Générer un résumé des prédictions par pays
        summary = future_data.groupby('Country').agg({
            'Year': ['min', 'max'],
            'Predicted_Total_Energy_Consumption': ['mean', 'min', 'max']
        }).round(2)
        
        summary_path = Path("predictions") / f"summary_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary.to_csv(summary_path)
        
        logger.info(f"Prédictions sauvegardées dans {output_path}")
        logger.info(f"Résumé des prédictions sauvegardé dans {summary_path}")
        
        # 9. Générer les visualisations
        logger.info("Génération des visualisations...")
        generate_all_visualizations(
            model_path=str(model_path),
            predictions_path=str(output_path),
            historical_data_path="data/processed/processed_global_energy_consumption.csv"
        )
        logger.info("Visualisations générées avec succès")
        
        # 10. Afficher un aperçu des prédictions
        print("\nAperçu des prédictions pour 2039 (dernière année):")
        last_year_predictions = future_data[future_data['Year'] == future_data['Year'].max()]
        print(last_year_predictions[['Country', 'Year', 'Predicted_Total_Energy_Consumption']].to_string())
        
    except Exception as e:
        logger.error(f"Une erreur est survenue : {str(e)}")
        raise

if __name__ == "__main__":
    main() 