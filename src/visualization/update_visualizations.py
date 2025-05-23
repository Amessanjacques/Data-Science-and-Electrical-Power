import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Ajout du répertoire src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from visualization.model_visualizations import ModelVisualizer
from visualization.data_visualizations import (
    plot_energy_consumption_trends,
    plot_renewable_energy_share,
    plot_energy_mix_comparison,
    plot_carbon_emissions_trends,
    plot_energy_price_trends,
    plot_energy_use_distribution
)

def update_all_visualizations(data_file: str, output_dir: str):
    """
    Met à jour toutes les visualisations avec les données nettoyées.
    
    Args:
        data_file (str): Chemin vers le fichier de données nettoyées
        output_dir (str): Répertoire de sortie pour les visualisations
    """
    # Création du répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lecture des données nettoyées
    df = pd.read_csv(data_file)
    
    # Configuration du style des graphiques
    sns.set_palette("husl")
    
    # 1. Visualisations des données
    print("Génération des visualisations de données...")
    
    # Tendance de la consommation énergétique
    plot_energy_consumption_trends(
        df,
        output_file=str(output_path / "energy_consumption_trends.png")
    )
    
    # Part des énergies renouvelables
    plot_renewable_energy_share(
        df,
        output_file=str(output_path / "renewable_energy_share.png")
    )
    
    # Comparaison du mix énergétique
    plot_energy_mix_comparison(
        df,
        output_file=str(output_path / "energy_mix_comparison.png")
    )
    
    # Tendance des émissions de carbone
    plot_carbon_emissions_trends(
        df,
        output_file=str(output_path / "carbon_emissions_trends.png")
    )
    
    # Tendance des prix de l'énergie
    plot_energy_price_trends(
        df,
        output_file=str(output_path / "energy_price_trends.png")
    )
    
    # Distribution de l'utilisation de l'énergie
    plot_energy_use_distribution(
        df,
        output_file=str(output_path / "energy_use_distribution.png")
    )
    
    # 2. Visualisations du modèle
    print("Génération des visualisations du modèle...")
    
    # Import du modèle et des prédictions
    from ..models.train_model import load_model, make_predictions
    model = load_model()
    predictions = make_predictions(model, df)
    
    # Initialisation du visualiseur de modèle
    model_visualizer = ModelVisualizer(output_dir=str(output_path / "model"))
    
    # Prédictions vs valeurs réelles
    model_visualizer.plot_predictions_vs_actual(
        y_true=df['Total Energy Consumption (TWh)'].values,
        y_pred=predictions['Predicted_Total_Energy_Consumption'].values
    )
    
    # Importance des caractéristiques
    feature_names = [col for col in df.columns if col not in ['Country', 'Year', 'Total Energy Consumption (TWh)']]
    model_visualizer.plot_feature_importance(
        model=model,
        feature_names=feature_names
    )
    
    # Résidus
    model_visualizer.plot_residuals(
        y_true=df['Total Energy Consumption (TWh)'].values,
        y_pred=predictions['Predicted_Total_Energy_Consumption'].values
    )
    
    # Tendances par pays
    model_visualizer.plot_predictions_by_country(predictions)
    
    print(f"Toutes les visualisations ont été mises à jour dans {output_dir}")

if __name__ == "__main__":
    # Chemins des fichiers
    data_file = "data/processed/cleaned_energy_data.csv"
    output_dir = "reports/figures"
    
    # Mise à jour des visualisations
    update_all_visualizations(data_file, output_dir) 