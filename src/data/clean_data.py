import pandas as pd
import numpy as np
from pathlib import Path

def clean_energy_data(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Nettoie et valide les données de consommation énergétique.
    
    Args:
        input_file (str): Chemin vers le fichier CSV d'entrée
        output_file (str): Chemin vers le fichier CSV de sortie nettoyé
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    # Lecture des données
    df = pd.read_csv(input_file)
    
    # Suppression des doublons
    df = df.drop_duplicates(subset=['Country', 'Year'])
    
    # Filtrage des années futures (après 2023)
    df = df[df['Year'] <= 2023]
    
    # Correction des valeurs aberrantes
    # 1. Consommation totale d'énergie (TWh)
    df['Total Energy Consumption (TWh)'] = df['Total Energy Consumption (TWh)'].clip(
        lower=df.groupby('Country')['Total Energy Consumption (TWh)'].transform('mean') * 0.1,
        upper=df.groupby('Country')['Total Energy Consumption (TWh)'].transform('mean') * 2
    )
    
    # 2. Consommation par habitant (kWh)
    df['Per Capita Energy Use (kWh)'] = df['Per Capita Energy Use (kWh)'].clip(
        lower=1000,  # Minimum réaliste
        upper=50000  # Maximum réaliste
    )
    
    # 3. Pourcentages (doivent être entre 0 et 100)
    percentage_columns = [
        'Renewable Energy Share (%)',
        'Fossil Fuel Dependency (%)',
        'Industrial Energy Use (%)',
        'Household Energy Use (%)'
    ]
    
    for col in percentage_columns:
        df[col] = df[col].clip(0, 100)
    
    # 4. Émissions de carbone (Million Tons)
    df['Carbon Emissions (Million Tons)'] = df['Carbon Emissions (Million Tons)'].clip(
        lower=0,
        upper=df.groupby('Country')['Carbon Emissions (Million Tons)'].transform('mean') * 2
    )
    
    # 5. Prix de l'énergie (USD/kWh)
    df['Energy Price Index (USD/kWh)'] = df['Energy Price Index (USD/kWh)'].clip(
        lower=0.05,  # Minimum réaliste
        upper=0.5    # Maximum réaliste
    )
    
    # Vérification de la cohérence des pourcentages
    df['Total Energy Use (%)'] = df['Industrial Energy Use (%)'] + df['Household Energy Use (%)']
    df.loc[df['Total Energy Use (%)'] > 100, ['Industrial Energy Use (%)', 'Household Energy Use (%)']] = \
        df.loc[df['Total Energy Use (%)'] > 100, ['Industrial Energy Use (%)', 'Household Energy Use (%)']].div(
            df.loc[df['Total Energy Use (%)'] > 100, 'Total Energy Use (%)'], axis=0) * 100
    
    # Suppression de la colonne temporaire
    df = df.drop('Total Energy Use (%)', axis=1)
    
    # Sauvegarde des données nettoyées
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df

if __name__ == "__main__":
    # Chemins des fichiers
    input_file = "data/raw/global_energy_consumption.csv"
    output_file = "data/processed/cleaned_energy_data.csv"
    
    # Nettoyage des données
    cleaned_df = clean_energy_data(input_file, output_file)
    
    # Affichage des statistiques de base
    print("\nStatistiques des données nettoyées :")
    print(cleaned_df.describe())
    
    # Vérification des pays et années
    print("\nNombre de pays uniques :", cleaned_df['Country'].nunique())
    print("Période couverte :", cleaned_df['Year'].min(), "à", cleaned_df['Year'].max()) 