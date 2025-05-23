import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def generate_all_plots(data_file: str, output_dir: str):
    """Génère toutes les visualisations dans un seul script."""
    # Créer le dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lire les données nettoyées
    df = pd.read_csv(data_file)
    
    # 1. Consommation énergétique par pays
    plt.figure(figsize=(15, 8))
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        plt.plot(country_data['Year'], 
                country_data['Total Energy Consumption (TWh)'],
                label=country, marker='o')
    plt.title('Évolution de la Consommation Énergétique par Pays')
    plt.xlabel('Année')
    plt.ylabel('Consommation (TWh)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'consommation_energetique.png')
    plt.close()
    
    # 2. Mix énergétique (Renouvelables vs Fossiles)
    plt.figure(figsize=(15, 8))
    mix_data = df.groupby('Country')[['Renewable Energy Share (%)', 
                                    'Fossil Fuel Dependency (%)']].mean()
    mix_data.plot(kind='bar', ax=plt.gca())
    plt.title('Mix Énergétique Moyen par Pays')
    plt.xlabel('Pays')
    plt.ylabel('Pourcentage (%)')
    plt.legend(['Énergies Renouvelables', 'Énergies Fossiles'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'mix_energetique.png')
    plt.close()
    
    # 3. Émissions de CO2
    plt.figure(figsize=(15, 8))
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        plt.plot(country_data['Year'], 
                country_data['Carbon Emissions (Million Tons)'],
                label=country, marker='o')
    plt.title('Évolution des Émissions de CO2 par Pays')
    plt.xlabel('Année')
    plt.ylabel('Émissions (Million Tons)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'emissions_co2.png')
    plt.close()
    
    # 4. Prix de l'énergie
    plt.figure(figsize=(15, 8))
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        plt.plot(country_data['Year'], 
                country_data['Energy Price Index (USD/kWh)'],
                label=country, marker='o')
    plt.title('Évolution du Prix de l\'Énergie par Pays')
    plt.xlabel('Année')
    plt.ylabel('Prix (USD/kWh)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'prix_energie.png')
    plt.close()
    
    # 5. Distribution de l'utilisation (Industriel vs Résidentiel)
    plt.figure(figsize=(15, 8))
    use_data = df.melt(id_vars=['Country', 'Year'],
                      value_vars=['Industrial Energy Use (%)', 'Household Energy Use (%)'],
                      var_name='Type', value_name='Pourcentage')
    sns.boxplot(data=use_data, x='Country', y='Pourcentage', hue='Type')
    plt.title('Distribution de l\'Utilisation de l\'Énergie par Secteur')
    plt.xlabel('Pays')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'utilisation_energie.png')
    plt.close()
    
    print(f"Toutes les visualisations ont été générées dans {output_dir}")

if __name__ == "__main__":
    data_file = "data/processed/cleaned_energy_data.csv"
    output_dir = "reports/figures"
    generate_all_plots(data_file, output_dir) 