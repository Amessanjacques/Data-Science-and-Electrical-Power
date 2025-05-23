import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_energy_consumption_trends(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Total Energy Consumption (TWh)', hue='Country', marker='o')
    plt.title('Tendance de la consommation énergétique totale par pays')
    plt.ylabel('Consommation totale (TWh)')
    plt.xlabel('Année')
    plt.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_renewable_energy_share(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Renewable Energy Share (%)', hue='Country', marker='o')
    plt.title('Part des énergies renouvelables (%) par pays')
    plt.ylabel('Part renouvelable (%)')
    plt.xlabel('Année')
    plt.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_energy_mix_comparison(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    mix = df.groupby('Country')[['Renewable Energy Share (%)', 'Fossil Fuel Dependency (%)']].mean().reset_index()
    mix = mix.melt(id_vars='Country', var_name='Type', value_name='Pourcentage')
    sns.barplot(data=mix, x='Country', y='Pourcentage', hue='Type')
    plt.title('Mix énergétique moyen par pays')
    plt.ylabel('Pourcentage moyen (%)')
    plt.xlabel('Pays')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_carbon_emissions_trends(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Carbon Emissions (Million Tons)', hue='Country', marker='o')
    plt.title('Tendance des émissions de carbone par pays')
    plt.ylabel('Émissions de carbone (Millions de tonnes)')
    plt.xlabel('Année')
    plt.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_energy_price_trends(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Energy Price Index (USD/kWh)', hue='Country', marker='o')
    plt.title("Tendance de l'indice des prix de l'énergie par pays")
    plt.ylabel('Prix de l\'énergie (USD/kWh)')
    plt.xlabel('Année')
    plt.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_energy_use_distribution(df: pd.DataFrame, output_file: str):
    plt.figure(figsize=(12, 6))
    use = df.melt(id_vars=['Country', 'Year'], value_vars=['Industrial Energy Use (%)', 'Household Energy Use (%)'], var_name='Secteur', value_name='Pourcentage')
    sns.boxplot(data=use, x='Country', y='Pourcentage', hue='Secteur')
    plt.title("Distribution de l'utilisation de l'énergie par secteur et par pays")
    plt.ylabel('Pourcentage (%)')
    plt.xlabel('Pays')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close() 