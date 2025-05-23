import os
import pandas as pd
import numpy as np
from visualization_utils import EnergyVisualizer
from typing import Dict, List
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class VisualizationGenerator:
    def __init__(self, data_path: str, output_dir: str):
        """
        Initialise le générateur de visualisations.
        
        Args:
            data_path (str): Chemin vers le fichier de données
            output_dir (str): Répertoire de sortie pour les visualisations
        """
        self.data = pd.read_csv(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.subdirs = {
            'temporal': self.output_dir / 'temporal',
            'correlation': self.output_dir / 'correlation',
            'model_performance': self.output_dir / 'model_performance',
            'feature_importance': self.output_dir / 'feature_importance',
            'predictions': self.output_dir / 'predictions',
            'maps': self.output_dir / 'maps'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
            
        # Sous-répertoires pour différents types de visualisations
        self.subdirs = {
            'temporal': self.output_dir / 'temporal',
            'correlation': self.output_dir / 'correlation',
            'model_performance': self.output_dir / 'model_performance',
            'feature_importance': self.output_dir / 'feature_importance',
            'predictions': self.output_dir / 'predictions',
            'maps': self.output_dir / 'maps'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
            
        # Conversion de la colonne Year en datetime
        # self.data['Year'] = pd.to_datetime(self.data['Year'], format='%Y')
        
        self.visualizer = EnergyVisualizer(self.data)
        
    def generate_all_visualizations(self, 
                                  model_predictions: Dict[str, Dict] = None,
                                  feature_importance: Dict[str, np.ndarray] = None) -> Dict:
        """
        Génère toutes les visualisations.
        
        Args:
            model_predictions (Dict[str, Dict], optional): Prédictions des modèles
            feature_importance (Dict[str, np.ndarray], optional): Importance des variables
            
        Returns:
            Dict: Métriques de performance et chemins des visualisations
        """
        results = {
            'visualizations': {},
            'metrics': {}
        }
        
        # 1. Visualisations temporelles
        results['visualizations']['temporal'] = self._generate_temporal_visualizations()
        
        # 2. Matrices de corrélation
        results['visualizations']['correlation'] = self._generate_correlation_visualizations()
        
        # 3. Performance des modèles
        if model_predictions:
            results['visualizations']['model_performance'] = self._generate_model_performance_visualizations(
                model_predictions)
            results['metrics'] = self._get_model_metrics(model_predictions)
            
        # 4. Importance des variables
        if feature_importance:
            results['visualizations']['feature_importance'] = self._generate_feature_importance_visualizations(
                feature_importance)
            
        # 5. Cartes interactives
        results['visualizations']['maps'] = self._generate_map_visualizations()
        
        # Sauvegarder les résultats
        self._save_results(results)
        
        return results
        
    def _generate_temporal_visualizations(self) -> Dict[str, str]:
        """Génère les visualisations temporelles."""
        results = {}
        
        # Évolution de la consommation totale d'énergie
        fig = self.visualizer.plot_temporal_evolution(
            date_col='Year',
            value_col='Total Energy Consumption (TWh)',
            title='Évolution de la Consommation Totale d\'Énergie',
            save_path=str(self.subdirs['temporal'] / 'total_consumption.png')
        )
        results['total_consumption'] = str(self.subdirs['temporal'] / 'total_consumption.png')
        
        # Évolution de la consommation par habitant
        fig = self.visualizer.plot_temporal_evolution(
            date_col='Year',
            value_col='Per Capita Energy Use (kWh)',
            title='Évolution de la Consommation d\'Énergie par Habitant',
            save_path=str(self.subdirs['temporal'] / 'per_capita_consumption.png')
        )
        results['per_capita_consumption'] = str(self.subdirs['temporal'] / 'per_capita_consumption.png')
        
        # Évolution de la part des énergies renouvelables
        fig = self.visualizer.plot_temporal_evolution(
            date_col='Year',
            value_col='Renewable Energy Share (%)',
            title='Évolution de la Part des Énergies Renouvelables',
            save_path=str(self.subdirs['temporal'] / 'renewable_share.png')
        )
        results['renewable_share'] = str(self.subdirs['temporal'] / 'renewable_share.png')
        
        return results
        
    def _generate_correlation_visualizations(self) -> Dict[str, str]:
        """Génère les matrices de corrélation."""
        results = {}
        
        # Sélection des colonnes numériques
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Matrice de corrélation globale
        fig = self.visualizer.plot_correlation_matrix(
            columns=list(numeric_cols),
            title='Matrice de Corrélation des Variables Numériques',
            save_path=str(self.subdirs['correlation'] / 'correlation_matrix.png')
        )
        results['correlation_matrix'] = str(self.subdirs['correlation'] / 'correlation_matrix.png')
        
        return results
        
    def _generate_model_performance_visualizations(self,
                                                model_predictions: Dict[str, Dict]) -> Dict[str, str]:
        """Génère les visualisations de performance des modèles."""
        paths = {}
        
        for model_name, predictions in model_predictions.items():
            save_path = self.subdirs['model_performance'] / f'performance_{model_name}.png'
            
            fig = self.visualizer.plot_model_performance(
                y_true=predictions['y_true'],
                y_pred=predictions['y_pred'],
                model_name=model_name,
                save_path=str(save_path)
            )
            paths[f'performance_{model_name}'] = str(save_path)
            
        return paths
        
    def _generate_feature_importance_visualizations(self,
                                                 feature_importance: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Génère les visualisations d'importance des variables."""
        paths = {}
        
        for model_name, importance in feature_importance.items():
            save_path = self.subdirs['feature_importance'] / f'importance_{model_name}.png'
            
            fig = self.visualizer.plot_feature_importance(
                feature_names=self.data.columns,
                importance_values=importance,
                title=f"Importance des Variables - {model_name}",
                save_path=str(save_path)
            )
            paths[f'importance_{model_name}'] = str(save_path)
            
        return paths
        
    def _generate_map_visualizations(self) -> Dict[str, str]:
        """Génère les cartes interactives."""
        paths = {}
        
        if all(col in self.data.columns for col in ['latitude', 'longitude']):
            save_path = self.subdirs['maps'] / 'energy_performance_map.html'
            
            fig = self.visualizer.create_interactive_map(
                lat_col='latitude',
                lon_col='longitude',
                value_col='consumption',
                title="Carte Interactive de la Performance Énergétique",
                save_path=str(save_path)
            )
            paths['energy_performance_map'] = str(save_path)
            
        return paths
        
    def _get_model_metrics(self, model_predictions: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Calcule les métriques de performance pour chaque modèle."""
        metrics = {}
        
        for model_name, predictions in model_predictions.items():
            metrics[model_name] = {
                'R²': r2_score(predictions['y_true'], predictions['y_pred']),
                'RMSE': np.sqrt(mean_squared_error(predictions['y_true'], predictions['y_pred'])),
                'MAE': mean_absolute_error(predictions['y_true'], predictions['y_pred'])
            }
            
        return metrics
        
    def _save_results(self, results: Dict) -> None:
        """Sauvegarde les résultats dans un fichier JSON."""
        # Conversion des chemins en chaînes de caractères
        results_str = {}
        for category, paths in results['visualizations'].items():
            if isinstance(paths, dict):
                results_str[category] = {k: str(v) for k, v in paths.items()}
            else:
                results_str[category] = str(paths)
                
        results_str['metrics'] = results['metrics']
        
        with open(self.output_dir / 'visualization_results.json', 'w') as f:
            json.dump(results_str, f, indent=4)

if __name__ == "__main__":
    # Exemple d'utilisation
    data_path = "data/processed/processed_global_energy_consumption.csv"
    output_dir = "visualizations"
    
    generator = VisualizationGenerator(data_path, output_dir)
    
    # Exemple de prédictions de modèles (à remplacer par les vraies prédictions)
    model_predictions = {
        'Régression': {
            'y_true': np.random.rand(100),
            'y_pred': np.random.rand(100)
        },
        'Random Forest': {
            'y_true': np.random.rand(100),
            'y_pred': np.random.rand(100)
        },
        'XGBoost': {
            'y_true': np.random.rand(100),
            'y_pred': np.random.rand(100)
        }
    }
    
    # Exemple d'importance des variables (à remplacer par les vraies valeurs)
    feature_importance = {
        'Random Forest': np.random.rand(10),
        'XGBoost': np.random.rand(10)
    }
    
    # Générer toutes les visualisations
    results = generator.generate_all_visualizations(
        model_predictions=model_predictions,
        feature_importance=feature_importance
    ) 