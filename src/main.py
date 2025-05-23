#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour l'analyse prédictive de l'efficacité énergétique des bâtiments.
"""

import logging
from pathlib import Path
import json
from datetime import datetime
import sys

# Ajout du répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.models.model_trainer import ModelTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'energy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def save_analysis_results(results: dict, filename: str) -> None:
    """
    Sauvegarde les résultats de l'analyse dans un fichier JSON.
    
    Args:
        results (dict): Résultats à sauvegarder
        filename (str): Nom du fichier de sortie
    """
    output_path = Path("results") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Résultats sauvegardés dans {filename}")

def main():
    """
    Fonction principale orchestrant l'analyse des données énergétiques.
    """
    try:
        logger.info("Démarrage de l'analyse énergétique")
        
        # Initialisation des composants
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        # Liste des fichiers à analyser dans le dossier data/raw
        raw_files = list(data_loader.raw_data_dir.glob("*.*"))
        if not raw_files:
            logger.error("Aucun fichier de données trouvé dans le dossier data/raw")
            return
            
        logger.info(f"Fichiers trouvés : {[f.name for f in raw_files]}")
        
        # Analyse de chaque fichier
        for file_path in raw_files:
            logger.info(f"Analyse du fichier : {file_path.name}")
            
            # 1. Chargement des données
            df = data_loader.load_data(file_path.name)
            if df is None:
                continue
                
            # 2. Validation des données
            # Conversion des types pour la sérialisation JSON
            validation_results = data_loader.validate_data(df)
            validation_results["types_colonnes"] = {col: str(dtype) for col, dtype in validation_results["types_colonnes"].items()}
            logger.info(f"Résultats de la validation : {json.dumps(validation_results, indent=2)}")
            
            # 3. Prétraitement des données
            X, y = preprocessor.preprocess_data(df)
            
            # 4. Sauvegarde des données prétraitées
            processed_filename = f"processed_{file_path.stem}.csv"
            if data_loader.save_processed_data(X, processed_filename):
                logger.info(f"Données prétraitées sauvegardées dans {processed_filename}")
            
            # 5. Sauvegarde des résultats de l'analyse
            analysis_results = {
                "fichier_source": file_path.name,
                "validation": validation_results,
                "statistiques": {
                    "nombre_lignes": len(df),
                    "nombre_colonnes": len(df.columns),
                    "colonnes": list(df.columns),
                    "types_colonnes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "valeurs_manquantes": df.isnull().sum().to_dict(),
                    "statistiques_descriptives": df.describe().to_dict()
                }
            }
            
            save_analysis_results(
                analysis_results,
                f"analyse_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # 6. Entraînement et évaluation du modèle prédictif
            # On suppose que la cible est 'Total Energy Consumption (TWh)'
            target_col = 'Total Energy Consumption (TWh)'
            if target_col in df.columns:
                # On retire la colonne cible de X
                X_model = X.drop(columns=[target_col], errors='ignore')
                y_model = df[target_col]
                model_trainer = ModelTrainer()
                results, best_model_name = model_trainer.train_and_evaluate(X_model, y_model)
                model_trainer.save_best_model(features=list(X_model.columns))
                # Sauvegarde des résultats d'évaluation
                save_analysis_results(
                    {"resultats_modeles": results, "meilleur_modele": best_model_name},
                    f"resultats_modeles_{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            else:
                logger.warning(f"Colonne cible '{target_col}' non trouvée dans les données.")
        
        logger.info("Analyse terminée avec succès")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue : {str(e)}")
        raise

if __name__ == "__main__":
    main() 