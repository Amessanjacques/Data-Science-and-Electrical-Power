#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour le chargement et la validation des données énergétiques.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    """Classe pour le chargement et la validation des données."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialise le chargeur de données.
        
        Args:
            data_dir (str): Chemin vers le dossier contenant les données
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Création des dossiers s'ils n'existent pas
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Charge les données depuis un fichier.
        
        Args:
            filename (str): Nom du fichier à charger
            
        Returns:
            Optional[pd.DataFrame]: DataFrame contenant les données ou None si erreur
        """
        try:
            file_path = self.raw_data_dir / filename
            if not file_path.exists():
                logger.error(f"Le fichier {filename} n'existe pas dans {self.raw_data_dir}")
                return None
                
            # Détection automatique du type de fichier
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Format de fichier non supporté : {filename}")
                return None
                
            logger.info(f"Données chargées avec succès depuis {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {str(e)}")
            return None
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Sauvegarde les données traitées.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            filename (str): Nom du fichier de sortie
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            output_path = self.processed_data_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Données sauvegardées avec succès dans {filename}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données : {str(e)}")
            return False
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valide les données chargées.
        
        Args:
            df (pd.DataFrame): DataFrame à valider
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant les résultats de la validation
        """
        validation_results = {
            "nombre_lignes": len(df),
            "colonnes_manquantes": df.columns[df.isnull().any()].tolist(),
            "types_colonnes": df.dtypes.to_dict(),
            "valeurs_uniques": {col: df[col].nunique() for col in df.columns}
        }
        
        logger.info("Validation des données terminée")
        return validation_results 