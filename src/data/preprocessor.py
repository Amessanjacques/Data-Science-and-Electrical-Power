#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour le prétraitement des données énergétiques.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Classe pour le prétraitement des données énergétiques."""
    
    def __init__(self):
        """Initialise le prétraiteur de données."""
        self.scalers = {}
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        
    def identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identifie les types de colonnes (catégorielles et numériques).
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
        """
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Colonnes catégorielles identifiées : {self.categorical_columns}")
        logger.info(f"Colonnes numériques identifiées : {self.numerical_columns}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans le DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame à traiter
            strategy (str): Stratégie de remplacement ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: DataFrame avec les valeurs manquantes traitées
        """
        df_clean = df.copy()
        
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.numerical_columns:
                    if strategy == 'mean':
                        df_clean[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df_clean[col] = df[col].fillna(df[col].median())
                elif col in self.categorical_columns:
                    if strategy == 'mode':
                        df_clean[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
        
        logger.info(f"Traitement des valeurs manquantes terminé avec la stratégie : {strategy}")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode les variables catégorielles.
        
        Args:
            df (pd.DataFrame): DataFrame à encoder
            
        Returns:
            pd.DataFrame: DataFrame avec les variables catégorielles encodées
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        logger.info("Encodage des variables catégorielles terminé")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les variables numériques.
        
        Args:
            df (pd.DataFrame): DataFrame à normaliser
            
        Returns:
            pd.DataFrame: DataFrame avec les variables numériques normalisées
        """
        df_scaled = df.copy()
        
        for col in self.numerical_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df_scaled[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df_scaled[col] = self.scalers[col].transform(df[[col]])
        
        logger.info("Normalisation des variables numériques terminée")
        return df_scaled
    
    def preprocess_data(self, df: pd.DataFrame, 
                       handle_missing_strategy: str = 'mean') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applique l'ensemble du prétraitement aux données.
        
        Args:
            df (pd.DataFrame): DataFrame à prétraiter
            handle_missing_strategy (str): Stratégie de gestion des valeurs manquantes
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple contenant (X, y) pour l'apprentissage
        """
        # Identification des types de colonnes
        self.identify_column_types(df)
        
        # Traitement des valeurs manquantes
        df_clean = self.handle_missing_values(df, strategy=handle_missing_strategy)
        
        # Encodage des variables catégorielles
        df_encoded = self.encode_categorical_features(df_clean)
        
        # Normalisation des variables numériques
        df_scaled = self.scale_numerical_features(df_encoded)
        
        # TODO: Séparer les features (X) et la target (y)
        # Pour l'instant, on retourne le même DataFrame pour X et y
        # À adapter selon la structure de vos données
        
        logger.info("Prétraitement complet terminé")
        return df_scaled, df_scaled 