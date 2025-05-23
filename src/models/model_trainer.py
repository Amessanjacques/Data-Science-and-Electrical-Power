#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour l'entraînement et l'évaluation de modèles de machine learning.
"""
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        for name, model in self.models.items():
            logger.info(f"Entraînement du modèle : {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            self.results[name] = {
                "rmse": rmse,
                "r2": r2
            }
            logger.info(f"{name} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
        # Sélection du meilleur modèle (par RMSE)
        self.best_model_name = min(self.results, key=lambda k: self.results[k]["rmse"])
        self.best_model = self.models[self.best_model_name]
        logger.info(f"Meilleur modèle : {self.best_model_name}")
        return self.results, self.best_model_name

    def save_best_model(self, filename="best_model.joblib", features=None):
        if self.best_model is not None:
            path = self.output_dir / filename
            joblib.dump(self.best_model, path)
            logger.info(f"Modèle sauvegardé sous {path}")
            # Sauvegarde des features si fournies
            if features is not None:
                features_path = self.output_dir / "features.joblib"
                joblib.dump(features, features_path)
                logger.info(f"Liste des features sauvegardée sous {features_path}")
        else:
            logger.warning("Aucun modèle à sauvegarder.") 