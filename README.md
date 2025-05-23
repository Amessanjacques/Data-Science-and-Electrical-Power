# Analyse et Prédiction de la Consommation Énergétique

Ce projet analyse les tendances de consommation énergétique mondiale et développe des modèles prédictifs pour anticiper l'évolution future.

## Résultats et Visualisations

### Données Nettoyées
- Période d'analyse : 2000-2023
- Nombre de pays : 10 (Australie, Brésil, Canada, Chine, Allemagne, Inde, Japon, Russie, UK, USA)
- Nombre total d'enregistrements : 160

### Bornes Réalistes Appliquées
- Consommation Totale : 100-4000 TWh
- Consommation par Habitant : 1000-15000 kWh
- Part d'Énergies Renouvelables : 0-85%
- Dépendance aux Énergies Fossiles : 0-95%
- Utilisation Industrielle : 15-65%
- Utilisation Résidentielle : 10-45%
- Émissions de CO2 : 50-10000 Millions de Tonnes
- Prix de l'Énergie : 0.05-0.35 USD/kWh

### Modèles de Prédiction
Trois modèles ont été entraînés pour prédire la consommation énergétique totale :

1. **Régression Linéaire**
   - R² : 0.7167
   - RMSE : 507.7236
   - MAE : 338.1186

2. **Random Forest**
   - R² : 0.8815
   - RMSE : 328.3965
   - MAE : 102.9842

3. **XGBoost**
   - R² : 0.8792
   - RMSE : 331.6059
   - MAE : 103.6303

Le modèle Random Forest montre les meilleures performances globales.

### Visualisations Générées

#### 1. Évolution de la Consommation Énergétique
![Évolution de la Consommation Énergétique](reports/figures/consommation_energetique.png)
- Tendance de la consommation par pays (2000-2023)
- Visualisation des pics et baisses de consommation

#### 2. Mix Énergétique
![Mix Énergétique](reports/figures/mix_energetique.png)
- Répartition énergies renouvelables vs fossiles
- Comparaison entre pays

#### 3. Émissions de CO2
![Émissions de CO2](reports/figures/emissions_co2.png)
- Évolution des émissions par pays
- Identification des tendances de réduction

#### 4. Prix de l'Énergie
![Prix de l'Énergie](reports/figures/prix_energie.png)
- Tendance des prix par pays
- Analyse de la volatilité des prix

#### 5. Utilisation de l'Énergie
![Utilisation de l'Énergie](reports/figures/utilisation_energie.png)
- Répartition usage industriel vs résidentiel
- Comparaison entre pays

#### 6. Matrice de Corrélation
![Matrice de Corrélation](reports/figures/correlation_matrix.png)
- Relations entre variables numériques
- Identification des facteurs clés

### Visualisations des Modèles

#### Performance des Modèles

##### Régression Linéaire
![Performance Régression Linéaire](reports/figures/model_performance_linear_regression.png)

##### Random Forest
![Performance Random Forest](reports/figures/model_performance_random_forest.png)

##### XGBoost
![Performance XGBoost](reports/figures/model_performance_xgboost.png)

#### Importance des Variables
![Importance des Variables](reports/figures/feature_importance.png)

### Visualisations Supplémentaires

#### Tendances de la Consommation Énergétique
![Tendances de la Consommation](reports/figures/energy_consumption_trends.png)

#### Part des Énergies Renouvelables
![Part des Énergies Renouvelables](reports/figures/renewable_energy_share.png)

#### Comparaison du Mix Énergétique
![Comparaison du Mix Énergétique](reports/figures/energy_mix_comparison.png)

#### Tendances des Émissions de CO2
![Tendances des Émissions](reports/figures/carbon_emissions_trends.png)

#### Tendances des Prix de l'Énergie
![Tendances des Prix](reports/figures/energy_price_trends.png)

#### Distribution de l'Utilisation Énergétique
![Distribution de l'Utilisation](reports/figures/energy_use_distribution.png)

## Méthodologie

### Nettoyage des Données
1. **Prétraitement Initial**
   - Suppression des doublons
   - Application de bornes réalistes
   - Vérification de la cohérence des pourcentages

2. **Validation des Données**
   - Vérification des sommes de pourcentages ≤ 100%
   - Correction des valeurs aberrantes par pays
   - Standardisation des données

### Modélisation
1. **Préparation des Features**
   - Création de features temporelles
   - Encodage des variables catégorielles
   - Standardisation des variables numériques

2. **Entraînement des Modèles**
   - Split train/test (80/20)
   - Validation croisée
   - Optimisation des hyperparamètres

## Structure du Projet
```
projet-energetique/
├── data/
│   ├── raw/                    # Données brutes
│   └── processed/              # Données nettoyées
├── src/
│   ├── data/                   # Scripts de nettoyage
│   ├── models/                 # Scripts de modélisation
│   └── visualization/          # Scripts de visualisation
├── reports/
│   └── figures/               # Graphiques générés
├── models/                    # Modèles sauvegardés
└── README.md
```

## Environnement Technique
- Python 3.8+
- Bibliothèques principales :
  - pandas, numpy : Manipulation des données
  - scikit-learn : Modélisation
  - xgboost : Modèle XGBoost
  - matplotlib, seaborn : Visualisation
  - joblib : Sauvegarde des modèles

## Installation et Utilisation
1. Cloner le repository
2. Installer les dépendances : `pip install -r requirements.txt`
3. Exécuter le pipeline complet : `python src/main.py`

## Auteur
Jacques Amessan

## Licence
Ce projet est sous licence MIT. 