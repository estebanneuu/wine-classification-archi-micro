import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import json
import numpy as np
import csv
import os
from pathlib import Path

# Fichier de paramètres avec toutes les informations et paramètres concernant le modèle
# Utilisé dans la fonction model_generator pour utiliser les paramètres et updater le champ accuracy_score_test
with open('env_params.json', 'r') as f:
    params = json.load(f)


def data_preparation(file_name: str) -> tuple:
    """Fonction qui permet de préparer les données pour l'entrainement du modèle

    Args:
        file_name (str): nom du fichier csv à utiliser pour l'entrainement du modèle

    Returns:
        np.ndarray: les données de test et d'entrainement normalisées 
    """
    df_wines = pd.read_csv(file_name)
    df_wines.drop("Id", axis=1, inplace=True)

    # 1 : Mauvaise qualité | 2 : Bonne qualité
    var_quality = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
                   6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
    df_wines["quality"] = df_wines["quality"].map(var_quality)
    df_wines.quality.value_counts()

    x = df_wines.drop('quality', axis=1).values
    y = df_wines['quality'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.20,
                                                        shuffle=True,
                                                        random_state=14)

    # Normalisation des données
    norm = MinMaxScaler(feature_range=(0, 1))
    norm.fit(x_train)
    x_train = norm.transform(x_train)
    x_test = norm.transform(x_test)
    return x_train, x_test, y_train, y_test


def model_generator(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Fonction qui génère un model KNN et l'enregistre dans un fichier knn_model.sav

    Args:
        x_train (np.ndarray): les données d'entraînement
        x_test (np.ndarray): les données de test
        y_train (np.ndarray): les étiquettes d'entraînement
        y_test (np.ndarray): les étiquettes de test
    """
    # Création du modèle
    knn = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                               leaf_size=params["leaf_size"],
                               weights=params["weights"],
                               p=params["p"],
                               metric=params["metric"])
    # Entrainement du modèle
    knn.fit(x_train, y_train)
    # Enregistrement du modèle
    filename = 'knn_model.sav'
    pickle.dump(knn, open(filename, 'wb'))

    # Prédiction sur les données de test et calcul de la précision
    y_pred = knn.predict(x_test)
    params["accuracy_score_test"] = round(accuracy_score(y_test, y_pred), 2)
    # Mise à jour de la précision dans le fichier de paramètres
    with open('../env_params.json', 'w') as f:
        json.dump(params, f)


def load_model() -> KNeighborsClassifier:
    """Fonction qui permet de charger le modèle sérialisé

    Returns:
        KNeighborsClassifier : retourne le modèle chargé
    """
    loaded_model = pickle.load(open('knn_model.sav', 'rb'))
    return loaded_model


def predict_quality(model: KNeighborsClassifier, data: list) -> None:
    """Fonction qui permet de prédire la qualité d'un vin à partir d'un modèle et d'un jeu de données

    Args:
        model (KNeighborsClassifier): modèle KNN sérialisé
        data (list): jeu de données à prédire, transformé en array numpy pour être utilisé par le modèle dans le bon format
    """
    # formatage des données
    data = np.array(data).reshape(1, -1)
    # Prédiction
    pred = model.predict(data)[0]
    if pred == 1:
        return "Mauvaise qualité"
    else:
        return "Bonne qualité"


def model_description() -> None:
    """Fonction qui permet d'afficher les paramètres du modèle ainsi que la précision du modèle sur les données de test
    """
    desc = f"Le modèle KNN a une précision de {params['accuracy_score_test']} sur les données de test \nLe modèle KNN " \
           f"a été entrainé avec les paramètres suivants :\nn_neighbors : {params['n_neighbors']} \nleaf_" \
           f"size : {params['leaf_size']} \nweights : {params['weights']} \np : {params['p']} \nmetric : " \
           f"{params['metric']} "
    return desc


def retrain_model(file_name: str) -> None:
    """Fonction qui permet de recharger les données, de recharger le modèle et de le réentrainer avec les nouvelles données

    Args:
        file_name (str): nom du fichier csv à utiliser pour l'entrainement du modèle
    """
    x_train, x_test, y_train, y_test = data_preparation(file_name)
    model_generator(x_train, x_test, y_train, y_test)
    print("Le modèle a été re-entrainé avec les nouvelles données \n")


def data_enrichment(data_list: list, data_file_name: str) -> None:
    """Fonction qui permet d'enrichir le jeu de données avec les nouvelles données (1 ligne)

    Args:
        data_list (list): nouvelles données à ajouter au jeu de données
        data_file_name (str): nom du fichier csv auquel ajouter les nouvelles données
    """
    len_list = len(data_list)
    if len_list == 12:
        # Ajout des données dans le fichier csv
        with open(data_file_name, 'a') as f:
            writer = csv.writer(f)
            # Récupération de l'id de la dernière ligne
            last_row = list(csv.reader(open(data_file_name)))[-1]
            id_add = int(last_row[-1]) + 1
            # concatenate data_list and id_add
            data_list = data_list + [id_add]
            writer.writerow(data_list)
            f.close()
    else:
        print("Le nombre de données à ajouter est incorrect\nAttendu : 12\nReçu : ", len_list)


# Beaucoup de paramètres dans cette fonctions sont subjectives. Tout d'abord le seuil pour la corrélation est de 0.4
# mais cela est discutable, dans le cas ou la corr est > 0.4 on prend la valeur max de la colonne et si la corr est <
# -0.4 on prend la valeur min de la colonne. Autrement on prend la valeur moyenne de la colonne, considerant que la
# valeur n'a pas un fort impact sur la qualité du vin au vu de la corrélation avec la qualité.
def determine_perfect_wine():
    df = pd.read_csv("Wines.csv")
    # Calculer la correlation de chaque colonne avec la colonne "Quality"
    correlations = df.corr()["quality"].sort_values(ascending=False)

    # Créer un dictionnaire qui contiendra les valeurs optimales pour chaque colonne
    perfect_wine = {}

    # Pour chaque colonne du jeu de données, trouver la valeur optimale qui a une forte correlation avec "Quality"
    for col in df:
        if col == "quality" or col == "Id":
            # On ne veut pas inclure la colonne "Quality" elle-même dans les calculs
            continue

        if correlations[col] > 0.4:
            # Si la correlation est positive, utiliser la valeur maximale de la colonne
            value = df[col].max()
        elif correlations[col] < -0.4:
            # Si la correlation est négative, utiliser la valeur minimale de la colonne
            value = df[col].min()
        else:
            # Si la correlation est faible, utiliser la valeur moyenne de la colonne
            value = df[col].mean()

        perfect_wine[col] = value

    # Afficher les valeurs de chaque colonne pour le "vin parfait"
    return perfect_wine


############################################################################################################
my_file = Path("knn_model.sav")
if my_file.is_file():
    next
else:
    retrain_model(params["data_file_name"])

# (POST /api/predict)
# EXEMPLE
var_to_predict = [7.4, 0.7, 0.0, 1.9, 0.076,
                  11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
# Simulation de prédiction, le format doit être une liste de 11 valeurs (sans quality ni id)
predict_quality(load_model(), var_to_predict)

# (GET /api/model/description)
# Description du modèle
# model_description()

# (PUT /api/model)
# EXEMPLE
var_to_add = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# Ajout de données dans le fichier csv (12 valeurs avec quality) l'id est ajouté automatiquement dans la fonction data_enrichment
data_enrichment(var_to_add, params["data_file_name"])

# (GET /api/model)
# Chargement du modèle
# model = load_model()
