# Filtre Anti-Spam — Classifieur Naïf de Bayes

Ce projet présente l'implémentation d'un filtre anti-spam fondé sur un **classifieur naïf de Bayes** (modèle de Bernoulli multivarié / *bag-of-words* binaire). I

## 📌 Présentation du Projet

L'objectif est d'effectuer une classification binaire supervisée afin de catégoriser automatiquement les e-mails en deux classes :

* **`SPAM`** : Courrier indésirable.

* **`HAM`** : E-mail légitime.

### Points clés de l'implémentation

* **Représentation vectorielle** : Extraction des mots par expression régulière (`[A-Z]+`) et filtrage des mots de moins de 3 lettres pour construire un dictionnaire binaire de présence/absence.

* **Apprentissage & Lissage de Laplace** : Estimation des paramètres $b_j^y$ avec lissage ($\epsilon = 1$) pour éviter le problème des probabilités nulles.

* **Stabilité numérique** : Calcul des log-vraisemblances pour éviter l'*underflow*, et calcul des probabilités a posteriori via la formule **log-sum-exp** pour prévenir l'*overflow*.

* **Sauvegarde & Apprentissage en ligne** : Export du modèle avec `pickle` et mise à jour incrémentale exacte sans réentraînement complet depuis zéro.

## 📊 Résultats Expérimentaux

Le tableau ci-dessous résume les performances du filtre (entraîné sur `baseapp/` avec un lissage de Laplace $\epsilon = 1$) :

| Ensemble d'évaluation | Classe | Nombre de mails | Taux d'erreur | 
| ----- | ----- | ----- | ----- | 
| **Base de test (`basetest/`)** | SPAM | 500 | 5,6 % | 
| **Base de test (`basetest/`)** | HAM | 500 | **0,0 %** | 
| **Base de test (`basetest/`)** | **Global** | **1 000** | **2,8 %** | 
| **Base d'apprentissage (`baseapp/`)** | SPAM | 500 | 21,2 % | 
| **Base d'apprentissage (`baseapp/`)** | HAM | 2 500 | 2,6 % | 
| **Base d'apprentissage (`baseapp/`)** | **Global** | **3 000** | **5,7 %** | 

> **Remarque :** Le taux d'erreur très faible sur les e-mails légitimes (`HAM` à 0,0% en test) reflète la prise en compte du déséquilibre initial dans la base d'apprentissage ($P(\text{HAM}) \approx 0,833$). Cela réduit fortement le risque de classer par erreur un e-mail important en spam.

## 📁 Structure du Dépôt

```
.
├── baseapp/          # Base d'apprentissage (500 spams, 2500 hams)
├── basetest/         # Base de test (500 spams, 500 hams)
├── main.py           # Script principal d'apprentissage et de test
├── .gitignore        # Fichiers ignorés par Git
└── README.md         # Documentation du projet

```

## ⚙️ Prérequis et Installation

Le projet fonctionne avec **Python 3** et requiert la bibliothèque **NumPy**.

```
# Installation des dépendances
pip install numpy

```

## 🚀 Utilisation

Pour lancer l'apprentissage et l'évaluation du classifieur :

```
python main.py

```

## 