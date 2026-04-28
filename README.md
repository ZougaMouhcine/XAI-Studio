# XAI Studio — Phase 1 : Infrastructure & ML Core

> Plateforme Interactive d'Explicabilité des Modèles IA avec Assistance LLM

## 📋 Description

XAI Studio est une application desktop Python/Tkinter qui constitue le noyau ML d'une plateforme d'explicabilité IA. Cette Phase 1 couvre :

- **Chargement de données** CSV avec détection automatique d'encodage
- **Préprocessing automatique** (imputation, encodage, scaling, split)
- **Entraînement multi-modèles** (6 classifieurs + 6 régresseurs scikit-learn)
- **Évaluation des performances** avec tableau comparatif
- **Sauvegarde / chargement** de modèles en `.pkl`
- **UI professionnelle light-first** avec bascule instantanée vers le mode sombre

## 🏗 Architecture

```
XAI_Studio/
├── main.py                    # Point d'entrée
├── config/settings.py         # Configuration centralisée
├── core/                      # Logique ML pure
│   ├── data_loader.py         # Chargement CSV
│   ├── preprocessing.py       # Pipeline de préprocessing
│   ├── training.py            # Entraînement multi-modèles
│   ├── evaluation.py          # Métriques de performance
│   └── persistence.py         # Sérialisation .pkl
├── services/
│   └── pipeline_service.py    # Orchestrateur métier
├── ui/                        # Interface Tkinter
│   ├── app.py                 # Fenêtre principale
│   ├── theme.py               # Thème sombre professionnel
│   ├── components/            # Composants réutilisables
│   └── views/                 # 5 écrans (données, preprocessing, etc.)
├── utils/logger.py            # Logging centralisé
├── data/samples/              # Datasets d'exemple
└── models/saved/              # Modèles sérialisés
```

### Séparation des responsabilités

| Couche | Rôle | Dépendances |
|--------|------|-------------|
| `ui/` | Présentation | → `services/` |
| `services/` | Logique métier | → `core/` |
| `core/` | ML pur | → `config/`, `utils/` |

## 🚀 Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python main.py
```

## 📦 Dépendances

- Python 3.11+
- pandas, numpy, scikit-learn, joblib
- Tkinter (inclus avec Python)

## 🔧 Modèles supportés

### Classification
| Modèle | Classe scikit-learn |
|--------|---------------------|
| Logistic Regression | `LogisticRegression` |
| Random Forest | `RandomForestClassifier` |
| SVM | `SVC` |
| KNN | `KNeighborsClassifier` |
| Decision Tree | `DecisionTreeClassifier` |
| Gradient Boosting | `GradientBoostingClassifier` |

### Régression
| Modèle | Classe scikit-learn |
|--------|---------------------|
| Linear Regression | `LinearRegression` |
| Random Forest | `RandomForestRegressor` |
| SVR | `SVR` |
| KNN | `KNeighborsRegressor` |
| Decision Tree | `DecisionTreeRegressor` |
| Gradient Boosting | `GradientBoostingRegressor` |

## 👥 Intégration Équipe

Le code est conçu pour être **réutilisé par les autres membres de l'équipe** :

```python
from services.pipeline_service import PipelineService

service = PipelineService()
service.load_data("data.csv")
service.set_target_column("target")
result = service.run_preprocessing()
models = service.run_training()
metrics = service.run_evaluation()
service.save_all_trained_models()
```

## 📄 Licence

Projet académique — Usage interne.
