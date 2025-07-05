# IA de Prédiction de Mots 🤖

Une intelligence artificielle complète pour prédire le prochain mot dans un texte, utilisant deux approches différentes :

- **Modèle N-gramme** : Approche statistique basée sur la fréquence des séquences de mots
- **Modèle LSTM** : Réseau de neurones récurrent pour des prédictions plus sophistiquées

## 🚀 Installation

1. Clonez ce projet
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 📋 Utilisation

### Utilisation simple

```python
from word_predictor import WordPredictor

# Initialiser le prédicteur
predictor = WordPredictor()

# Entraîner les modèles
predictor.train_models()

# Prédire le prochain mot
predictions = predictor.predict_next_word("The quick brown")
print(predictions)
```

### Démonstration interactive

Lancez le script de démonstration :

```bash
python demo_example.py
```

Ou utilisez la version interactive du module principal :

```bash
python word_predictor.py
```

## 🔧 Fonctionnalités principales

### 1. Modèle N-gramme
- Basé sur les statistiques de fréquence des mots
- Rapide et efficace
- Idéal pour des prédictions simples

### 2. Modèle LSTM
- Réseau de neurones récurrent
- Capture les dépendances à long terme
- Prédictions plus nuancées

### 3. Préprocesseur de données
- Nettoyage automatique du texte
- Tokenisation intelligente
- Gestion des séquences

## 🎯 Exemples d'utilisation

### Prédiction simple

```python
from word_predictor import WordPredictor

predictor = WordPredictor()
predictor.train_models(epochs=10)

# Test avec différents textes
textes = [
    "The weather is",
    "I love machine", 
    "Natural language processing"
]

for texte in textes:
    predictions = predictor.predict_next_word(texte)
    print(f"Texte: '{texte}'")
    print(f"N-gramme: {predictions['ngram']}")
    print(f"LSTM: {predictions['lstm']}")
    print("-" * 40)
```

### Entraînement avec données personnalisées

```python
# Vos propres textes
mes_textes = [
    "Votre premier texte d'entraînement...",
    "Votre deuxième texte d'entraînement...",
    "Et ainsi de suite..."
]

predictor = WordPredictor()
predictor.train_models(texts=mes_textes, epochs=20)

# Sauvegarder le modèle
predictor.save_model("mon_modele")

# Charger le modèle plus tard
nouveau_predictor = WordPredictor()
nouveau_predictor.load_model("mon_modele")
```

## 🎮 Scripts de démonstration

### 1. Démonstration simple
```bash
python demo_example.py
# Choisir option 1
```

### 2. Comparaison des modèles
```bash
python demo_example.py
# Choisir option 2
```

### 3. Démonstration avancée
```bash
python demo_example.py
# Choisir option 3
```

## 🛠️ Configuration avancée

### Paramètres du modèle N-gramme

```python
from word_predictor import NGramModel

# Modèle 4-gramme au lieu de 3-gramme
model = NGramModel(n=4)
```

### Paramètres du modèle LSTM

```python
from word_predictor import LSTMModel

# Modèle LSTM personnalisé
model = LSTMModel(
    vocab_size=10000,
    embedding_dim=200,
    lstm_units=256
)
```

### Préprocesseur personnalisé

```python
from word_predictor import DataPreprocessor

preprocessor = DataPreprocessor()
X, y = preprocessor.prepare_sequences(
    texts=mes_textes,
    sequence_length=15  # Séquences plus longues
)
```

## 📊 Évaluation des performances

```python
# Comparer les modèles
def evaluer_modeles(predictor, textes_test):
    for texte in textes_test:
        predictions = predictor.predict_next_word(texte)
        print(f"Texte: {texte}")
        print(f"N-gramme: {predictions['ngram'][0][0] if predictions['ngram'] else 'N/A'}")
        print(f"LSTM: {predictions['lstm'][0][0] if predictions['lstm'] else 'N/A'}")
        print()

textes_test = ["The cat is", "I want to", "Machine learning"]
evaluer_modeles(predictor, textes_test)
```

## 🔍 Structure du projet

```
.
├── word_predictor.py      # Module principal
├── demo_example.py        # Script de démonstration
├── requirements.txt       # Dépendances
└── README.md             # Documentation
```

## 🧠 Comment ça marche

### Modèle N-gramme
1. **Entraînement** : Analyse les séquences de mots dans le texte
2. **Stockage** : Compte la fréquence de chaque séquence
3. **Prédiction** : Utilise les probabilités pour prédire le mot suivant

### Modèle LSTM
1. **Préprocessing** : Convertit le texte en séquences numériques
2. **Entraînement** : Apprend les patterns avec un réseau de neurones
3. **Prédiction** : Utilise l'état caché pour prédire le mot suivant

## 🎯 Cas d'usage

- **Autocomplétion** : Terminer automatiquement les phrases
- **Correction de texte** : Suggérer des mots probables
- **Génération de texte** : Créer du contenu automatiquement
- **Analyse linguistique** : Étudier les patterns de langage

## 🔮 Améliorations futures

- [ ] Support pour plusieurs langues
- [ ] Intégration de modèles pré-entraînés (GPT, BERT)
- [ ] Interface web interactive
- [ ] API REST pour intégration
- [ ] Mécanisme d'attention pour le modèle LSTM
- [ ] Support pour les émojis et caractères spéciaux

## 🐛 Dépannage

### Erreurs communes

1. **Erreur d'importation TensorFlow** :
   ```bash
   pip install tensorflow==2.13.0
   ```

2. **Erreur NLTK** :
   ```python
   import nltk
   nltk.download('gutenberg')
   ```

3. **Mémoire insuffisante** :
   - Réduire le nombre d'époques
   - Utiliser des textes plus courts
   - Réduire la taille des batchs

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles fonctionnalités

## 📜 Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser, le modifier et le distribuer.

---

## 🔗 Liens utiles

- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation NLTK](https://www.nltk.org/)
- [Guide des N-grammes](https://en.wikipedia.org/wiki/N-gram)
- [Introduction aux LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

**Créé avec ❤️ pour l'apprentissage automatique et le traitement du langage naturel**