import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
import pickle
import json
from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import gutenberg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Classe pour préprocesser les données textuelles"""
    
    def __init__(self):
        self.tokenizer = None
        self.vocab_size = 0
        self.max_sequence_length = 0
        
    def clean_text(self, text: str) -> str:
        """Nettoie le texte"""
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux mais garder la ponctuation de base
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def prepare_sequences(self, texts: List[str], sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les séquences pour l'entraînement"""
        # Combiner tous les textes
        combined_text = ' '.join(texts)
        cleaned_text = self.clean_text(combined_text)
        
        # Tokenisation
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([cleaned_text])
        sequences = self.tokenizer.texts_to_sequences([cleaned_text])[0]
        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_sequence_length = sequence_length
        
        # Créer les séquences d'entraînement
        X, y = [], []
        for i in range(sequence_length, len(sequences)):
            X.append(sequences[i-sequence_length:i])
            y.append(sequences[i])
        
        X = np.array(X)
        y = to_categorical(y, num_classes=self.vocab_size)
        
        return X, y

class NGramModel:
    """Modèle n-gramme pour prédiction de mots"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        
    def train(self, texts: List[str]):
        """Entraîne le modèle n-gramme"""
        print(f"Entraînement du modèle {self.n}-gramme...")
        
        for text in texts:
            # Nettoyer le texte
            cleaned_text = re.sub(r'[^\w\s\.\,\!\?]', '', text.lower())
            words = cleaned_text.split()
            self.vocab.update(words)
            
            # Créer les n-grammes
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[context][next_word] += 1
        
        print(f"Modèle entraîné avec {len(self.ngrams)} n-grammes et {len(self.vocab)} mots uniques")
    
    def predict(self, context: List[str], num_predictions: int = 5) -> List[Tuple[str, float]]:
        """Prédit les mots suivants"""
        context_tuple = tuple(context[-(self.n-1):])  # Prendre les derniers n-1 mots
        
        if context_tuple not in self.ngrams:
            return [("", 0.0)] * num_predictions
        
        # Calculer les probabilités
        word_counts = self.ngrams[context_tuple]
        total_count = sum(word_counts.values())
        
        # Retourner les mots les plus probables
        predictions = []
        for word, count in word_counts.most_common(num_predictions):
            probability = count / total_count
            predictions.append((word, probability))
        
        return predictions

class LSTMModel:
    """Modèle LSTM pour prédiction de mots"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, lstm_units: int = 128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.tokenizer = None
        
    def build_model(self, sequence_length: int):
        """Construit le modèle LSTM"""
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=sequence_length),
            LSTM(self.lstm_units, return_sequences=True, dropout=0.2),
            LSTM(self.lstm_units, dropout=0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Entraîne le modèle LSTM"""
        if self.model is None:
            self.build_model(X.shape[1])
        
        print("Entraînement du modèle LSTM...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def predict(self, sequence: List[int], tokenizer: Tokenizer, num_predictions: int = 5) -> List[Tuple[str, float]]:
        """Prédit les mots suivants"""
        if self.model is None:
            return [("", 0.0)] * num_predictions
        
        # Préparer la séquence
        sequence_padded = pad_sequences([sequence], maxlen=self.model.input_shape[1])
        
        # Prédiction
        predictions = self.model.predict(sequence_padded, verbose=0)[0]
        
        # Obtenir les indices des mots les plus probables
        top_indices = np.argsort(predictions)[-num_predictions:][::-1]
        
        # Convertir les indices en mots
        index_to_word = {v: k for k, v in tokenizer.word_index.items()}
        
        results = []
        for idx in top_indices:
            if idx in index_to_word:
                word = index_to_word[idx]
                probability = predictions[idx]
                results.append((word, probability))
        
        return results

class WordPredictor:
    """Classe principale pour la prédiction de mots"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.ngram_model = NGramModel(n=3)
        self.lstm_model = None
        self.is_trained = False
        
    def load_sample_data(self) -> List[str]:
        """Charge des données d'exemple"""
        try:
            # Télécharger les données NLTK si nécessaire
            nltk.download('gutenberg', quiet=True)
            
            # Utiliser quelques textes du corpus Gutenberg
            sample_texts = []
            for fileid in ['austen-emma.txt', 'austen-persuasion.txt'][:2]:  # Limiter pour l'exemple
                text = gutenberg.raw(fileid)
                sample_texts.append(text[:10000])  # Prendre seulement les premiers 10000 caractères
            
            return sample_texts
            
        except Exception as e:
            print(f"Erreur lors du chargement des données NLTK: {e}")
            # Données d'exemple de base
            return [
                "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing.",
                "Machine learning is a fascinating field. It involves training models on data to make predictions.",
                "Natural language processing helps computers understand human language. It's used in many applications."
            ]
    
    def train_models(self, texts: Optional[List[str]] = None, epochs: int = 20):
        """Entraîne les modèles"""
        if texts is None:
            texts = self.load_sample_data()
        
        print("Début de l'entraînement des modèles...")
        
        # Entraîner le modèle n-gramme
        self.ngram_model.train(texts)
        
        # Préparer les données pour le modèle LSTM
        X, y = self.preprocessor.prepare_sequences(texts, sequence_length=10)
        
        # Entraîner le modèle LSTM
        self.lstm_model = LSTMModel(self.preprocessor.vocab_size)
        self.lstm_model.train(X, y, epochs=epochs, batch_size=16)
        
        self.is_trained = True
        print("Entraînement terminé!")
    
    def predict_next_word(self, text: str, model_type: str = "both", num_predictions: int = 5) -> Dict:
        """Prédit le prochain mot"""
        if not self.is_trained:
            return {"error": "Le modèle n'est pas encore entraîné. Utilisez train_models() d'abord."}
        
        words = text.lower().split()
        results = {}
        
        if model_type in ["ngram", "both"]:
            # Prédiction avec n-gramme
            ngram_predictions = self.ngram_model.predict(words, num_predictions)
            results["ngram"] = ngram_predictions
        
        if model_type in ["lstm", "both"] and self.lstm_model:
            # Prédiction avec LSTM
            try:
                # Convertir le texte en séquence
                sequence = self.preprocessor.tokenizer.texts_to_sequences([text])[0]
                if len(sequence) > 0:
                    lstm_predictions = self.lstm_model.predict(
                        sequence, self.preprocessor.tokenizer, num_predictions
                    )
                    results["lstm"] = lstm_predictions
                else:
                    results["lstm"] = [("", 0.0)] * num_predictions
            except Exception as e:
                results["lstm"] = [("", 0.0)] * num_predictions
        
        return results
    
    def interactive_demo(self):
        """Démonstration interactive"""
        if not self.is_trained:
            print("Entraînement des modèles...")
            self.train_models()
        
        print("\n" + "="*50)
        print("DÉMONSTRATION DE PRÉDICTION DE MOTS")
        print("="*50)
        print("Tapez votre texte et l'IA prédira les mots suivants.")
        print("Tapez 'quit' pour quitter.")
        print("="*50)
        
        while True:
            user_input = input("\nEntrez votre texte: ").strip()
            
            if user_input.lower() == 'quit':
                print("Au revoir!")
                break
            
            if not user_input:
                continue
            
            # Obtenir les prédictions
            predictions = self.predict_next_word(user_input)
            
            print(f"\nTexte: '{user_input}'")
            print("-" * 40)
            
            # Afficher les prédictions n-gramme
            if "ngram" in predictions:
                print("Prédictions N-gramme:")
                for i, (word, prob) in enumerate(predictions["ngram"], 1):
                    if word:
                        print(f"  {i}. {word} (probabilité: {prob:.3f})")
                print()
            
            # Afficher les prédictions LSTM
            if "lstm" in predictions:
                print("Prédictions LSTM:")
                for i, (word, prob) in enumerate(predictions["lstm"], 1):
                    if word:
                        print(f"  {i}. {word} (probabilité: {prob:.3f})")
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        if not self.is_trained:
            print("Aucun modèle entraîné à sauvegarder.")
            return
        
        # Sauvegarder le modèle n-gramme
        with open(f"{filepath}_ngram.pkl", 'wb') as f:
            pickle.dump(self.ngram_model, f)
        
        # Sauvegarder le modèle LSTM
        if self.lstm_model and self.lstm_model.model:
            self.lstm_model.model.save(f"{filepath}_lstm.h5")
        
        # Sauvegarder le tokenizer
        with open(f"{filepath}_tokenizer.pkl", 'wb') as f:
            pickle.dump(self.preprocessor.tokenizer, f)
        
        print(f"Modèles sauvegardés avec le préfixe: {filepath}")
    
    def load_model(self, filepath: str):
        """Charge le modèle"""
        try:
            # Charger le modèle n-gramme
            with open(f"{filepath}_ngram.pkl", 'rb') as f:
                self.ngram_model = pickle.load(f)
            
            # Charger le modèle LSTM
            self.lstm_model = LSTMModel(0)  # vocab_size sera mis à jour
            self.lstm_model.model = tf.keras.models.load_model(f"{filepath}_lstm.h5")
            
            # Charger le tokenizer
            with open(f"{filepath}_tokenizer.pkl", 'rb') as f:
                self.preprocessor.tokenizer = pickle.load(f)
            
            self.is_trained = True
            print(f"Modèles chargés depuis: {filepath}")
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")

def main():
    """Fonction principale"""
    print("Initialisation du prédicteur de mots...")
    predictor = WordPredictor()
    
    # Lancer la démonstration interactive
    predictor.interactive_demo()

if __name__ == "__main__":
    main()