#!/usr/bin/env python3
"""
Version simplifiée de l'IA de prédiction de mots
Utilise seulement les packages de base Python
"""

import re
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

class SimpleWordPredictor:
    """Version simplifiée du prédicteur de mots"""
    
    def __init__(self, n: int = 3):
        self.n = n  # Taille du n-gramme
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.is_trained = False
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte d'entrée"""
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux mais garder la ponctuation de base
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train(self, texts: List[str]):
        """Entraîne le modèle sur les textes donnés"""
        print(f"🔄 Entraînement du modèle {self.n}-gramme...")
        
        word_count = 0
        for text in texts:
            # Nettoyer le texte
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            word_count += len(words)
            
            # Ajouter au vocabulaire
            self.vocab.update(words)
            
            # Créer les n-grammes
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[context][next_word] += 1
        
        self.is_trained = True
        print(f"✅ Modèle entraîné avec:")
        print(f"   - {len(self.ngrams)} n-grammes")
        print(f"   - {len(self.vocab)} mots uniques")
        print(f"   - {word_count} mots totaux")
    
    def predict(self, text: str, num_predictions: int = 5) -> List[Tuple[str, float]]:
        """Prédit les mots suivants"""
        if not self.is_trained:
            print("❌ Le modèle n'est pas encore entraîné!")
            return []
        
        # Nettoyer et diviser le texte
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        # Prendre les derniers n-1 mots comme contexte
        context = tuple(words[-(self.n-1):])
        
        if context not in self.ngrams:
            # Si le contexte n'existe pas, essayer avec des contextes plus courts
            for i in range(len(context)-1, 0, -1):
                shorter_context = context[-i:]
                if shorter_context in self.ngrams:
                    context = shorter_context
                    break
            else:
                # Si aucun contexte n'est trouvé, retourner des mots aléatoires du vocabulaire
                random_words = random.sample(list(self.vocab), min(num_predictions, len(self.vocab)))
                return [(word, 1.0/len(random_words)) for word in random_words]
        
        # Calculer les probabilités
        word_counts = self.ngrams[context]
        total_count = sum(word_counts.values())
        
        # Retourner les mots les plus probables
        predictions = []
        for word, count in word_counts.most_common(num_predictions):
            probability = count / total_count
            predictions.append((word, probability))
        
        return predictions
    
    def interactive_demo(self):
        """Démonstration interactive"""
        print("\n" + "="*50)
        print("🤖 DÉMONSTRATION INTERACTIVE")
        print("="*50)
        print("Tapez votre texte et l'IA prédira les mots suivants.")
        print("Tapez 'quit' pour quitter.")
        print("="*50)
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Au revoir!")
                break
            
            if not user_input:
                continue
            
            # Obtenir les prédictions
            predictions = self.predict(user_input)
            
            print(f"\n📝 Texte: '{user_input}'")
            print("-" * 30)
            
            if predictions:
                print("🔮 Prédictions:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word} (probabilité: {prob:.3f})")
            else:
                print("❌ Aucune prédiction disponible")

def load_sample_data() -> List[str]:
    """Charge des données d'exemple"""
    return [
        """The quick brown fox jumps over the lazy dog. This is a classic sentence used for testing.
        The brown fox is very agile and quick. The lazy dog prefers to sleep in the sun.
        Every quick brown fox knows how to jump. The dog is lazy but friendly.""",
        
        """Machine learning is a fascinating field of computer science. It involves training algorithms
        to make predictions based on data. Natural language processing is a subset of machine learning
        that focuses on understanding human language. These technologies are revolutionizing how we
        interact with computers.""",
        
        """Python is a popular programming language for data science and machine learning.
        It has many libraries like NumPy, Pandas, and TensorFlow that make it easy to work with data.
        The Python community is very active and helpful. Learning Python is a great way to start
        your journey in programming and data science.""",
        
        """Artificial intelligence is changing the world. AI systems can now perform tasks that
        were once thought to be impossible for machines. From image recognition to natural language
        understanding, AI is becoming more sophisticated every day. The future of AI is bright
        and full of possibilities.""",
        
        """The weather today is beautiful. The sun is shining and the sky is clear.
        It's a perfect day for outdoor activities. The temperature is just right for a walk
        in the park. Nature is beautiful when the weather is nice."""
    ]

def main():
    """Fonction principale"""
    print("🚀 IA DE PRÉDICTION DE MOTS - VERSION SIMPLIFIÉE")
    print("="*60)
    
    # Créer le prédicteur
    predictor = SimpleWordPredictor(n=3)
    
    # Charger les données d'exemple
    sample_data = load_sample_data()
    
    # Entraîner le modèle
    predictor.train(sample_data)
    
    # Tests automatiques
    print("\n🎯 TESTS AUTOMATIQUES")
    print("-" * 30)
    
    test_examples = [
        "The quick brown",
        "Machine learning is",
        "Python is a",
        "The weather is",
        "Artificial intelligence",
        "It's a perfect"
    ]
    
    for example in test_examples:
        predictions = predictor.predict(example, num_predictions=3)
        print(f"\n📝 '{example}' →", end=" ")
        if predictions:
            words = [word for word, _ in predictions]
            print(f"{', '.join(words)}")
        else:
            print("Aucune prédiction")
    
    # Démonstration interactive
    print("\n" + "="*60)
    print("🎮 VOULEZ-VOUS TESTER VOS PROPRES TEXTES ?")
    print("="*60)
    
    response = input("Appuyez sur Entrée pour continuer ou tapez 'skip' pour passer: ").strip()
    
    if response.lower() != 'skip':
        predictor.interactive_demo()
    
    print("\n✅ Démonstration terminée!")
    print("="*60)
    print("💡 Cette version simplifiée utilise seulement les packages de base Python.")
    print("💡 Pour des prédictions plus avancées, installez TensorFlow et utilisez word_predictor.py")

if __name__ == "__main__":
    main()