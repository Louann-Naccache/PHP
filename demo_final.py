#!/usr/bin/env python3
"""
Démonstration finale de l'IA de prédiction de mots
Version non-interactive pour démonstration rapide
"""

import re
from collections import defaultdict, Counter

class QuickWordPredictor:
    """Prédicteur de mots rapide et simple"""
    
    def __init__(self):
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
    
    def clean_text(self, text):
        """Nettoie le texte"""
        return re.sub(r'[^\w\s]', '', text.lower())
    
    def train(self, texts):
        """Entraîne le modèle"""
        print("🔄 Entraînement du modèle...")
        
        for text in texts:
            words = self.clean_text(text).split()
            self.vocab.update(words)
            
            # Créer des trigrammes (3-grammes)
            for i in range(len(words) - 2):
                context = (words[i], words[i+1])
                next_word = words[i+2]
                self.ngrams[context][next_word] += 1
        
        print(f"✅ Modèle entraîné avec {len(self.ngrams)} trigrammes")
        print(f"✅ Vocabulaire de {len(self.vocab)} mots")
    
    def predict(self, text, num_predictions=3):
        """Prédit les mots suivants"""
        words = self.clean_text(text).split()
        
        if len(words) < 2:
            return []
        
        # Prendre les deux derniers mots comme contexte
        context = (words[-2], words[-1])
        
        if context not in self.ngrams:
            return []
        
        # Retourner les prédictions les plus probables
        predictions = self.ngrams[context].most_common(num_predictions)
        return [(word, count) for word, count in predictions]

def main():
    """Démonstration principale"""
    print("🚀 IA DE PRÉDICTION DE MOTS - DÉMONSTRATION FINALE")
    print("="*60)
    
    # Données d'entraînement
    training_data = [
        "the quick brown fox jumps over the lazy dog",
        "the brown fox is very quick and agile",
        "the lazy dog sleeps in the warm sun",
        "every quick brown fox knows how to jump",
        "the dog is lazy but very friendly",
        "machine learning is a fascinating field",
        "artificial intelligence is changing the world",
        "python is a popular programming language",
        "natural language processing helps computers understand",
        "the weather today is beautiful and sunny",
        "the sun is shining very brightly today",
        "it is a perfect day for outdoor activities"
    ]
    
    # Créer et entraîner le modèle
    predictor = QuickWordPredictor()
    predictor.train(training_data)
    
    # Tests de prédiction
    test_cases = [
        "the quick",
        "machine learning",
        "the weather",
        "artificial intelligence",
        "python is",
        "the sun",
        "natural language",
        "the lazy"
    ]
    
    print("\n🎯 RÉSULTATS DES PRÉDICTIONS:")
    print("="*60)
    
    for test in test_cases:
        predictions = predictor.predict(test)
        print(f"\n📝 Texte: '{test}'")
        
        if predictions:
            words = [word for word, _ in predictions]
            print(f"🔮 Prédictions: {', '.join(words)}")
        else:
            print("❌ Aucune prédiction disponible")
    
    print("\n" + "="*60)
    print("✅ DÉMONSTRATION TERMINÉE!")
    print("="*60)
    print("🎉 L'IA de prédiction de mots fonctionne parfaitement!")
    print("💡 Ce modèle utilise des trigrammes pour prédire le mot suivant.")
    print("💡 Plus vous ajoutez de données, plus les prédictions s'améliorent.")

if __name__ == "__main__":
    main()