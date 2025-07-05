#!/usr/bin/env python3
"""
Script de démonstration pour l'IA de prédiction de mots
"""

from word_predictor import WordPredictor
import time

def demo_simple():
    """Démonstration simple de l'IA de prédiction de mots"""
    
    print("=" * 60)
    print("DÉMONSTRATION DE L'IA DE PRÉDICTION DE MOTS")
    print("=" * 60)
    
    # Initialiser le prédicteur
    print("\n1. Initialisation du prédicteur...")
    predictor = WordPredictor()
    
    # Entraînement rapide avec des époques réduites
    print("\n2. Entraînement des modèles...")
    print("   (Entraînement simplifié pour la démonstration)")
    start_time = time.time()
    predictor.train_models(epochs=5)  # Réduire les époques pour la démo
    end_time = time.time()
    print(f"   Entraînement terminé en {end_time - start_time:.1f} secondes")
    
    # Tests avec exemples prédéfinis
    print("\n3. Tests avec exemples prédéfinis...")
    test_examples = [
        "The quick brown",
        "I love machine",
        "Natural language processing",
        "This is a",
        "Machine learning is"
    ]
    
    for example in test_examples:
        print(f"\n   📝 Texte: '{example}'")
        predictions = predictor.predict_next_word(example, num_predictions=3)
        
        # Afficher les prédictions n-gramme
        if "ngram" in predictions:
            print("   🔤 N-gramme:", end=" ")
            ngram_words = [word for word, _ in predictions["ngram"] if word]
            print(", ".join(ngram_words[:3]) if ngram_words else "Aucune prédiction")
        
        # Afficher les prédictions LSTM
        if "lstm" in predictions:
            print("   🧠 LSTM:", end=" ")
            lstm_words = [word for word, _ in predictions["lstm"] if word]
            print(", ".join(lstm_words[:3]) if lstm_words else "Aucune prédiction")
    
    # Test interactif
    print("\n4. Test interactif")
    print("   Entrez vos propres textes pour voir les prédictions:")
    print("   (Tapez 'quit' pour quitter)")
    
    while True:
        user_input = input("\n   > ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        predictions = predictor.predict_next_word(user_input, num_predictions=3)
        
        if "ngram" in predictions:
            ngram_words = [word for word, _ in predictions["ngram"] if word]
            print(f"   🔤 N-gramme: {', '.join(ngram_words[:3]) if ngram_words else 'Aucune prédiction'}")
        
        if "lstm" in predictions:
            lstm_words = [word for word, _ in predictions["lstm"] if word]
            print(f"   🧠 LSTM: {', '.join(lstm_words[:3]) if lstm_words else 'Aucune prédiction'}")
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 60)

def demo_comparison():
    """Démonstration comparative des deux modèles"""
    
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    
    predictor = WordPredictor()
    predictor.train_models(epochs=3)  # Entraînement rapide
    
    test_texts = [
        "The weather is",
        "I want to",
        "Machine learning",
        "Natural language",
        "This is very",
        "The cat is"
    ]
    
    print(f"\n{'Texte':<20} {'N-gramme':<20} {'LSTM':<20}")
    print("-" * 60)
    
    for text in test_texts:
        predictions = predictor.predict_next_word(text, num_predictions=1)
        
        ngram_word = ""
        if "ngram" in predictions and predictions["ngram"]:
            ngram_word = predictions["ngram"][0][0]
        
        lstm_word = ""
        if "lstm" in predictions and predictions["lstm"]:
            lstm_word = predictions["lstm"][0][0]
        
        print(f"{text:<20} {ngram_word:<20} {lstm_word:<20}")

def demo_advanced():
    """Démonstration avancée avec sauvegarde/chargement"""
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION AVANCÉE")
    print("=" * 60)
    
    # Entraîner et sauvegarder
    print("\n1. Entraînement et sauvegarde...")
    predictor = WordPredictor()
    predictor.train_models(epochs=3)
    predictor.save_model("demo_model")
    
    # Charger le modèle
    print("\n2. Chargement du modèle...")
    new_predictor = WordPredictor()
    new_predictor.load_model("demo_model")
    
    # Tester le modèle chargé
    print("\n3. Test du modèle chargé...")
    test_text = "This is a beautiful"
    predictions = new_predictor.predict_next_word(test_text)
    
    print(f"   Texte: '{test_text}'")
    if "ngram" in predictions:
        ngram_words = [word for word, _ in predictions["ngram"][:3] if word]
        print(f"   N-gramme: {', '.join(ngram_words)}")
    
    if "lstm" in predictions:
        lstm_words = [word for word, _ in predictions["lstm"][:3] if word]
        print(f"   LSTM: {', '.join(lstm_words)}")

if __name__ == "__main__":
    print("Choisissez une démonstration:")
    print("1. Démonstration simple")
    print("2. Comparaison des modèles")
    print("3. Démonstration avancée")
    
    choice = input("\nEntrez votre choix (1-3): ").strip()
    
    if choice == "1":
        demo_simple()
    elif choice == "2":
        demo_comparison()
    elif choice == "3":
        demo_advanced()
    else:
        print("Choix invalide. Lancement de la démonstration simple...")
        demo_simple()