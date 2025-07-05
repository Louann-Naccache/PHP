#!/usr/bin/env python3
"""
Test rapide de l'IA de prédiction de mots
Version simplifiée pour démonstration immédiate
"""

import warnings
warnings.filterwarnings('ignore')

def test_rapide():
    """Test rapide de l'IA de prédiction de mots"""
    
    print("🚀 TEST RAPIDE DE L'IA DE PRÉDICTION DE MOTS")
    print("=" * 50)
    
    try:
        # Import et initialisation
        from word_predictor import WordPredictor
        print("✅ Modules importés avec succès")
        
        # Créer le prédicteur
        predictor = WordPredictor()
        print("✅ Prédicteur initialisé")
        
        # Entraînement rapide (3 époques pour la démo)
        print("\n🔄 Entraînement en cours...")
        print("   (Ceci peut prendre 1-2 minutes)")
        predictor.train_models(epochs=3)
        print("✅ Entraînement terminé!")
        
        # Tests rapides
        print("\n🎯 Tests de prédiction:")
        print("-" * 30)
        
        exemples = [
            "The quick brown",
            "I love machine", 
            "This is a",
            "Natural language"
        ]
        
        for i, exemple in enumerate(exemples, 1):
            print(f"\n{i}. Texte: '{exemple}'")
            predictions = predictor.predict_next_word(exemple, num_predictions=3)
            
            # Afficher les résultats
            if "ngram" in predictions:
                mots_ngram = [mot for mot, _ in predictions["ngram"] if mot]
                print(f"   📊 N-gramme: {', '.join(mots_ngram[:3])}")
            
            if "lstm" in predictions:
                mots_lstm = [mot for mot, _ in predictions["lstm"] if mot]
                print(f"   🧠 LSTM: {', '.join(mots_lstm[:3])}")
        
        # Test interactif simple
        print("\n" + "=" * 50)
        print("💬 TESTEZ VOS PROPRES TEXTES")
        print("=" * 50)
        print("Tapez un texte et voyez les prédictions!")
        print("(Tapez 'quit' pour quitter)")
        
        while True:
            try:
                texte = input("\n> ").strip()
                
                if texte.lower() == 'quit':
                    break
                
                if not texte:
                    continue
                
                predictions = predictor.predict_next_word(texte)
                
                if "ngram" in predictions:
                    mots_ngram = [mot for mot, _ in predictions["ngram"][:3] if mot]
                    print(f"📊 N-gramme: {', '.join(mots_ngram)}")
                
                if "lstm" in predictions:
                    mots_lstm = [mot for mot, _ in predictions["lstm"][:3] if mot]
                    print(f"🧠 LSTM: {', '.join(mots_lstm)}")
                
            except KeyboardInterrupt:
                break
        
        print("\n✅ Test terminé avec succès!")
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("💡 Installez les dépendances avec: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        print("💡 Vérifiez votre installation et réessayez")

def demo_simple_sans_entrainement():
    """Démonstration avec un modèle n-gramme simple sans deep learning"""
    
    print("\n🎯 DÉMONSTRATION SIMPLE (sans deep learning)")
    print("=" * 50)
    
    try:
        from word_predictor import NGramModel
        
        # Données d'exemple simples
        textes_exemple = [
            "the quick brown fox jumps over the lazy dog",
            "the cat sits on the mat",
            "the dog runs in the park",
            "the brown fox is very quick",
            "the lazy cat sleeps on the soft mat"
        ]
        
        # Créer et entraîner le modèle n-gramme
        model = NGramModel(n=3)
        model.train(textes_exemple)
        
        print("✅ Modèle n-gramme entraîné")
        
        # Tests
        exemples_test = [
            ["the", "quick"],
            ["the", "cat"],
            ["on", "the"],
            ["the", "lazy"]
        ]
        
        print("\n🔍 Prédictions:")
        for contexte in exemples_test:
            predictions = model.predict(contexte, num_predictions=3)
            mots = [mot for mot, _ in predictions if mot]
            print(f"   '{' '.join(contexte)}' → {', '.join(mots)}")
        
        print("\n✅ Démonstration simple terminée!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("Choisissez une option:")
    print("1. Test complet (avec LSTM)")
    print("2. Test simple (n-gramme seulement)")
    
    choix = input("\nEntrez votre choix (1 ou 2): ").strip()
    
    if choix == "2":
        demo_simple_sans_entrainement()
    else:
        test_rapide()