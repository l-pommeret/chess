import os
from .model import ChessGPT
from .trainer import ChessTrainer
from .config import ModelConfig, TrainingConfig, GenerationConfig

def main():
    # Chargement des données (supposons que train_dataset, test_dataset et tokenizer sont déjà créés)
    from .dataset import create_datasets
    from .tokenizer import ChessTokenizer
    
    # Configuration
    model_config = ModelConfig(vocab_size=len(tokenizer))
    training_config = TrainingConfig()
    generation_config = GenerationConfig()
    
    # Création du modèle
    chess_gpt = ChessGPT(model_config, tokenizer)
    
    # Sauvegarde du tokenizer
    chess_gpt.save_tokenizer(os.path.join(training_config.output_dir, "tokenizer"))
    
    # Configuration et lancement de l'entraînement
    trainer = ChessTrainer(
        model=chess_gpt.model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=training_config
    )
    
    # Entraînement
    trainer.train()
    trainer.save_model()
    
    # Test de génération
    prompt = "1.e4 c5 2.Nf3 d6 3.d4"
    generated_game = chess_gpt.generate(prompt, generation_config)
    print(f"Partie générée : {generated_game}")

if __name__ == "__main__":
    main()