from huggingface_hub import notebook_login, HfApi, HfFolder, hf_hub_download
from transformers import AutoModelForCausalLM, AutoConfig
import shutil
import os
import json
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def setup_paths(checkpoint_path, tokenizer_path):
    paths = {
        'checkpoint': Path(checkpoint_path).resolve(),
        'tokenizer': Path(tokenizer_path).resolve(),
        'final_model': Path('final_model').resolve()
    }
    paths['final_model'].mkdir(exist_ok=True)
    return paths

def verify_paths(paths, logger):
    if not paths['checkpoint'].exists():
        logger.error(f"Checkpoint path does not exist: {paths['checkpoint']}")
        return False
    if not paths['tokenizer'].exists():
        logger.error(f"Tokenizer path does not exist: {paths['tokenizer']}")
        return False
    logger.info(f"Working with checkpoint: {paths['checkpoint']}")
    logger.info(f"Working with tokenizer: {paths['tokenizer']}")
    return True

def push_to_hub(model_path, repo_name, tokenizer_file_path, logger):
    try:
        # Configuration du token manuellement si nécessaire
        # os.environ["HF_TOKEN"] = "votre-token-ici"  # Décommentez et ajoutez votre token si nécessaire
        
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Vérifier si le modèle existe
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Charger et pousser le modèle
        logger.info("Loading model configuration...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True
        )
        
        logger.info(f"Pushing model to hub: {repo_name}")
        model.push_to_hub(repo_name, use_auth_token=True)

        # Upload du tokenizer
        api = HfApi()
        if tokenizer_file_path.exists():
            logger.info(f"Uploading tokenizer from: {tokenizer_file_path}")
            api.upload_file(
                path_or_fileobj=str(tokenizer_file_path),
                path_in_repo=tokenizer_file_path.name,
                repo_id=repo_name,
                repo_type="model",
                use_auth_token=True
            )
            logger.info('Tokenizer uploaded successfully')
        
        # Vérification des fichiers
        files = api.list_repo_files(repo_name)
        logger.info('\nFiles in repository:')
        for file in files:
            logger.info(f'- {file}')
            
        return True
    except Exception as e:
        logger.error(f'Error during hub upload: {str(e)}')
        return False

def main():
    # Configuration
    logger = setup_logging()
    
    # Utilisez des chemins absolus ou relatifs corrects
    CHECKPOINT_PATH = "./gpt2-chess-games/checkpoint-1000"  # Ajustez ce chemin
    TOKENIZER_PATH = "./gpt2-chess-games/tokenizer"        # Ajustez ce chemin
    REPO_NAME = "Zual/chess"
    TOKENIZER_FILE = "tokenizer_data.json"

    try:
        # Login
        notebook_login()
        logger.info('Successfully logged in to Hugging Face')

        # Setup et vérification des chemins
        paths = setup_paths(CHECKPOINT_PATH, TOKENIZER_PATH)
        if not verify_paths(paths, logger):
            raise Exception("Path verification failed")

        # Tentative de push vers le hub
        tokenizer_file_path = paths['final_model'] / TOKENIZER_FILE
        if push_to_hub(str(paths['checkpoint']), REPO_NAME, tokenizer_file_path, logger):
            logger.info("Successfully pushed model to hub")
        else:
            raise Exception("Failed to push to hub")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()