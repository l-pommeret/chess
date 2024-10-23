# Token api HuggingFace

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, HfFolder
import shutil
import os

# Chemin vers le checkpoint que vous voulez sauvegarder
checkpoint_path = "/content/gpt2-chess-games/checkpoint-14000"

# Chemin où vous voulez sauvegarder le modèle final
final_model_path = "/content/final_model"

# Créez le répertoire pour le modèle final s'il n'existe pas
os.makedirs(final_model_path, exist_ok=True)

# Copiez les fichiers nécessaires
files_to_copy = ['config.json', 'model.safetensors', 'generation_config.json']
for file in files_to_copy:
    shutil.copy(os.path.join(checkpoint_path, file), final_model_path)

# Chargez le modèle à partir du checkpoint
config = AutoConfig.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config)

# Sauvegardez le modèle dans le nouveau répertoire
model.save_pretrained(final_model_path)

# Définissez le nom du dépôt sur Hugging Face où vous voulez charger le modèle
repo_name = "Zual/chess"

# Chargez le modèle sur Hugging Face
model.push_to_hub(repo_name)

print(f"Le modèle final (checkpoint-40000) a été sauvegardé et chargé sur {repo_name}")

