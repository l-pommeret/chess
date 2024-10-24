# Token api HuggingFace
from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
from huggingface_hub import HfApi, HfFolder
import shutil
import os

# Login to HuggingFace
notebook_login()

# Chemins des fichiers
checkpoint_path = "gpt2-chess-games/checkpoint-14000"
tokenizer_path = "gpt2-chess-games/tokenizer"
final_model_path = "final_model"

# Créez le répertoire pour le modèle final
os.makedirs(final_model_path, exist_ok=True)

# Copiez les fichiers du modèle
files_to_copy = ['config.json', 'model.safetensors', 'generation_config.json']
for file in files_to_copy:
    if os.path.exists(os.path.join(checkpoint_path, file)):
        shutil.copy(os.path.join(checkpoint_path, file), final_model_path)

# Copiez les fichiers du tokenizer
tokenizer_files = os.listdir(tokenizer_path)
for file in tokenizer_files:
    shutil.copy(
        os.path.join(tokenizer_path, file),
        os.path.join(final_model_path, file)
    )

# Chargez le modèle et le tokenizer
config = AutoConfig.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config)
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Sauvegardez le modèle et le tokenizer dans le nouveau répertoire
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Définissez le nom du dépôt
repo_name = "Zual/chess"

# Chargez le modèle et le tokenizer sur Hugging Face
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

# Vérification des fichiers uploadés
api = HfApi()
files = api.list_repo_files(repo_name)
print("\nFichiers uploadés sur HuggingFace:")
for file in files:
    print(f"- {file}")

# Test rapide pour vérifier que tout fonctionne
test_text = "1.e4 e5 2.Nf3"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"\nTest de vérification:")
print(f"Texte original: {test_text}")
print(f"Texte décodé : {decoded}")

print(f"\nLe modèle et le tokenizer ont été sauvegardés et chargés sur {repo_name}")

# Instructions pour utiliser le modèle plus tard
print("\nPour utiliser ce modèle plus tard, utilisez le code suivant:")
print("""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Zual/chess")
tokenizer = AutoTokenizer.from_pretrained("Zual/chess")
""")