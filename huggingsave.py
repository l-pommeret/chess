from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, HfFolder
import shutil
import os
import json

# Login to HuggingFace
notebook_login()

# Chemins
checkpoint_path = "gpt2-chess-games/checkpoint-1000"
tokenizer_path = "gpt2-chess-games/tokenizer"
final_model_path = "final_model"

# Créez le répertoire final
os.makedirs(final_model_path, exist_ok=True)

# Copiez les fichiers du modèle
files_to_copy = ['config.json', 'model.safetensors', 'generation_config.json']
for file in files_to_copy:
    if os.path.exists(os.path.join(checkpoint_path, file)):
        shutil.copy(os.path.join(checkpoint_path, file), final_model_path)

# Copie des données du tokenizer
tokenizer_file = "tokenizer_data.json"
if os.path.exists(os.path.join(tokenizer_path, tokenizer_file)):
    shutil.copy(
        os.path.join(tokenizer_path, tokenizer_file),
        os.path.join(final_model_path, tokenizer_file)
    )

# Chargez le modèle
config = AutoConfig.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config)

# Définissez le nom du dépôt
repo_name = "Zual/chess"

# Chargez le modèle sur Hugging Face
model.push_to_hub(repo_name)

# Upload manuel du fichier du tokenizer
api = HfApi()
tokenizer_file_path = os.path.join(final_model_path, tokenizer_file)
if os.path.exists(tokenizer_file_path):
    api.upload_file(
        path_or_fileobj=tokenizer_file_path,
        path_in_repo=tokenizer_file,
        repo_id=repo_name,
        repo_type="model"
    )

print(f"Le modèle et les données du tokenizer ont été chargés sur {repo_name}")

# Vérification des fichiers
files = api.list_repo_files(repo_name)
print("\nFichiers dans le dépôt:")
for file in files:
    print(f"- {file}")

print("\nPour charger le modèle et le tokenizer plus tard:")
print("""
import json
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Charger le modèle
model = AutoModelForCausalLM.from_pretrained("Zual/chess")

# Charger les données du tokenizer
tokenizer_path = hf_hub_download(repo_id="Zual/chess", filename="tokenizer_data.json")
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

# Créer une nouvelle instance du tokenizer
tokenizer = ChessSquareTokenizer()
tokenizer.vocab = tokenizer_data['vocab']
tokenizer.ids_to_tokens = {int(k): v for k, v in tokenizer_data['ids_to_tokens'].items()}
tokenizer.next_id = tokenizer_data['next_id']
""")