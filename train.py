from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer
import torch
from preparing_data import train_dataset, test_dataset, tokenizer
import os
import json

# Création du dossier pour sauvegarder le modèle et le tokenizer
model_save_dir = "./gpt2-chess-games"
tokenizer_save_dir = os.path.join(model_save_dir, "tokenizer")
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(tokenizer_save_dir, exist_ok=True)

# Sauvegarde complète du tokenizer
tokenizer.save_pretrained(tokenizer_save_dir)

# Sauvegarde additionnelle du vocabulaire séparément pour plus de sécurité
tokenizer_save_path = os.path.join(tokenizer_save_dir, "tokenizer_vocab.json")
with open(tokenizer_save_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)

# Initialisation du modèle
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=600,
    n_ctx=600,
    n_embd=512,
    n_layer=10,
    n_head=8
)
model = GPT2LMHeadModel(config)
print(config)
print(model)

from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

# Définition des hyperparamètres
num_train_epochs = 50
learning_rate = 1e-3
batch_size = 64
warmup_steps = 500

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir=model_save_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=1000,
    eval_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    lr_scheduler_type="linear",
    warmup_steps=warmup_steps,
    fp16=True,
)

# Configuration du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Lancement de l'entraînement
print("Début de l'entraînement...")
trainer.train()
print("Entraînement terminé!")

# Sauvegarde du modèle
trainer.save_model()

# Sauvegarde des configurations spéciales du tokenizer
tokenizer_config = {
    "special_tokens": {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token,
        "bos_token": tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else None,
        # Ajoutez d'autres tokens spéciaux si nécessaire
    },
    "max_length": tokenizer.model_max_length,
    "padding_side": tokenizer.padding_side,
    "truncation_side": tokenizer.truncation_side if hasattr(tokenizer, 'truncation_side') else "right"
}
tokenizer_config_path = os.path.join(tokenizer_save_dir, "tokenizer_config.json")
with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

def generate_game(prompt, max_length=100):
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.1,
        pad_token_id=tokenizer.vocab['<PAD>']
    )
    return tokenizer.decode(output[0].tolist())

# Exemple de génération
prompt = "1.e4 c5 2.Nf3 d6 3.d4"
generated_game = generate_game(prompt)
print(f"Partie générée : {generated_game}")

# Test de chargement du tokenizer (optionnel)
print("\nTest de rechargement du tokenizer...")
from transformers import PreTrainedTokenizerFast
loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_save_dir)
test_text = "1.e4 e5"
encoded = loaded_tokenizer.encode(test_text)
decoded = loaded_tokenizer.decode(encoded)
print(f"Test de tokenization - Original: {test_text}")
print(f"Test de tokenization - Décodé : {decoded}")