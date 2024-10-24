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

# Sauvegarde des données du tokenizer
tokenizer_data = {
    'vocab': tokenizer.vocab,
    'ids_to_tokens': tokenizer.ids_to_tokens,
    'next_id': tokenizer.next_id
}

tokenizer_save_path = os.path.join(tokenizer_save_dir, "tokenizer_data.json")
with open(tokenizer_save_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

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

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir=model_save_dir,
    num_train_epochs=50,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=1000,
    eval_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    lr_scheduler_type="linear",
    warmup_steps=500,
    fp16=True,
)

# Configuration du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Début de l'entraînement...")
trainer.train()
print("Entraînement terminé!")

# Sauvegarde du modèle
trainer.save_model()

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