from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer
import torch
from preparing_data import train_dataset, test_dataset, tokenizer

# Assurez-vous que vous avez déjà défini et créé train_dataset et test_dataset

# Initialisation du modèle
config = GPT2Config(
    vocab_size=len(tokenizer),  # Utilisez la taille de votre vocabulaire personnalisé
    n_positions=600,  # Ajustez cela à la longueur maximale de vos séquences
    n_ctx=600,  # Même valeur que n_positions
    n_embd=512,  # Vous pouvez ajuster cela si vous voulez un modèle plus petit ou plus grand
    n_layer=10,
    n_head=8
)
model = GPT2LMHeadModel(config)
print(config)
print(model)

from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

# Définition des hyperparamètres
num_train_epochs = 50
learning_rate = 5e-3
batch_size = 64  # Ajustez en fonction de la mémoire de votre GPU
warmup_steps = 500

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./gpt2-chess-games",
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
    fp16=True,  # Utilisez ceci si vous avez un GPU compatible avec la précision mixte
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

def generate_game(prompt, max_length=100):
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95,
                            temperature=0.1,
                            pad_token_id=tokenizer.vocab['<PAD>'])
    return tokenizer.decode(output[0].tolist())

# Exemple de génération
prompt = "1.e4 c5 2.Nf3 d6 3.d4"

generated_game = generate_game(prompt)
print(f"Partie générée : {generated_game}")