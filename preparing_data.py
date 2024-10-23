import requests
import os
import zstandard as zstd
from tqdm import tqdm

def download_file(url, save_path):
    local_filename = url.split('/')[-1]
    full_path = os.path.join(save_path, local_filename)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(full_path, 'wb') as file, tqdm(
        desc=local_filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    return full_path

def decompress_zst(input_path, output_path):
    dctx = zstd.ZstdDecompressor()
    input_size = os.path.getsize(input_path)

    with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh, tqdm(
        desc="Décompression",
        total=input_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        reader = dctx.stream_reader(ifh)
        while True:
            chunk = reader.read(8192)
            if not chunk:
                break
            ofh.write(chunk)
            progress_bar.update(len(chunk))

    print(f'Fichier décompressé : {output_path}')

# URL du fichier .zst
file_url = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-09.pgn.zst"

# Chemin où sauvegarder le fichier
save_path = "."

print("Téléchargement du fichier...")
downloaded_file = download_file(file_url, save_path)

print("Décompression du fichier...")
output_file = downloaded_file.rsplit('.', 1)[0]  # Retire l'extension .zst
decompress_zst(downloaded_file, output_file)

print(f"Le fichier PGN est prêt à être utilisé : {output_file}")

# Optionnel : supprimer le fichier .zst après décompression
# os.remove(downloaded_file)
# print(f"Fichier compressé supprimé : {downloaded_file}")

import re
from tqdm import tqdm
import os

class ChessSquareTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.ids_to_tokens = {0: '<PAD>', 1: '<UNK>'}
        self.next_id = 2

        # Préparation des tokens pour toutes les cases possibles
        files = 'abcdefgh'
        ranks = '12345678'
        for f in files:
            for r in ranks:
                square = f + r
                self.vocab[square] = self.next_id
                self.ids_to_tokens[self.next_id] = square
                self.next_id += 1

    def build_vocab(self, texts):
        # Pattern pour identifier les cases d'échecs
        square_pattern = r'[a-h][1-8]'

        for text in texts:
            tokens = []
            last_end = 0
            for match in re.finditer(square_pattern, text):
                start, end = match.span()
                if start > last_end:
                    tokens.extend(list(text[last_end:start]))
                tokens.append(match.group())
                last_end = end
            if last_end < len(text):
                tokens.extend(list(text[last_end:]))

            for token in tokens:
                if token not in self.vocab and len(token) == 1:
                    self.vocab[token] = self.next_id
                    self.ids_to_tokens[self.next_id] = token
                    self.next_id += 1

        print(f"Taille du vocabulaire: {len(self.vocab)}")

    def tokenize(self, text):
        tokens = []
        last_end = 0
        square_pattern = r'[a-h][1-8]'

        for match in re.finditer(square_pattern, text):
            start, end = match.span()
            if start > last_end:
                tokens.extend(list(text[last_end:start]))
            tokens.append(match.group())
            last_end = end

        if last_end < len(text):
            tokens.extend(list(text[last_end:]))

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens[id] for id in ids]

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        return ''.join(self.convert_ids_to_tokens(ids))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)
    
    import re
from tqdm import tqdm
import os

class ChessSquareTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.ids_to_tokens = {0: '<PAD>', 1: '<UNK>'}
        self.next_id = 2

        # Préparation des tokens pour toutes les cases possibles
        files = 'abcdefgh'
        ranks = '12345678'
        for f in files:
            for r in ranks:
                square = f + r
                self.vocab[square] = self.next_id
                self.ids_to_tokens[self.next_id] = square
                self.next_id += 1

    def build_vocab(self, texts):
        # Pattern pour identifier les cases d'échecs
        square_pattern = r'[a-h][1-8]'

        for text in texts:
            tokens = []
            last_end = 0
            for match in re.finditer(square_pattern, text):
                start, end = match.span()
                if start > last_end:
                    tokens.extend(list(text[last_end:start]))
                tokens.append(match.group())
                last_end = end
            if last_end < len(text):
                tokens.extend(list(text[last_end:]))

            for token in tokens:
                if token not in self.vocab and len(token) == 1:
                    self.vocab[token] = self.next_id
                    self.ids_to_tokens[self.next_id] = token
                    self.next_id += 1

        print(f"Taille du vocabulaire: {len(self.vocab)}")

    def tokenize(self, text):
        tokens = []
        last_end = 0
        square_pattern = r'[a-h][1-8]'

        for match in re.finditer(square_pattern, text):
            start, end = match.span()
            if start > last_end:
                tokens.extend(list(text[last_end:start]))
            tokens.append(match.group())
            last_end = end

        if last_end < len(text):
            tokens.extend(list(text[last_end:]))

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens[id] for id in ids]

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        return ''.join(self.convert_ids_to_tokens(ids))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)

def optimize_pgn(pgn_text):
    optimized = re.sub(r'(\d+\.) +', r'\1', pgn_text)
    optimized = re.sub(r' +', ' ', optimized)
    return optimized.strip()

def parse_time_control(line):
    """Parse la ligne de contrôle du temps et retourne le temps initial en minutes"""
    match = re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', line)
    if match:
        initial_time = int(match.group(1))
        increment = int(match.group(2))
        return initial_time / 60  # Convertir en minutes
    return None

def read_pgn_file(file_path, max_games=None):
    games = []
    current_game = []
    game_count = 0
    filtered_game_count = 0
    total_game_count = 0
    in_header = True
    white_elo = black_elo = 0
    num_moves = 0
    time_control = None

    def parse_elo(line):
        match = re.search(r'\[(\w+)Elo "(\d+)"\]', line)
        if match:
            return int(match.group(2))
        return 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Lecture du fichier PGN"):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                if not in_header:
                    total_game_count += 1
                    # Vérifier tous les critères de filtrage
                    if (current_game and
                        white_elo > 1800 and black_elo > 1800 and
                        10 <= num_moves <= 60 and
                        time_control is not None and time_control >= 3):  # Ajouter le filtre de temps
                        games.append(optimize_pgn(' '.join(current_game)))
                        filtered_game_count += 1
                        game_count += 1
                        if max_games and game_count >= max_games:
                            break
                    current_game = []
                    white_elo = black_elo = 0
                    num_moves = 0
                    time_control = None
                in_header = True
                if "WhiteElo" in line:
                    white_elo = parse_elo(line)
                elif "BlackElo" in line:
                    black_elo = parse_elo(line)
                elif "TimeControl" in line:
                    time_control = parse_time_control(line)
            elif line:
                in_header = False
                current_game.append(line)
                num_moves += line.count('.')

    # Traiter la dernière partie
    if (current_game and
        white_elo > 1800 and black_elo > 1800 and
        10 <= num_moves <= 60 and
        time_control is not None and time_control >= 3):
        games.append(optimize_pgn(' '.join(current_game)))
        filtered_game_count += 1
        total_game_count += 1

    return games, filtered_game_count, total_game_count

# Chemin vers le fichier PGN
pgn_file_path = "./lichess_db_standard_rated_2016-09.pgn"

# Lire les parties d'échecs
print("Lecture du fichier PGN...")
chess_games, filtered_count, total_count = read_pgn_file(pgn_file_path, max_games=None)

# Créer et entraîner le tokenizer
tokenizer = ChessSquareTokenizer()
print("Construction du vocabulaire...")
tokenizer.build_vocab(chess_games)

# Tokenize toutes les parties et trouve la plus longue
print("Tokenization des parties...")
max_tokens = 0
for game in tqdm(chess_games, desc="Tokenization"):
    tokens = tokenizer.encode(game)
    max_tokens = max(max_tokens, len(tokens))

# Afficher les statistiques
print(f"\nNombre total de parties lues: {total_count}")
print(f"Nombre de parties après filtrage: {filtered_count}")
print(f"Taille du vocabulaire: {tokenizer.vocab_size}")
print(f"Nombre de tokens de la plus longue partie: {max_tokens}")

# Exemple d'utilisation
if chess_games:
    print("\nExemple d'utilisation du tokenizer:")
    example_game = chess_games[0]
    encoded_game = tokenizer.encode(example_game)
    print(f"Longueur de la partie encodée: {len(encoded_game)}")
    print(f"Premiers 100 tokens encodés: {encoded_game[:100]}")
    decoded_game = tokenizer.decode(encoded_game[:100])
    print(f"\nDécodage des 100 premiers tokens:\n{decoded_game}")
    print(f"\nDébut d'une partie d'échecs (sans en-tête):\n{chess_games[0][:200]}...")
else:
    print("Aucune partie n'a été trouvée qui corresponde aux critères de filtrage.")

import torch
from torch.utils.data import Dataset

class ChessGameDataset(Dataset):
    def __init__(self, games, tokenizer, max_length):
        self.games = games
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        encoded = self.tokenizer.encode(game)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded += [self.tokenizer.vocab['<PAD>']] * (self.max_length - len(encoded))

        return {
            'input_ids': torch.tensor(encoded),
            'labels': torch.tensor(encoded)
        }
    
import random
from torch.utils.data import random_split

# Assurez-vous que chess_games est défini (résultat de read_pgn_file)
# Si ce n'est pas le cas, exécutez d'abord le code de lecture du fichier PGN

# Mélange aléatoire des parties
random.shuffle(chess_games)

# Calcul de la taille du test dataset (0,5% des données)
test_size = int(0.0005 * len(chess_games))
train_size = len(chess_games) - test_size

# Création des datasets
train_dataset = ChessGameDataset(chess_games[:train_size], tokenizer, max_length=600)
test_dataset = ChessGameDataset(chess_games[train_size:], tokenizer, max_length=600)

# Affichage des tailles des datasets
print(f"Taille du train dataset : {len(train_dataset)}")
print(f"Taille du test dataset : {len(test_dataset)}")

import random

# Sélectionner un index aléatoire
random_index = random.randint(0, len(train_dataset) - 1)

# Récupérer le datapoint aléatoire
random_datapoint = train_dataset[random_index]

# Décoder et afficher la partie d'échecs originale
decoded_game = tokenizer.decode(random_datapoint['input_ids'].tolist())
print("\nPartie d'échecs décodée:")
print(decoded_game.strip())  # strip() pour enlever les PAD à la fin

# Afficher quelques statistiques
print("\nStatistiques:")
print(f"Longueur de la partie: {len(decoded_game.strip())} caractères")
print(f"Nombre de tokens: {len(random_datapoint['input_ids'])}")

# Afficher les 10 premiers caractères du vocabulaire (en excluant <PAD> et <UNK>)
print("\nExemple de caractères dans le vocabulaire:")
print(list(tokenizer.vocab.keys())[2:12])

