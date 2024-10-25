import os
import re
import torch
import requests
import zstandard as zstd
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class ChessGame:
    moves: str
    white_elo: int
    black_elo: int
    time_control: float
    num_moves: int

class ChessDataDownloader:
    def __init__(self, url: str, save_path: str = "."):
        self.url = url
        self.save_path = save_path

    def download(self) -> str:
        local_filename = self.url.split('/')[-1]
        full_path = os.path.join(self.save_path, local_filename)

        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(full_path, 'wb') as file, tqdm(
            desc=f"Downloading {local_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

        return full_path

    def decompress(self, input_path: str) -> str:
        output_path = input_path.rsplit('.', 1)[0]
        dctx = zstd.ZstdDecompressor()
        
        with open(input_path, 'rb') as ifh, \
             open(output_path, 'wb') as ofh, \
             tqdm(desc="Decompressing", unit='iB', unit_scale=True) as pbar:
            reader = dctx.stream_reader(ifh)
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                ofh.write(chunk)
                pbar.update(len(chunk))

        return output_path

class ChessSquareTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0}  # Removed <UNK> as we won't use it
        self.ids_to_tokens = {0: '<PAD>'}
        self.valid_characters = self._initialize_vocab()

    def _initialize_vocab(self) -> Set[str]:
        valid_chars = set()
        next_id = 1  # Start from 1 since 0 is <PAD>
        
        # Add chess squares (e.g., 'e4', 'a1', etc.)
        for file in 'abcdefgh':
            for rank in '12345678':
                square = file + rank
                self.vocab[square] = next_id
                self.ids_to_tokens[next_id] = square
                next_id += 1
                valid_chars.add(file)
                valid_chars.add(rank)

        # Add other valid chess notation characters
        chess_chars = set('RNBQK()x+#=O-123456789 ')  # All valid chess notation characters
        for char in chess_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.ids_to_tokens[next_id] = char
                next_id += 1
                valid_chars.add(char)

        return valid_chars

    def is_valid_text(self, text: str) -> bool:
        """Vérifie si le texte ne contient que des caractères valides."""
        return all(char in self.valid_characters for char in text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize le texte en préservant les cases d'échecs comme tokens uniques."""
        if not self.is_valid_text(text):
            raise ValueError("Le texte contient des caractères non valides pour la notation d'échecs")
            
        tokens = []
        last_end = 0
        # Capture d'abord les cases d'échecs (e.g., e4, a1, etc.)
        for match in re.finditer(r'[a-h][1-8]', text):
            start, end = match.span()
            # Ajouter les caractères entre la dernière correspondance et celle-ci
            if start > last_end:
                tokens.extend(list(text[last_end:start]))
            tokens.append(match.group())  # Ajouter la case d'échecs comme un seul token
            last_end = end
            
        # Ajouter les caractères restants
        if last_end < len(text):
            tokens.extend(list(text[last_end:]))
            
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode le texte en IDs. Lève une exception si des caractères invalides sont trouvés."""
        return [self.vocab[token] for token in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.ids_to_tokens[id] for id in ids if id != self.vocab['<PAD>'])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

class PGNParser:
    def __init__(self, min_elo: int = 1800, min_moves: int = 10, max_moves: int = 60, 
                 min_time_control: float = 3.0, tokenizer: ChessSquareTokenizer = None):
        self.min_elo = min_elo
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.min_time_control = min_time_control
        self.tokenizer = tokenizer

    @staticmethod
    def _optimize_pgn(pgn_text: str) -> str:
        """Optimise le format PGN en supprimant les espaces superflus."""
        optimized = re.sub(r'(\d+\.) +', r'\1', pgn_text)
        optimized = re.sub(r' +', ' ', optimized)
        # Supprimer les commentaires entre accolades et les variantes entre parenthèses
        optimized = re.sub(r'\{[^}]*\}', '', optimized)
        optimized = re.sub(r'\([^)]*\)', '', optimized)
        return optimized.strip()

    @staticmethod
    def _parse_time_control(line: str) -> Optional[float]:
        match = re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', line)
        if match:
            return int(match.group(1)) / 60
        return None

    @staticmethod
    def _parse_elo(line: str) -> int:
        match = re.search(r'\[(\w+)Elo "(\d+)"\]', line)
        return int(match.group(2)) if match else 0

    def parse_file(self, file_path: str, max_games: Optional[int] = None) -> Tuple[List[str], int, int]:
        games = []
        filtered_count = total_count = 0
        invalid_format_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as file:
            current_game = []
            current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)
            
            for line in tqdm(file, desc="Parsing PGN"):
                line = line.strip()
                
                if line.startswith('['):
                    if "WhiteElo" in line:
                        current.white_elo = self._parse_elo(line)
                    elif "BlackElo" in line:
                        current.black_elo = self._parse_elo(line)
                    elif "TimeControl" in line:
                        current.time_control = self._parse_time_control(line) or 0
                elif line:
                    current_game.append(line)
                    current.num_moves += line.count('.')
                elif current_game:
                    total_count += 1
                    
                    if self._is_valid_game(current):
                        optimized_game = self._optimize_pgn(' '.join(current_game))
                        try:
                            # Vérifier si le texte peut être tokenizé sans erreur
                            if self.tokenizer and self.tokenizer.is_valid_text(optimized_game):
                                games.append(optimized_game)
                                filtered_count += 1
                                if max_games and filtered_count >= max_games:
                                    break
                            else:
                                invalid_format_count += 1
                        except ValueError:
                            invalid_format_count += 1
                            
                    current_game = []
                    current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)

        print(f"Parties filtrées pour format invalide: {invalid_format_count}")
        return games, filtered_count, total_count

    def _is_valid_game(self, game: ChessGame) -> bool:
        return (game.white_elo > self.min_elo and 
                game.black_elo > self.min_elo and
                self.min_moves <= game.num_moves <= self.max_moves and
                game.time_control >= self.min_time_control)

class ChessGameDataset(Dataset):
    def __init__(self, games: List[str], tokenizer: ChessSquareTokenizer, max_length: int):
        self.games = games
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_games()

    def _validate_games(self):
        """Vérifie que tous les jeux peuvent être tokenizés sans erreur."""
        valid_games = []
        for game in tqdm(self.games, desc="Validating games"):
            try:
                _ = self.tokenizer.encode(game)  # Test encoding
                valid_games.append(game)
            except ValueError as e:
                continue
        self.games = valid_games
        print(f"Jeux valides après vérification: {len(self.games)}")

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

def create_datasets(games: List[str], tokenizer: ChessSquareTokenizer, 
                   max_length: int, test_ratio: float = 0.0005) -> Tuple[Dataset, Dataset]:
    test_size = int(test_ratio * len(games))
    train_size = len(games) - test_size
    
    train_data = ChessGameDataset(games[:train_size], tokenizer, max_length)
    test_data = ChessGameDataset(games[train_size:], tokenizer, max_length)
    
    return train_data, test_data