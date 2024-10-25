import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

@dataclass
class ChessGame:
    """Stocke les informations d'une partie d'échecs."""
    moves: str
    white_elo: int
    black_elo: int
    time_control: float
    num_moves: int

class PGNParser:
    """Parse et filtre les fichiers PGN."""
    
    def __init__(self, tokenizer, min_elo: int = 1800, min_moves: int = 10,
                 max_moves: int = 60, min_time_control: float = 3.0):
        self.tokenizer = tokenizer
        self.min_elo = min_elo
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.min_time_control = min_time_control

    def clean_pgn(self, text: str) -> str:
        """Nettoie le texte PGN."""
        # Supprime les commentaires et variantes
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        # Optimise les espaces
        text = re.sub(r'(\d+\.) +', r'\1', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def parse_time_control(line: str) -> Optional[float]:
        """Extrait le contrôle du temps d'une ligne d'en-tête."""
        if match := re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', line):
            return int(match.group(1)) / 60
        return None

    @staticmethod
    def parse_elo(line: str) -> int:
        """Extrait l'Elo d'une ligne d'en-tête."""
        if match := re.search(r'\[(\w+)Elo "(\d+)"\]', line):
            return int(match.group(2))
        return 0

    def is_valid_game(self, game: ChessGame) -> bool:
        """Vérifie si une partie répond aux critères de filtrage."""
        return (game.white_elo > self.min_elo and
                game.black_elo > self.min_elo and
                self.min_moves <= game.num_moves <= self.max_moves and
                game.time_control >= self.min_time_control)

    def parse_file(self, filepath: str, max_games: Optional[int] = None) -> Tuple[List[str], int, int]:
        """Parse un fichier PGN et retourne les parties valides."""
        games = []
        total_count = 0
        filtered_count = 0
        current_game = []
        current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing PGN"):
                line = line.strip()
                
                if line.startswith('['):
                    if "WhiteElo" in line:
                        current.white_elo = self.parse_elo(line)
                    elif "BlackElo" in line:
                        current.black_elo = self.parse_elo(line)
                    elif "TimeControl" in line:
                        current.time_control = self.parse_time_control(line) or 0
                elif line:
                    current_game.append(line)
                    current.num_moves += line.count('.')
                elif current_game:
                    total_count += 1
                    
                    if self.is_valid_game(current):
                        clean_game = self.clean_pgn(' '.join(current_game))
                        if self.tokenizer.is_valid_text(clean_game):
                            games.append(clean_game)
                            filtered_count += 1
                            if max_games and filtered_count >= max_games:
                                break
                    
                    current_game = []
                    current = ChessGame(moves="", white_elo=0, black_elo=0, time_control=0, num_moves=0)

        return games, filtered_count, total_count