# tokenizer.py

import re
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import json
import os
from tqdm import tqdm

@dataclass
class TokenizerConfig:
    """Configuration du tokenizer."""
    pad_token: str = '<PAD>'
    initial_tokens: List[str] = None
    special_tokens: List[str] = None
    save_path: Optional[str] = None

class ChessTokenizer:
    """
    Tokenizer spécialisé pour la notation d'échecs.
    Gère la tokenization et le nettoyage des parties d'échecs en notation PGN.
    """
    
    def __init__(self, config: TokenizerConfig = None):
        """
        Initialise le tokenizer avec une configuration optionnelle.
        
        Args:
            config: Configuration du tokenizer (optionnel)
        """
        self.config = config or TokenizerConfig()
        
        # Initialisation du vocabulaire
        self.vocab: Dict[str, int] = {self.config.pad_token: 0}
        self.id2token: Dict[int, str] = {0: self.config.pad_token}
        self.token_id: int = 1
        
        # Ensemble des caractères valides
        self.valid_chars: Set[str] = set()
        
        # Initialisation du vocabulaire
        self._initialize_vocab()
        
        # Affichage des statistiques
        print(f"Taille du vocabulaire: {len(self.vocab)}")
        print(f"Nombre de caractères valides: {len(self.valid_chars)}")

    def is_valid_text(self, text: str) -> bool:
        """
        Vérifie si le texte ne contient que des caractères valides.
        
        Args:
            text: Texte à vérifier
            
        Returns:
            bool: True si le texte est valide, False sinon
        """
        # On nettoie d'abord le texte
        text = self.clean_text(text)
        
        # On vérifie que tous les caractères sont valides
        invalid_chars = set()
        for char in text:
            if char not in self.valid_chars:
                invalid_chars.add(char)
        
        if invalid_chars:
            print(f"Caractères invalides trouvés : {sorted(invalid_chars)}")
            return False
            
        # On vérifie que le texte n'est pas vide après nettoyage
        if not text.strip():
            print("Texte vide après nettoyage")
            return False
            
        # On vérifie que le texte contient au moins un coup d'échecs valide
        if not re.search(r'\d+\.', text):
            print("Aucun coup d'échecs trouvé")
            return False
            
        return True

# Vous pouvez aussi ajouter cette méthode debug pour aider au diagnostic :

    def debug_text(self, text: str) -> None:
        """
        Affiche des informations de debug sur le texte.
        
        Args:
            text: Texte à analyser
        """
        print("\nDEBUG TEXT:")
        print("Texte original:", text)
        print("Longueur:", len(text))
        print("Caractères uniques:", sorted(set(text)))
        
        cleaned = self.clean_text(text)
        print("\nTexte nettoyé:", cleaned)
        print("Longueur après nettoyage:", len(cleaned))
        print("Caractères uniques après nettoyage:", sorted(set(cleaned)))
        
        invalid_chars = set(char for char in text if char not in self.valid_chars)
        print("\nCaractères invalides:", sorted(invalid_chars))

    def _initialize_vocab(self) -> None:
        """Initialise le vocabulaire avec tous les tokens nécessaires."""
        # 1. Squares (e4, a1, etc.)
        for file in 'abcdefgh':
            for rank in '12345678':
                square = file + rank
                self._add_token(square)
                self.valid_chars.add(file)
                self.valid_chars.add(rank)

        # 2. Pièces et symboles de base
        basic_chars = set('RNBQKP')  # Pièces
        self.valid_chars.update(basic_chars)
        for char in basic_chars:
            self._add_token(char)

        # 3. Symboles de notation
        notation_chars = set('x+#=O-0123456789.')
        self.valid_chars.update(notation_chars)
        for char in notation_chars:
            self._add_token(char)

        # 4. Caractères de formatage
        format_chars = set(' \n\t(),[]{}"\'/!')
        self.valid_chars.update(format_chars)
        for char in format_chars:
            self._add_token(char)

        # 5. Tokens spéciaux de la configuration
        if self.config.special_tokens:
            for token in self.config.special_tokens:
                self._add_token(token)

    def _add_token(self, token: str) -> None:
        """Ajoute un token au vocabulaire s'il n'existe pas déjà."""
        if token not in self.vocab:
            self.vocab[token] = self.token_id
            self.id2token[self.token_id] = token
            self.token_id += 1

    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte en remplaçant ou supprimant les caractères invalides.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        # 1. Normalisation des caractères spéciaux
        replacements = {
            "'": "'",
            """: '"',
            """: '"',
            "–": "-",
            "—": "-",
            "…": "...",
            "\r": "\n",
            "\u2026": "..."
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 2. Suppression des caractères invalides
        cleaned = ''.join(char for char in text if char in self.valid_chars)
        
        # 3. Normalisation des espaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """
        Convertit le texte en liste de tokens.
        
        Args:
            text: Texte à tokenizer
            
        Returns:
            Liste de tokens
        """
        text = self.clean_text(text)
        tokens = []
        last_pos = 0
        
        # Capture d'abord les cases d'échecs (e.g., e4, a1)
        for match in re.finditer(r'[a-h][1-8]', text):
            start, end = match.span()
            # Ajouter les caractères entre la dernière correspondance et celle-ci
            if start > last_pos:
                tokens.extend(list(text[last_pos:start]))
            tokens.append(match.group())
            last_pos = end
            
        # Ajouter les caractères restants
        if last_pos < len(text):
            tokens.extend(list(text[last_pos:]))
            
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Convertit le texte en liste d'IDs.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Liste d'IDs de tokens
        """
        return [self.vocab[token] for token in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """
        Convertit une liste d'IDs en texte.
        
        Args:
            ids: Liste d'IDs à décoder
            
        Returns:
            Texte décodé
        """
        return ''.join(self.id2token[id] for id in ids 
                      if id != self.vocab[self.config.pad_token])

    def save(self, path: Optional[str] = None) -> None:
        """
        Sauvegarde le tokenizer au format JSON.
        
        Args:
            path: Chemin de sauvegarde (optionnel)
        """
        save_path = path or self.config.save_path
        if save_path is None:
            raise ValueError("Aucun chemin de sauvegarde spécifié")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'id2token': self.id2token,
            'valid_chars': list(self.valid_chars),
            'config': {
                'pad_token': self.config.pad_token,
                'special_tokens': self.config.special_tokens
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer sauvegardé dans {save_path}")

    @classmethod
    def load(cls, path: str) -> 'ChessTokenizer':
        """
        Charge un tokenizer depuis un fichier JSON.
        
        Args:
            path: Chemin du fichier à charger
            
        Returns:
            Instance de ChessTokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        config = TokenizerConfig(
            pad_token=data['config']['pad_token'],
            special_tokens=data['config']['special_tokens'],
            save_path=path
        )
        
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.id2token = {int(k): v for k, v in data['id2token'].items()}
        tokenizer.valid_chars = set(data['valid_chars'])
        tokenizer.token_id = max(map(int, data['id2token'].keys())) + 1
        
        return tokenizer

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                    show_progress: bool = True) -> List[List[int]]:
        """
        Encode une liste de textes avec gestion de la longueur maximale.
        
        Args:
            texts: Liste de textes à encoder
            max_length: Longueur maximale (optionnel)
            show_progress: Affiche une barre de progression
            
        Returns:
            Liste de listes d'IDs de tokens
        """
        iterator = tqdm(texts) if show_progress else texts
        encoded = [self.encode(text) for text in iterator]
        
        if max_length is not None:
            pad_id = self.vocab[self.config.pad_token]
            encoded = [
                seq[:max_length] if len(seq) > max_length
                else seq + [pad_id] * (max_length - len(seq))
                for seq in encoded
            ]
            
        return encoded

    @property
    def vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        return len(self.vocab)

    def __len__(self) -> int:
        """Retourne la taille du vocabulaire."""
        return self.vocab_size

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = TokenizerConfig(
        pad_token='<PAD>',
        special_tokens=['<START>', '<END>'],
        save_path='./models/chess_tokenizer.json'
    )
    
    # Création du tokenizer
    tokenizer = ChessTokenizer(config)
    
    # Test avec une partie d'échecs
    game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    tokens = tokenizer.tokenize(game)
    print(f"\nTokens: {tokens}")
    
    ids = tokenizer.encode(game)
    print(f"IDs: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"Décodé: {decoded}")
    
    # Sauvegarde et chargement
    tokenizer.save()
    loaded_tokenizer = ChessTokenizer.load('./models/chess_tokenizer.json')