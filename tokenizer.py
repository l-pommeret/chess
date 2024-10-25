import re
from typing import List, Set

class ChessTokenizer:
    """Tokenizer spécialisé pour la notation d'échecs."""
    
    def __init__(self):
        self.vocab = {'<PAD>': 0}
        self.id2token = {0: '<PAD>'}
        self.valid_chars = self._initialize_vocab()

    def _initialize_vocab(self) -> Set[str]:
        """Initialise le vocabulaire avec tous les caractères valides de la notation d'échecs."""
        valid_chars = set()
        token_id = 1

        # Cases d'échecs (e.g., e4, a1)
        for file in 'abcdefgh':
            for rank in '12345678':
                square = file + rank
                self.vocab[square] = token_id
                self.id2token[token_id] = square
                token_id += 1
                valid_chars.add(file)
                valid_chars.add(rank)

        # Caractères de notation d'échecs
        chess_chars = set('RNBQK()x+#=O-123456789 ')
        for char in chess_chars:
            if char not in self.vocab:
                self.vocab[char] = token_id
                self.id2token[token_id] = char
                token_id += 1
                valid_chars.add(char)

        return valid_chars

    def is_valid_text(self, text: str) -> bool:
        """Vérifie si le texte ne contient que des caractères valides."""
        return all(char in self.valid_chars for char in text)

    def tokenize(self, text: str) -> List[str]:
        """Convertit le texte en liste de tokens."""
        if not self.is_valid_text(text):
            raise ValueError("Caractères invalides détectés")
        
        tokens = []
        last_pos = 0
        
        # Capture les cases d'échecs comme tokens uniques
        for match in re.finditer(r'[a-h][1-8]', text):
            start, end = match.span()
            if start > last_pos:
                tokens.extend(text[last_pos:start])
            tokens.append(match.group())
            last_pos = end
            
        if last_pos < len(text):
            tokens.extend(text[last_pos:])
            
        return tokens

    def encode(self, text: str) -> List[int]:
        """Convertit le texte en liste d'IDs."""
        return [self.vocab[token] for token in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """Convertit une liste d'IDs en texte."""
        return ''.join(self.id2token[id] for id in ids if id != self.vocab['<PAD>'])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
