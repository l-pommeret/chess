from preparing_data import ChessDataDownloader, ChessSquareTokenizer, PGNParser, create_datasets

# Configuration
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-09.pgn.zst"

# Téléchargement et décompression
downloader = ChessDataDownloader(url)
zst_file = downloader.download()
pgn_file = downloader.decompress(zst_file)

# Créer d'abord le tokenizer
tokenizer = ChessSquareTokenizer()

# L'utiliser dans le parser
parser = PGNParser(min_elo=1800, tokenizer=tokenizer)
games, filtered_count, total_count = parser.parse_file(pgn_file)

# Créer les datasets
train_dataset, test_dataset = create_datasets(games, tokenizer, max_length=600)