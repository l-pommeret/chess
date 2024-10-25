from hugging import HuggingFaceConfig, HuggingFaceUploader

config = HuggingFaceConfig(
    checkpoint_path="./gpt2-chess/checkpoint-1000",
    tokenizer_path="./gpt2-chess/tokenizer",
    repo_name="Zual/chess",
    model_name="gpt2-chess"
)

uploader = HuggingFaceUploader(config)
uploader.upload()