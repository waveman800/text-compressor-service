class TokenizerWrapper:
    def __init__(self, model_name: str = "gpt2"):
        self._tokenizer = None
        self.use_huggingface = False
        
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.use_huggingface = True
        except ImportError:
            print("Warning: transformers library not found. Using character-based tokenization as fallback.")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {model_name}. Using character-based tokenization as fallback: {e}")

    def encode(self, text: str) -> list:
        if self.use_huggingface and self._tokenizer:
            return self._tokenizer.encode(text)
        else:
            return list(text)

    def decode(self, tokens: list) -> str:
        if self.use_huggingface and self._tokenizer:
            return self._tokenizer.decode(tokens)
        else:
            return ''.join(tokens)

    def count_tokens(self, text: str) -> int:
        if self.use_huggingface and self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            return len(text)
