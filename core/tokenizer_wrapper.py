import json
from typing import List, Union

class TokenizerWrapper:
    """
    分词器包装器，支持Hugging Face的AutoTokenizer和字符级别的回退
    """
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
            print(f"Warning: Failed to load tokenizer {model_name}. Using character-based fallback: {e}")
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token ID列表
        """
        if self.use_huggingface:
            return self._tokenizer.encode(text)
        else:
            # 字符级别的回退
            return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """
        将token ID列表解码为文本
        """
        if self.use_huggingface:
            return self._tokenizer.decode(tokens)
        else:
            # 字符级别的回退
            return ''.join([chr(t) for t in tokens])
    
    def count_tokens(self, text_or_obj: Union[str, dict, list]) -> int:
        """
        计算文本、字典或列表的token数量
        """
        if isinstance(text_or_obj, str):
            return len(self.encode(text_or_obj))
        elif isinstance(text_or_obj, dict):
            return sum(self.count_tokens(value) for value in text_or_obj.values())
        elif isinstance(text_or_obj, list):
            return sum(self.count_tokens(item) for item in text_or_obj)
        else:
            return len(self.encode(str(text_or_obj)))
