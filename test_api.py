import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.main import app, tokenizer, compressor
from core.model_config import get_model_config

print("Testing API imports...")
print(f"Tokenizer initialized: {tokenizer is not None}")
print(f"Compressor initialized: {compressor is not None}")
print(f"App initialized: {app is not None}")

# Test model config
model_config = get_model_config("gpt-3.5-turbo")
print(f"Model config: {model_config}")
if model_config:
    print(f"Max output tokens: {model_config.max_output_tokens}")
    print(f"Context length: {model_config.context_length}")

print("Test completed successfully!")