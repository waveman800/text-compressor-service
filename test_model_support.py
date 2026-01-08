"""
测试不同模型的压缩功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper
from core.model_config import get_model_config, get_all_models, calculate_session_len


def test_model_config():
    """测试模型配置"""
    print("=" * 70)
    print("测试1: 模型配置")
    print("=" * 70)
    
    models = get_all_models()
    print(f"支持的模型数量: {len(models)}")
    print()
    
    for model_id in ["gpt-3.5-turbo", "gpt-4", "claude-3-opus", "llama-3-8b"]:
        config = get_model_config(model_id)
        if config:
            print(f"{model_id}:")
            print(f"  名称: {config.name}")
            print(f"  上下文长度: {config.context_length}")
            print(f"  最大输出: {config.max_output_tokens}")
            print(f"  描述: {config.description}")
            print()


def test_session_len_calculation():
    """测试session_len计算"""
    print("=" * 70)
    print("测试2: session_len计算")
    print("=" * 70)
    
    test_cases = [
        ("gpt-3.5-turbo", 0.1),
        ("gpt-4", 0.15),
        ("claude-3-opus", 0.05),
        ("llama-3-8b", 0.1),
    ]
    
    for model_id, safety_margin in test_cases:
        config = get_model_config(model_id)
        session_len = calculate_session_len(model_id, safety_margin)
        
        print(f"{model_id}:")
        print(f"  原始上下文: {config.context_length}")
        print(f"  安全余量: {safety_margin:.0%}")
        print(f"  计算后session_len: {session_len}")
        print(f"  节省: {config.context_length - session_len} tokens")
        print()


def test_compression_with_different_models():
    """测试不同模型的压缩"""
    print("=" * 70)
    print("测试3: 不同模型的压缩")
    print("=" * 70)
    
    tokenizer = TokenizerWrapper()
    
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，
    但能像人那样思考、也可能超过人的智能。
    
    机器学习是人工智能的核心，是使计算机具有智能的根本途径。
    深度学习是机器学习领域中一种新的方法，它源于人工神经网络的研究。
    
    卷积神经网络（CNN）是一种专门用来处理具有类似网格结构数据的神经网络。
    循环神经网络（RNN）是一种用于处理序列数据的神经网络。
    Transformer是一种基于自注意力机制的神经网络架构。
    
    大语言模型（LLM）是指具有大量参数的语言模型，通常使用Transformer架构。
    GPT（Generative Pre-trained Transformer）是一种生成式预训练Transformer模型。
    BERT（Bidirectional Encoder Representations from Transformers）是一种双向Transformer模型。
    
    自然语言处理（NLP）是人工智能的一个重要分支，它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    计算机视觉是另一个重要分支，它研究如何让计算机"看"和理解图像和视频。
    """ * 10
    
    test_models = [
        ("gpt-3.5-turbo", 256),
        ("gpt-4", 512),
        ("claude-3-opus", 1024),
        ("llama-3-8b", 512),
    ]
    
    for model_id, max_new_tokens in test_models:
        config = get_model_config(model_id)
        session_len = calculate_session_len(model_id, safety_margin=0.1)
        
        compressor = DynamicContextCompressor(
            session_len=session_len,
            max_new_tokens=max_new_tokens,
            enable_dynamic_compression=True
        )
        
        prompt_tokens = tokenizer.count_tokens("请总结这段文字")
        text_tokens = tokenizer.count_tokens(long_text)
        total_tokens = prompt_tokens + text_tokens + max_new_tokens
        
        print(f"{model_id}:")
        print(f"  上下文长度: {config.context_length}")
        print(f"  session_len: {session_len}")
        print(f"  提示词tokens: {prompt_tokens}")
        print(f"  文本tokens: {text_tokens}")
        print(f"  max_new_tokens: {max_new_tokens}")
        print(f"  总tokens: {total_tokens}")
        
        if total_tokens > session_len:
            print(f"  状态: 需要压缩")
            compressed_text, was_compressed = compressor.dynamic_compress(
                text=long_text,
                current_prompt="请总结这段文字",
                max_new_tokens=max_new_tokens
            )
            compressed_tokens = tokenizer.count_tokens(compressed_text)
            compression_ratio = compressed_tokens / text_tokens
            
            print(f"  压缩后tokens: {compressed_tokens}")
            print(f"  压缩比: {compression_ratio:.2%}")
        else:
            print(f"  状态: 无需压缩")
        
        print()


def test_openai_compatible_api():
    """测试OpenAI兼容API"""
    print("=" * 70)
    print("测试4: OpenAI兼容API")
    print("=" * 70)
    
    import requests
    
    try:
        # 测试健康检查
        response = requests.get("http://localhost:8000/health")
        print(f"健康检查: {response.json()}")
        print()
        
        # 测试模型列表
        response = requests.get("http://localhost:8000/models")
        models = response.json()
        print(f"支持的模型数量: {len(models['models'])}")
        print()
        
        # 测试特定模型信息
        response = requests.get("http://localhost:8000/models/gpt-3.5-turbo")
        model_info = response.json()
        print(f"GPT-3.5 Turbo信息:")
        print(f"  ID: {model_info['id']}")
        print(f"  名称: {model_info['name']}")
        print(f"  上下文长度: {model_info['context_length']}")
        print(f"  最大输出: {model_info['max_output_tokens']}")
        print()
        
        # 测试文本压缩（使用model_name参数）
        long_text = "这是一个很长的文本..." * 500
        response = requests.post(
            "http://localhost:8000/compress/text",
            json={
                "text": long_text,
                "model_name": "gpt-3.5-turbo",
                "max_new_tokens": 256,
                "safety_margin": 0.1
            }
        )
        result = response.json()
        print(f"文本压缩结果:")
        print(f"  原始长度: {result['original_length']} tokens")
        print(f"  压缩后长度: {result['compressed_length']} tokens")
        print(f"  是否压缩: {result['was_compressed']}")
        print()
        
        # 测试聊天历史压缩
        chat_history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ] * 100
        
        response = requests.post(
            "http://localhost:8000/compress/chat",
            json={
                "chat_history": chat_history,
                "model_name": "gpt-4",
                "max_new_tokens": 512,
                "safety_margin": 0.15
            }
        )
        result = response.json()
        print(f"聊天历史压缩结果:")
        print(f"  原始长度: {result['original_length']} tokens")
        print(f"  压缩后长度: {result['compressed_length']} tokens")
        print(f"  是否压缩: {result['was_compressed']}")
        print()
        
    except requests.exceptions.ConnectionError:
        print("⚠️  服务未启动，跳过API测试")
        print("   请先运行: cd api && python3 main.py")
        print()


if __name__ == "__main__":
    test_model_config()
    test_session_len_calculation()
    test_compression_with_different_models()
    test_openai_compatible_api()
    
    print("=" * 70)
    print("所有测试完成！")
    print("=" * 70)
