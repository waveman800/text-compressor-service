"""
文本压缩服务使用示例

演示如何使用支持多LLM提供商的文本压缩服务
"""

import requests
import json


def list_models():
    """列出所有可用的模型"""
    response = requests.get("http://localhost:8000/models")
    models = response.json()
    
    print("=" * 70)
    print("可用的模型列表")
    print("=" * 70)
    for model in models["models"]:
        print(f"  - {model['id']}: {model['name']}")
        print(f"    上下文长度: {model['context_length']}")
        print(f"    服务提供商: {model['provider']}")
        print()


def list_providers():
    """列出所有配置的服务提供商"""
    response = requests.get("http://localhost:8000/providers")
    providers = response.json()
    
    print("=" * 70)
    print("配置的服务提供商")
    print("=" * 70)
    for provider in providers["providers"]:
        print(f"  - {provider['id']}: {provider['name']}")
        print(f"    API地址: {provider['api_base_url']}")
        print(f"    已配置密钥: {'是' if provider['has_api_key'] else '否'}")
        print()


def compress_text_with_model(model_name, text):
    """使用指定模型压缩文本"""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一个文本压缩助手。请将用户提供的文本进行压缩，保留核心信息。"
            },
            {
                "role": "user",
                "content": f"请压缩这段文本：{text}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
        return None


def example_openai():
    """使用OpenAI模型压缩文本"""
    print("=" * 70)
    print("示例1: 使用OpenAI GPT-3.5 Turbo")
    print("=" * 70)
    
    text = "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
    
    compressed = compress_text_with_model("gpt-3.5-turbo", text)
    
    if compressed:
        print(f"原始文本: {text}")
        print()
        print(f"压缩结果: {compressed}")
        print()


def example_qwen():
    """使用通义千问模型压缩文本"""
    print("=" * 70)
    print("示例2: 使用通义千问模型")
    print("=" * 70)
    
    text = "机器学习是人工智能的核心，是使计算机具有智能的根本途径。机器学习通过算法使计算机能够从数据中学习，从而在没有明确编程的情况下做出决策或预测。"
    
    compressed = compress_text_with_model("qwen-72b-chat", text)
    
    if compressed:
        print(f"原始文本: {text}")
        print()
        print(f"压缩结果: {compressed}")
        print()


def example_long_chat():
    """测试长对话历史的压缩"""
    print("=" * 70)
    print("示例3: 长对话历史压缩")
    print("=" * 70)
    
    url = "http://localhost:8000/v1/chat/completions"
    
    long_text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。" * 50
    
    messages = [
        {"role": "system", "content": "你是一个文本压缩助手。"},
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": "这是对上述文本的压缩版本：AI是计算机科学分支，研究智能实质，制造智能机器。"},
        {"role": "user", "content": "请继续压缩：机器学习是人工智能的核心，是使计算机具有智能的根本途径。"}
    ]
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"响应: {result['choices'][0]['message']['content']}")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
    
    print()


def example_multiple_models():
    """使用多个模型压缩同一段文本"""
    print("=" * 70)
    print("示例4: 使用多个模型压缩同一段文本")
    print("=" * 70)
    
    text = "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理、语音识别等领域取得了显著成果。"
    
    models = ["gpt-3.5-turbo", "claude-3-haiku", "llama-3-8b"]
    
    for model in models:
        print(f"使用模型: {model}")
        compressed = compress_text_with_model(model, text)
        if compressed:
            print(f"压缩结果: {compressed}")
        print()


def example_with_openai_sdk():
    """使用OpenAI SDK调用服务"""
    print("=" * 70)
    print("示例5: 使用OpenAI SDK")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "请压缩这段文本：自然语言处理（NLP）是人工智能的重要分支，它研究如何让计算机理解、生成和处理人类语言。"
                }
            ]
        )
        
        print(f"压缩结果: {response.choices[0].message.content}")
        print()
        
    except ImportError:
        print("未安装OpenAI SDK，跳过此示例")
        print("安装命令: pip install openai")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("文本压缩服务使用示例")
    print("=" * 70 + "\n")
    
    list_models()
    list_providers()
    
    example_openai()
    example_qwen()
    example_long_chat()
    example_multiple_models()
    example_with_openai_sdk()
    
    print("=" * 70)
    print("所有示例运行完成")
    print("=" * 70)
