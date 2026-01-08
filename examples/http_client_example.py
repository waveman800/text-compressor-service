import requests
import json

BASE_URL = "http://localhost:8000"

# 测试文本
long_text = """
这是一段很长的测试文本。这段文本包含了多个句子和段落，用于测试动态压缩功能。

在这段文本中，我们会包含一些关键词，例如结论、重要、关键等，以测试重要性评分功能。

结论：动态文本压缩可以有效地减少文本长度，同时保留重要信息。这对于处理长文本和聊天历史非常有用。
"""

# 测试聊天历史
chat_history = [
    {"role": "user", "content": "你好，我想了解一下文本压缩服务的功能。"},
    {"role": "assistant", "content": "您好！文本压缩服务可以动态压缩文本和聊天历史，保留重要信息，同时减少token数量。"},
    {"role": "user", "content": "它支持哪些压缩模式？"},
    {"role": "assistant", "content": "它支持快速压缩和标准压缩两种模式。快速压缩更高效，标准压缩更精确。"},
    {"role": "user", "content": "压缩后的文本质量如何？"},
    {"role": "assistant", "content": "压缩后的文本会保留关键词和重要信息，例如结论、重要、关键等。这是通过重要性评分算法实现的。"},
    {"role": "user", "content": "非常感谢您的回答！"},
    {"role": "assistant", "content": "不客气！如果您有任何其他问题，随时可以问我。"}
]

def test_text_compression():
    """测试文本压缩API"""
    print("\n=== 测试文本压缩API ===")
    
    url = f"{BASE_URL}/compress/text"
    payload = {
        "text": long_text,
        "session_len": 100,
        "use_fast_compression": False
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"原始长度: {result['original_length']} tokens")
        print(f"压缩后长度: {result['compressed_length']} tokens")
        print(f"是否压缩: {result['was_compressed']}")
        print(f"压缩后的文本: {result['compressed_text']}")
    else:
        print(f"请求失败: {response.status_code} - {response.text}")

def test_chat_compression():
    """测试聊天历史压缩API"""
    print("\n=== 测试聊天历史压缩API ===")
    
    url = f"{BASE_URL}/compress/chat"
    payload = {
        "chat_history": chat_history,
        "session_len": 150
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"原始长度: {result['original_length']} tokens")
        print(f"压缩后长度: {result['compressed_length']} tokens")
        print(f"是否压缩: {result['was_compressed']}")
        print("压缩后的聊天历史:")
        for msg in result['compressed_chat']:
            print(f"{msg['role']}: {msg['content']}")
    else:
        print(f"请求失败: {response.status_code} - {response.text}")

def test_openai_protocol():
    """测试OpenAI协议兼容API"""
    print("\n=== 测试OpenAI协议兼容API ===")
    
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": chat_history,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("OpenAI API响应:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"请求失败: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"连接失败: {e}")
        print("注意: 这可能是因为目标OpenAI服务未运行。请确保配置了正确的OPENAI_API_BASE_URL。")

def test_health_check():
    """测试健康检查API"""
    print("\n=== 测试健康检查API ===")
    
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"服务状态: {result['status']}")
        print(f"服务名称: {result['service']}")
    else:
        print(f"请求失败: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("文本压缩服务HTTP客户端示例")
    print("=" * 50)
    
    test_health_check()
    test_text_compression()
    test_chat_compression()
    test_openai_protocol()
