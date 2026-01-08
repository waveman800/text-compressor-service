"""
透明代理模式使用示例

客户端只需像调用普通OpenAI API一样，压缩服务会自动处理：
1. 接收OpenAI格式的请求
2. 自动压缩聊天历史
3. 转发到真实的大模型服务
4. 返回模型响应

对客户端完全透明！
"""

import requests
import json

# 配置
COMPRESSOR_SERVICE_URL = "http://localhost:8000"
REAL_MODEL_SERVICE_URL = "http://localhost:8001"  # 假设这是真实的大模型服务

def example_transparent_proxy():
    """
    示例1：透明代理模式（推荐）
    
    客户端只需要调用压缩服务的/v1/chat/completions端点，
    压缩服务会自动：
    1. 接收请求
    2. 压缩聊天历史
    3. 转发到真实的大模型服务
    4. 返回响应
    
    客户端完全不知道压缩的存在！
    """
    
    # 构建一个很长的聊天历史（模拟上下文溢出场景）
    long_chat_history = [
        {"role": "system", "content": "你是一个专业的AI助手，擅长回答各种问题。"},
        {"role": "user", "content": "你好，请介绍一下人工智能的发展历史。"},
        {"role": "assistant", "content": "人工智能的发展可以追溯到20世纪50年代..."},
        {"role": "user", "content": "能详细说说机器学习的发展吗？"},
        {"role": "assistant", "content": "机器学习是人工智能的核心分支..."},
        # 假设这里有很多轮对话，导致上下文过长
        {"role": "user", "content": "现在请总结一下深度学习的主要应用领域。"},
    ]
    
    # 客户端只需要像调用普通OpenAI API一样
    # 压缩服务会自动处理压缩和转发
    response = requests.post(
        f"{COMPRESSOR_SERVICE_URL}/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": long_chat_history,
            "max_tokens": 256,
            "temperature": 0.7
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 代理模式调用成功！")
        print(f"模型响应: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ 调用失败: {response.status_code}")
        print(response.text)


def example_direct_compression():
    """
    示例2：直接使用压缩API（如果需要查看压缩结果）
    
    如果您想先看看压缩效果，可以直接调用压缩API
    """
    
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。
    """
    
    # 直接调用压缩API
    response = requests.post(
        f"{COMPRESSOR_SERVICE_URL}/compress/text",
        json={
            "text": long_text,
            "current_prompt": "请总结这段文字",
            "max_new_tokens": 50,
            "session_len": 200
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 压缩成功！")
        print(f"原始长度: {result['original_length']} tokens")
        print(f"压缩后长度: {result['compressed_length']} tokens")
        print(f"是否压缩: {result['was_compressed']}")
        print(f"压缩结果: {result['compressed_text']}")
    else:
        print(f"❌ 压缩失败: {response.status_code}")
        print(response.text)


def example_comparison():
    """
    示例3：对比直接调用模型服务和通过压缩服务调用
    """
    
    chat_history = [
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "请介绍一下Python编程语言。"},
    ]
    
    print("\n" + "="*60)
    print("方案对比")
    print("="*60)
    
    # 方案1：直接调用真实模型服务（可能上下文溢出）
    print("\n📌 方案1：直接调用真实模型服务")
    print("URL:", REAL_MODEL_SERVICE_URL)
    print("问题：如果聊天历史过长，可能超过模型的上下文窗口")
    
    # 方案2：通过压缩服务调用（自动处理上下文溢出）
    print("\n📌 方案2：通过压缩服务调用（推荐）")
    print("URL:", COMPRESSOR_SERVICE_URL)
    print("优势：自动压缩聊天历史，确保不超过上下文窗口")
    print("      客户端代码完全不变，只是URL不同")
    
    print("\n" + "="*60)
    print("客户端代码对比")
    print("="*60)
    
    print("\n❌ 直接调用真实模型服务：")
    print("""
response = requests.post(
    "http://localhost:8001/v1/chat/completions",  # 真实模型服务
    json={"model": "gpt-3.5-turbo", "messages": chat_history, ...}
)
""")
    
    print("\n✅ 通过压缩服务调用：")
    print("""
response = requests.post(
    "http://localhost:8000/v1/chat/completions",  # 压缩服务
    json={"model": "gpt-3.5-turbo", "messages": chat_history, ...}
)
""")
    
    print("\n🎯 结论：只需要改变URL，其他代码完全相同！")


if __name__ == "__main__":
    print("文本压缩服务 - 透明代理模式示例\n")
    
    # 示例1：透明代理模式
    print("\n" + "="*60)
    print("示例1：透明代理模式")
    print("="*60)
    example_transparent_proxy()
    
    # 示例2：直接使用压缩API
    print("\n" + "="*60)
    print("示例2：直接使用压缩API")
    print("="*60)
    example_direct_compression()
    
    # 示例3：对比
    print("\n" + "="*60)
    print("示例3：方案对比")
    print("="*60)
    example_comparison()
