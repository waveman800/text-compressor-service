"""
简单测试：透明代理模式演示
"""

import requests
import json

# 压缩服务地址
COMPRESSOR_URL = "http://localhost:8000"

print("="*60)
print("文本压缩服务 - 透明代理模式演示")
print("="*60)

# 1. 健康检查
print("\n1. 健康检查")
try:
    response = requests.get(f"{COMPRESSOR_URL}/health")
    if response.status_code == 200:
        print("✅ 压缩服务运行正常")
        print(f"   响应: {response.json()}")
    else:
        print(f"❌ 压缩服务异常: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ 无法连接到压缩服务: {e}")
    exit(1)

# 2. 直接压缩测试
print("\n2. 直接压缩测试")
long_text = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
人工智能可以对人的意识、思维的信息过程的模拟。
"""

try:
    response = requests.post(
        f"{COMPRESSOR_URL}/compress/text",
        json={
            "text": long_text,
            "current_prompt": "请总结这段文字",
            "max_new_tokens": 50,
            "session_len": 200
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 压缩成功")
        print(f"   原始tokens: {result['original_length']}")
        print(f"   压缩后tokens: {result['compressed_length']}")
        print(f"   压缩比: {result['compressed_length']/result['original_length']:.2%}")
    else:
        print(f"❌ 压缩失败: {response.status_code}")
except Exception as e:
    print(f"❌ 压缩请求失败: {e}")

# 3. 聊天历史压缩测试
print("\n3. 聊天历史压缩测试")
chat_history = [
    {"role": "system", "content": "你是一个专业的AI助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
    {"role": "user", "content": "请介绍一下Python。"},
]

try:
    response = requests.post(
        f"{COMPRESSOR_URL}/compress/chat",
        json={
            "chat_history": chat_history,
            "current_prompt": "请介绍一下Python。",
            "max_new_tokens": 256,
            "session_len": 500
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 聊天历史压缩成功")
        print(f"   原始tokens: {result['original_length']}")
        print(f"   压缩后tokens: {result['compressed_length']}")
        print(f"   压缩比: {result['compressed_length']/result['original_length']:.2%}")
    else:
        print(f"❌ 聊天历史压缩失败: {response.status_code}")
except Exception as e:
    print(f"❌ 聊天历史压缩请求失败: {e}")

# 4. 透明代理模式说明
print("\n4. 透明代理模式说明")
print("="*60)
print("🎯 核心优势：客户端零感知")
print("="*60)
print()
print("❌ 方案1：手动两步调用（复杂）")
print("   第1步：调用压缩服务")
print("   第2步：手动调用大模型服务")
print("   问题：需要客户端处理两次请求，代码复杂")
print()
print("❌ 方案2：请求报文加标识（侵入）")
print("   在请求中添加 compress: true")
print("   问题：需要修改大模型服务的代码")
print()
print("✅ 方案3：透明代理模式（推荐）")
print("   客户端只需像调用普通OpenAI API一样")
print("   压缩服务自动处理压缩和转发")
print("   对客户端完全透明！")
print()
print("="*60)
print("使用透明代理模式")
print("="*60)
print()
print("步骤1：配置 .env 文件")
print("  OPENAI_API_BASE_URL=http://your-model-service/v1")
print("  OPENAI_API_KEY=your-api-key")
print()
print("步骤2：客户端代码（完全不需要修改）")
print("""
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",  # 压缩服务
    json={
        "model": "gpt-3.5-turbo",
        "messages": chat_history,
        "max_tokens": 256
    }
)

result = response.json()
print(result['choices'][0]['message']['content'])
""")
print()
print("🎯 关键点：只需要改变URL，其他代码完全相同！")
print()
print("="*60)
print("内部处理流程（对客户端透明）")
print("="*60)
print("1. 压缩服务接收请求")
print("2. 分析聊天历史的token数量")
print("3. 如果超过上下文窗口，自动压缩")
print("4. 构建新的请求（使用压缩后的聊天历史）")
print("5. 转发到真实模型服务")
print("6. 返回模型响应给客户端")
print()
print("✅ 客户端完全不知道压缩的存在！")
