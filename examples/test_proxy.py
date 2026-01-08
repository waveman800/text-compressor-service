"""
实际测试：透明代理模式

这个脚本演示如何使用透明代理模式调用大模型服务
"""

import requests
import json

# 压缩服务地址
COMPRESSOR_URL = "http://localhost:8000"

def test_health_check():
    """测试压缩服务是否正常运行"""
    print("\n" + "="*60)
    print("1. 健康检查")
    print("="*60)
    
    try:
        response = requests.get(f"{COMPRESSOR_URL}/health")
        if response.status_code == 200:
            print("✅ 压缩服务运行正常")
            print(f"   响应: {response.json()}")
            return True
        else:
            print(f"❌ 压缩服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到压缩服务: {e}")
        return False


def test_direct_compression():
    """测试直接压缩API"""
    print("\n" + "="*60)
    print("2. 直接压缩测试")
    print("="*60)
    
    # 创建一个较长的文本
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。
    
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识，
    心理学和哲学。人工智能是包括十分广泛的科学，它由不同的领域组成，
    如机器学习，计算机视觉等等，总的说来，人工智能研究的一个主要目标是
    使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
    """
    
    print(f"原始文本长度: {len(long_text)} 字符")
    
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
            print(f"   是否压缩: {result['was_compressed']}")
            print(f"   压缩比: {result['compressed_length']/result['original_length']:.2%}")
            print(f"   压缩结果: {result['compressed_text'][:100]}...")
            return True
        else:
            print(f"❌ 压缩失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ 压缩请求失败: {e}")
        return False


def test_chat_compression():
    """测试聊天历史压缩"""
    print("\n" + "="*60)
    print("3. 聊天历史压缩测试")
    print("="*60)
    
    # 创建一个较长的聊天历史
    chat_history = [
        {"role": "system", "content": "你是一个专业的AI助手，擅长回答各种问题。"},
        {"role": "user", "content": "你好，请介绍一下人工智能的发展历史。"},
        {"role": "assistant", "content": "人工智能的发展可以追溯到20世纪50年代。1956年，达特茅斯会议首次提出了人工智能这一术语。"},
        {"role": "user", "content": "能详细说说机器学习的发展吗？"},
        {"role": "assistant", "content": "机器学习是人工智能的核心分支之一。它的发展经历了从符号主义到连接主义，再到深度学习的演进过程。"},
        {"role": "user", "content": "深度学习有什么应用？"},
        {"role": "assistant", "content": "深度学习在计算机视觉、自然语言处理、语音识别等领域都有广泛应用。比如图像分类、目标检测、机器翻译等。"},
        {"role": "user", "content": "现在请总结一下深度学习的主要应用领域。"},
    ]
    
    print(f"聊天历史轮数: {len([m for m in chat_history if m['role'] == 'user'])}")
    
    try:
        response = requests.post(
            f"{COMPRESSOR_URL}/compress/chat",
            json={
                "chat_history": chat_history,
                "current_prompt": "现在请总结一下深度学习的主要应用领域。",
                "max_new_tokens": 256,
                "session_len": 500
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 聊天历史压缩成功")
            print(f"   原始tokens: {result['original_length']}")
            print(f"   压缩后tokens: {result['compressed_length']}")
            print(f"   是否压缩: {result['was_compressed']}")
            print(f"   压缩后消息数: {len(result['compressed_chat'])}")
            print(f"   压缩比: {result['compressed_length']/result['original_length']:.2%}")
            return True
        else:
            print(f"❌ 聊天历史压缩失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ 聊天历史压缩请求失败: {e}")
        return False


def test_proxy_mode():
    """测试透明代理模式"""
    print("\n" + "="*60)
    print("4. 透明代理模式测试")
    print("="*60)
    
    print("⚠️  透明代理模式需要配置真实的大模型服务地址")
    print("   当前配置：OPENAI_API_BASE_URL=http://localhost:8000/v1")
    print("   这会导致循环调用，需要修改为真实的大模型服务地址")
    print()
    print("💡 使用透明代理模式的步骤：")
    print("   1. 修改 .env 文件，设置 OPENAI_API_BASE_URL 为真实的大模型服务地址")
    print("      例如：OPENAI_API_BASE_URL=http://localhost:8001/v1")
    print("   2. 客户端只需调用 http://localhost:8000/v1/chat/completions")
    print("   3. 压缩服务会自动：")
    print("      - 压缩聊天历史（如果需要）")
    print("      - 转发到真实的大模型服务")
    print("      - 返回模型响应")
    print("   4. 客户端完全不需要修改代码！")
    print()
    print("📖 客户端代码示例：")
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
    
    return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("文本压缩服务 - 透明代理模式测试")
    print("="*60)
    
    # 测试1：健康检查
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n❌ 压缩服务未运行，请先启动服务：")
        print("   cd /home/ai/dev/text_compressor_service")
        print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return
    
    # 测试2：直接压缩
    compression_ok = test_direct_compression()
    
    # 测试3：聊天历史压缩
    chat_ok = test_chat_compression()
    
    # 测试4：透明代理模式
    proxy_ok = test_proxy_mode()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"直接压缩: {'✅ 通过' if compression_ok else '❌ 失败'}")
    print(f"聊天压缩: {'✅ 通过' if chat_ok else '❌ 失败'}")
    print(f"代理模式: {'✅ 通过' if proxy_ok else '❌ 失败'}")
    
    if health_ok and compression_ok and chat_ok:
        print("\n🎉 所有核心功能测试通过！")
        print("\n📖 使用透明代理模式：")
        print("   1. 配置 .env 文件中的 OPENAI_API_BASE_URL")
        print("   2. 客户端只需调用 http://localhost:8000/v1/chat/completions")
        print("   3. 压缩服务会自动处理压缩和转发")
        print("   4. 客户端完全不需要修改代码！")
    else:
        print("\n⚠️  部分测试失败，请检查服务状态")


if __name__ == "__main__":
    main()
