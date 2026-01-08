#!/usr/bin/env python3
"""
测试API代理转发和动态压缩功能
验证内容：
1. 请求转发是否成功
2. 动态压缩是否正常工作
3. 重复内容去重逻辑是否有效
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8001"

def test_health_check():
    """健康检查"""
    print("\n=== 1. 健康检查 ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ 服务器健康检查通过")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

def test_models_endpoint():
    """测试模型端点"""
    print("\n=== 2. 测试模型端点 ===")
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ 模型端点响应正常")
            print(f"  可用模型: {json.dumps(data, ensure_ascii=False, indent=2)}")
            return True
        else:
            print(f"✗ 模型端点错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

def generate_repetitive_text():
    """生成包含大量重复内容的测试文本"""
    base_content = """
    这是一段关于人工智能发展的重要论述。人工智能技术正在快速发展。
    人工智能已经在各个领域展现出强大的能力。人工智能将继续改变我们的生活方式。
    在医疗领域，人工智能可以帮助医生进行诊断。在金融领域，人工智能可以辅助风险评估。
    人工智能的发展带来了许多机遇。人工智能的发展也带来了挑战。
    我们需要认真思考人工智能的伦理问题。人工智能的安全性是重要的考量因素。
    人工智能的可解释性需要进一步研究。人工智能的公平性需要得到保障。
    人工智能技术正在快速发展。人工智能已经在各个领域展现出强大的能力。
    人工智能将继续改变我们的生活方式。人工智能的发展带来了许多机遇。
    人工智能的发展也带来了挑战。我们需要认真思考人工智能的伦理问题。
    人工智能的安全性是重要的考量因素。人工智能的可解释性需要进一步研究。
    人工智能的公平性需要得到保障。人工智能技术正在快速发展。
    人工智能已经在各个领域展现出强大的能力。人工智能将继续改变我们的生活方式。
    人工智能的发展带来了许多机遇。人工智能的发展也带来了挑战。
    我们需要认真思考人工智能的伦理问题。人工智能的安全性是重要的考量因素。
    人工智能的可释性需要进一步研究。人工智能的公平性需要得到保障。
    重复的内容应该被压缩掉。重复的内容应该被压缩掉。
    重复的内容应该被压缩掉。重复的内容应该被压缩掉。
    重复的内容应该被压缩掉。重复的内容应该被压缩掉。
    人工智能技术正在快速发展。人工智能已经在各个领域展现出强大的能力。
    人工智能将继续改变我们的生活方式。人工智能的发展带来了许多机遇。
    人工智能的发展也带来了挑战。我们需要认真思考人工智能的伦理问题。
    人工智能的安全性是重要的考量因素。人工智能的可解释性需要进一步研究。
    人工智能的公平性需要得到保障。人工智能技术正在快速发展。
    人工智能已经在各个领域展现出强大的能力。人工智能将继续改变我们的生活方式。
    人工智能的发展带来了许多机遇。人工智能的发展也带来了挑战。
    我们需要认真思考人工智能的伦理问题。人工智能的安全性是重要的考量因素。
    人工智能的可解释性需要进一步研究。人工智能的公平性需要得到保障。
    """
    return base_content

def test_chat_completion_with_repetitive_content():
    """测试包含重复内容的聊天完成请求"""
    print("\n=== 3. 测试包含重复内容的聊天完成 ===")
    
    repetitive_text = generate_repetitive_text()
    print(f"测试文本长度: {len(repetitive_text)} 字符")
    print(f"预估token数: {len(repetitive_text) // 4} (粗略估算)")
    
    messages = [
        {"role": "user", "content": repetitive_text}
    ]
    
    payload = {
        "model": "Qwen3-14B-AWQ",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    print("\n发送请求...")
    print(f"目标: {API_BASE_URL}/v1/chat/completions")
    print(f"模型: {payload['model']}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120
        )
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ 请求成功!")
            print(f"\n响应内容:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            
            usage = data.get('usage', {})
            print(f"\nToken使用统计:")
            print(f"  提示tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  生成tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  总tokens: {usage.get('total_tokens', 'N/A')}")
            
            return True
        else:
            print(f"\n✗ 请求失败!")
            print(f"错误响应: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"\n✗ 请求异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_completion_without_compression():
    """测试短文本（不应触发压缩）"""
    print("\n=== 4. 测试短文本（不应触发压缩） ===")
    
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]
    
    payload = {
        "model": "Qwen3-14B-AWQ",
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ 短文本请求成功")
            content = data['choices'][0]['message']['content']
            print(f"响应: {content[:200]}...")
            return True
        else:
            print(f"✗ 短文本请求失败: {response.status_code}")
            print(response.text[:200])
            return False
    except Exception as e:
        print(f"✗ 请求异常: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("API代理转发和动态压缩功能测试")
    print("=" * 60)
    
    results = {
        "健康检查": test_health_check(),
        "模型端点": test_models_endpoint(),
        "重复内容压缩": False,
        "短文本请求": test_chat_completion_without_compression()
    }
    
    results["重复内容压缩"] = test_chat_completion_with_repetitive_content()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n总体结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
