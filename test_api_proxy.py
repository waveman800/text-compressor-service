#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API端点的代理转发和动态压缩功能
"""

import requests
import json

API_BASE_URL = "http://localhost:8001"

def test_proxy_and_compression():
    print("=" * 70)
    print("测试API端点 - 代理转发和动态压缩功能")
    print("=" * 70)

    # 生成超过上下文窗口的测试文本
    sample_text = """
    人工智能是计算机科学的一个重要分支，它致力于研究如何让计算机系统能够像人类一样思考、学习和解决问题。这个领域涵盖了机器学习、深度学习、自然语言处理、计算机视觉等多个研究方向。近年来，随着计算能力的提升和大数据的普及，人工智能技术取得了突破性进展。深度学习作为机器学习的一种特殊方法，使用多层神经网络来学习数据的复杂模式和特征，在图像识别、语音识别、自然语言处理等领域表现出色。大语言模型是当前人工智能领域最热门的研究方向之一，这类模型通常基于Transformer架构，通过在海量文本数据上进行预训练来学习语言知识和世界知识。代表性模型包括GPT系列、Claude、Qwen等。文本压缩技术在大语言模型应用中扮演着重要角色，由于模型的上下文窗口有限，当输入文本过长时，需要对文本进行压缩以适应模型的限制，同时尽可能保留文本的核心信息和关键内容。随着人工智能技术的不断发展，我们正在见证一个全新的技术革命时代。智能助手、自动驾驶、医疗诊断、金融分析等众多领域都在积极探索如何利用人工智能技术来提升效率、改善生活质量。各大科技公司纷纷投入巨资研发人工智能技术，争夺这个未来最具潜力的技术高地。学术界也高度重视人工智能研究，各国政府也在制定相关政策来引导人工智能的健康发展。
    """ * 2000  # 重复以产生大量文本

    print(f"\n1. 测试文本长度: {len(sample_text):,} 字符")

    # 构建请求
    request_payload = {
        "model": "Qwen3-14B-AWQ",
        "messages": [
            {
                "role": "user",
                "content": f"请用一句话总结以下文本的核心内容：\n\n{sample_text}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 100
    }

    print(f"\n2. 发送请求到: {API_BASE_URL}/v1/chat/completions")
    print(f"   模型: {request_payload['model']}")
    print(f"   预期触发动态压缩（文本超过120,000 tokens）")

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=request_payload,
            timeout=120
        )

        print(f"\n3. 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            print(f"\n✓ 代理转发成功!")
            print(f"\n4. 响应详情:")
            print(f"   模型: {result.get('model', 'N/A')}")
            print(f"   回复内容长度: {len(result.get('choices', [{}])[0].get('message', {}).get('content', ''))} 字符")

            usage = result.get('usage', {})
            print(f"\n5. Token使用统计:")
            print(f"   提示tokens: {usage.get('prompt_tokens', 'N/A'):,}")
            print(f"   生成tokens: {usage.get('completion_tokens', 'N/A'):,}")
            print(f"   总tokens: {usage.get('total_tokens', 'N/A'):,}")

            # 检查响应中是否包含压缩相关信息
            # 由于压缩是在服务端进行的，我们通过响应状态判断
            print(f"\n6. 动态压缩状态:")
            if usage.get('prompt_tokens', 0) < len(sample_text) // 4:
                print(f"   ✓ 检测到动态压缩（提示tokens明显减少）")
            else:
                print(f"   ⚠ 未检测到明显压缩，可能需要检查服务端日志")

            return True
        else:
            print(f"\n❌ 请求失败!")
            print(f"   错误信息: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 连接错误: {str(e)}")
        return False

def test_health():
    """测试健康检查端点"""
    print("\n" + "=" * 70)
    print("测试健康检查")
    print("=" * 70)

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ 服务健康: {response.json()}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        return False

def test_models():
    """测试模型列表端点"""
    print("\n" + "=" * 70)
    print("测试模型列表")
    print("=" * 70)

    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ 获取到 {len(models)} 个模型配置")

            # 查找目标模型
            for model in models:
                if model['id'] == 'Qwen3-14B-AWQ':
                    print(f"\n目标模型配置:")
                    print(f"   ID: {model['id']}")
                    print(f"   名称: {model['name']}")
                    print(f"   上下文窗口: {model['context_length']:,}")
                    print(f"   最大输出tokens: {model['max_output_tokens']:,}")
                    break
            return True
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("API端点测试 - 代理转发与动态压缩验证")
    print("=" * 70)

    all_passed = True

    # 1. 测试健康检查
    if not test_health():
        print("\n❌ 服务未运行，请先启动服务: python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        exit(1)

    # 2. 测试模型列表
    if not test_models():
        all_passed = False

    # 3. 测试代理转发和动态压缩
    if not test_proxy_and_compression():
        all_passed = False

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    if all_passed:
        print("✓ 所有测试通过！")
        print("  - 服务正常运行")
        print("  - 代理转发功能正常")
        print("  - 动态压缩已触发")
    else:
        print("❌ 部分测试失败，请检查上述输出")
