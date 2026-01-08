#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试与局域网大模型服务的连接
测试目标: Qwen3-14B-AWQ
服务地址: http://10.216.7.43:30011/v1/chat/completions
"""

import json
import sys
import requests


def test_connection():
    """测试与局域网大模型服务的连接"""
    
    base_url = "http://10.216.7.43:30011/v1"
    api_key = "sk-pyQKSv0m5ONhK77R2a89832eAb914eF4B8D85882C2D06c5d"
    model_name = "Qwen3-14B-AWQ"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "你好，请简单介绍一下你自己。"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("=" * 60)
    print("局域网大模型服务连接测试")
    print("=" * 60)
    print(f"服务地址: {base_url}")
    print(f"模型名称: {model_name}")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 连接成功！")
            print("\n响应内容:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                print("\n模型回复:")
                print(f"角色: {message['role']}")
                print(f"内容: {message['content']}")
            
            print("\n" + "=" * 60)
            print("测试通过！服务可正常使用。")
            print("=" * 60)
            return True
            
        else:
            print("✗ 连接失败")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ 请求超时 (30秒)")
        print("可能的原因:")
        print("  - 服务地址不可达")
        print("  - 网络连接问题")
        print("  - 服务响应过慢")
        return False
        
    except requests.exceptions.ConnectionError as e:
        print("✗ 连接错误")
        print(f"错误详情: {str(e)}")
        print("可能的原因:")
        print("  - IP地址不正确或无法访问")
        print("  - 端口号错误")
        print("  - 防火墙阻止连接")
        print("  - 服务未启动")
        return False
        
    except Exception as e:
        print(f"✗ 发生未知错误: {str(e)}")
        return False


def test_service_info():
    """测试获取服务信息"""
    
    base_url = "http://10.216.7.43:30011/v1"
    
    print("\n" + "=" * 60)
    print("测试服务健康状态")
    print("=" * 60)
    
    try:
        response = requests.get(
            f"{base_url}/models",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 服务健康检查通过")
            print("\n可用的模型列表:")
            if "data" in result:
                for model in result["data"]:
                    print(f"  - {model.get('id', 'Unknown')}")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 健康检查出错: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始测试局域网大模型服务连接...\n")
    
    success = test_connection()
    
    if success:
        test_service_info()
    
    print("\n测试完成")
    sys.exit(0 if success else 1)
