#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态压缩功能
发送超过120000 token的请求，观察是否触发动态压缩
"""

import sys
import time
import requests
import json
import re
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper
from core.model_config import get_model_config


def generate_large_text(target_tokens):
    """生成指定token数量的测试文本"""
    sample_texts = [
        "人工智能是计算机科学的一个重要分支，它致力于研究如何让计算机系统能够像人类一样思考、学习和解决问题。这个领域涵盖了机器学习、深度学习、自然语言处理、计算机视觉等多个研究方向。",
        "深度学习是机器学习的一种特殊方法，它使用多层神经网络来学习数据的复杂模式和特征。近年来，深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展，推动了人工智能技术的快速发展。",
        "大语言模型是当前人工智能领域最热门的研究方向之一。这类模型通常基于Transformer架构，通过在海量文本数据上进行预训练来学习语言知识和世界知识。代表性模型包括GPT系列、Claude、Qwen等。",
        "文本压缩技术在大语言模型应用中扮演着重要角色。由于模型的上下文窗口有限，当输入文本过长时，需要对文本进行压缩以适应模型的限制，同时尽可能保留文本的核心信息和关键内容。",
        "随着人工智能技术的不断发展，我们正在见证一个全新的技术革命时代。智能助手、自动驾驶、医疗诊断、金融分析等众多领域都在积极探索如何利用人工智能技术来提升效率、改善生活质量。",
    ]
    
    tokenizer = TokenizerWrapper()
    text_parts = []
    current_tokens = 0
    
    while current_tokens < target_tokens:
        for text in sample_texts:
            text_parts.append(text * 5)
            current_tokens = tokenizer.count_tokens("".join(text_parts))
            if current_tokens >= target_tokens:
                break
    
    full_text = "".join(text_parts)
    actual_tokens = tokenizer.count_tokens(full_text)
    
    return full_text, actual_tokens


def test_dynamic_compression():
    """测试动态压缩功能"""
    
    print("=" * 70)
    print("动态压缩功能测试")
    print("=" * 70)
    
    target_tokens = 130000
    
    print(f"\n1. 生成测试文本...")
    test_text, token_count = generate_large_text(target_tokens)
    print(f"   生成文本token数量: {token_count:,}")
    
    tokenizer = TokenizerWrapper()
    
    print(f"\n2. 文本统计信息:")
    print(f"   字符数: {len(test_text):,}")
    print(f"   Token数: {token_count:,}")
    print(f"   模型上下文窗口: 120,000")
    print(f"   超出限制: {token_count - 120000:,} tokens")
    print(f"   安全边际(10%)后可用: {int(120000 * 0.9):,} tokens")
    
    print(f"\n3. 测试动态压缩...")
    
    compressor = DynamicContextCompressor(
        session_len=108000,
        tokenizer=tokenizer
    )
    
    original_tokens = token_count
    threshold_tokens = 108000
    
    print(f"   压缩阈值: {threshold_tokens:,} tokens")
    print(f"   原始token数: {original_tokens:,}")
    
    if original_tokens > threshold_tokens:
        print("   ✓ 需要压缩：原始文本超过阈值")
        
        start_time = time.time()
        compressed_text, was_compressed = compressor.dynamic_compress(
            text=test_text,
            current_prompt="请总结以下文本的核心内容",
            max_new_tokens=256
        )
        compression_time = time.time() - start_time
        
        compressed_tokens = tokenizer.count_tokens(compressed_text)
        
        print(f"\n4. 压缩结果:")
        print(f"   压缩耗时: {compression_time:.2f}秒")
        print(f"   压缩后token数: {compressed_tokens:,}")
        print(f"   压缩比: {compressed_tokens/original_tokens*100:.1f}%")
        print(f"   减少token: {original_tokens - compressed_tokens:,}")
        print(f"   是否进行了压缩: {was_compressed}")
        
        if compressed_tokens <= threshold_tokens:
            print(f"\n✓ 压缩成功！压缩后的文本符合模型限制")
        else:
            print(f"\n⚠ 压缩后仍超出限制: {compressed_tokens:,} > {threshold_tokens:,}")
        
        return compressed_text, was_compressed, original_tokens, compressed_tokens
    else:
        print("   不需要压缩：原始文本在阈值内")
        return test_text, False, original_tokens, original_tokens


def send_large_request(text, original_tokens, compressed_text, compressed_tokens):
    """发送请求到模型服务"""

    base_url = "http://10.216.7.43:30011/v1"
    api_key = "sk-pyQKSv0m5ONhK77R2a89832eAb914eF4B8D85882C2D06c5d"
    model_name = "Qwen3-14B-AWQ"

    model_config = get_model_config(model_name)
    if model_config is None:
        print(f"❌ 未找到模型配置: {model_name}")
        return

    context_window = model_config.context_length
    max_output_tokens = model_config.max_output_tokens
    safety_buffer = max_output_tokens + 500  # 额外增加500 token作为缓存
    
    print(f"\n" + "=" * 70)
    print("发送实际请求测试")
    print("=" * 70)
    
    print(f"\n场景: 发送 {original_tokens:,} tokens 的请求")
    print(f"模型: {model_name}")
    print(f"模型上下文窗口: {context_window:,} tokens (包含输入+输出)")
    print(f"预留生成token数: {safety_buffer:,} (来自模型配置max_output_tokens)")
    print(f"实际可用输入token数: {context_window - safety_buffer:,}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 使用模型的token计算方法来生成符合要求的请求文案
    # 先计算提示词的token数
    prompt_prefix = "请用一句话总结以下文本的核心内容（只需简单回复\"已收到\"即可）：\n\n"
    
    # 计算提示词前缀的token数
    tokenizer = TokenizerWrapper()
    prefix_tokens = tokenizer.count_tokens(prompt_prefix)
    
    # 计算可用的文本token数（上下文窗口 - 提示词前缀 - 预留生成token数）
    available_text_tokens = context_window - prefix_tokens - safety_buffer
    
    # 如果压缩后的文本超过可用token数，需要进一步压缩
    if compressed_tokens > available_text_tokens:
        print(f"\n⚠ 压缩后的文本仍超过可用token数: {compressed_tokens:,} > {available_text_tokens:,}")
        print(f"   正在进一步压缩...")
        
        # 创建压缩器
        compressor = DynamicContextCompressor()
        
        # 进一步压缩到可用token数
        further_compressed_text, was_further_compressed = compressor.dynamic_compress(
            text, 
            current_prompt="", 
            max_new_tokens=safety_buffer
        )
        
        further_compressed_tokens = tokenizer.count_tokens(further_compressed_text)
        print(f"   进一步压缩后token数: {further_compressed_tokens:,}")
        text = further_compressed_text
    
    # 最后检查总token数是否超过上下文窗口
    full_prompt = f"{prompt_prefix}{text}"
    full_prompt_tokens = tokenizer.count_tokens(full_prompt)
    total_estimated_tokens = full_prompt_tokens + safety_buffer
    
    if total_estimated_tokens > context_window:
        print(f"\n⚠ 总token数（输入+输出）仍超过上下文窗口: {total_estimated_tokens:,} > {context_window:,}")
        print(f"   正在调整max_tokens...")
        
        # 调整max_tokens以确保总token数不超过上下文窗口
        max_tokens = context_window - full_prompt_tokens
        print(f"   调整后的max_tokens: {max_tokens}")
    else:
        # 使用安全缓冲值作为默认的max_tokens
        max_tokens = min(safety_buffer, context_window - full_prompt_tokens)
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt_prefix}{text}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "extra_body": {
            "enable_thinking": False
        }
    }
    
    try:
        print(f"\n正在发送请求...")
        print(f"原始文本Token数: {original_tokens:,}")
        print(f"压缩后文本Token数: {compressed_tokens:,}")
        
        # 计算实际发送的完整提示词Token数
        full_prompt = f"请用一句话总结以下文本的核心内容（只需简单回复\"已收到\"即可）：\n\n{text}"
        tokenizer = TokenizerWrapper()
        full_prompt_tokens = tokenizer.count_tokens(full_prompt)
        print(f"完整请求Token数: {full_prompt_tokens:,}")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        request_time = time.time() - start_time
        
        print(f"请求耗时: {request_time:.2f}秒")
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ 请求成功！")
            
            message = result['choices'][0]['message']
            content = message.get('content', '')
            
            content_bytes = content.encode('latin-1', errors='ignore')
            content_utf8 = content_bytes.decode('utf-8', errors='ignore')
            content_clean = re.sub(r'<[^>]+>', '', content_utf8)
            content_clean = content_clean.strip()
            
            if len(content_clean) > 10:
                print(f"回复内容: {content_clean[:300]}")
            else:
                print(f"原始回复: {content[:100]}")
                print(f"\n提示: 原始内容较短，可能已被正确清理")
            
            usage = result.get('usage', {})
            print(f"\nToken使用统计:")
            print(f"  提示tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  生成tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  总tokens: {usage.get('total_tokens', 'N/A')}")
            
            return True
        else:
            print(f"\n✗ 请求失败")
            error_text = response.text[:300]
            print(f"错误: {error_text}")
            
            if "too long" in error_text.lower() or "context" in error_text.lower():
                print(f"\n这证实了超过120000 tokens会导致模型拒绝请求！")
                print(f"需要使用动态压缩功能来处理长文本。")
            return False
            
    except Exception as e:
        print(f"\n✗ 请求出错: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始动态压缩功能测试...\n")
    
    compressed_text, was_compressed, original_tokens, compressed_tokens = test_dynamic_compression()
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"原始文本token数: {original_tokens:,}")
    print(f"压缩后token数: {compressed_tokens:,}")
    print(f"是否触发动态压缩: {'是 ✓' if was_compressed else '否'}")
    print(f"压缩比: {compressed_tokens/original_tokens*100:.1f}%" if was_compressed else "N/A")
    
    send = input(f"\n是否发送实际请求到模型服务? [y/N]: ")
    
    if send.lower() in ['y', 'yes', '是']:
        send_large_request(compressed_text, original_tokens, compressed_text, compressed_tokens)
    
    print("\n" + "=" * 70)
    print("测试结束")
    print("=" * 70)
