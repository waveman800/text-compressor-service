#!/usr/bin/env python3
"""
测试改进后的相似度去重逻辑
"""

import sys
sys.path.insert(0, '/home/ai/dev/text_compressor_service')

from core.tokenizer_wrapper import TokenizerWrapper
from core.dynamic_compressor import DynamicContextCompressor

tokenizer = TokenizerWrapper()
compressor = DynamicContextCompressor(session_len=4096, tokenizer=tokenizer)

def test_similarity_calculation():
    """测试相似度计算"""
    print("=" * 60)
    print("测试相似度计算")
    print("=" * 60)
    
    test_cases = [
        ("人工智能在当今社会发挥着越来越重要的作用。", 
         "人工智能在当今社会中发挥着越来越重要的作用。"),
        ("人工智能已经广泛应用于医疗、金融、教育等多个领域。",
         "人工智能已经被广泛应用于医疗、金融、教育等多个领域。"),
        ("这是一个完全不同的文本内容。",
         "这是另一个完全不同的文本内容。"),
        ("重复内容需要被识别和压缩。",
         "重复内容需要被识别和压缩。"),
    ]
    
    print("\n相似度测试结果:")
    for text1, text2 in test_cases:
        similarity = compressor._calculate_text_similarity(text1, text2)
        print(f"\n文本1: {text1}")
        print(f"文本2: {text2}")
        print(f"相似度: {similarity:.4f} ({'相似' if similarity >= 0.75 else '不相似'})")

def test_enhanced_deduplication():
    """测试增强版去重逻辑"""
    print("\n" + "=" * 60)
    print("测试增强版去重逻辑")
    print("=" * 60)
    
    test_text = """
段落一：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落二：人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。

段落三：人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。

段落一（重复）：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落四：人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。

段落二（变体）：人工智能已经被广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。

段落五：人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。

段落三（重复）：人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。

段落六：重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。

段落一（几乎相同）：人工智能在当今社会中发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落七：这是最后一段重要内容，应该被保留。
"""
    
    original_tokens = tokenizer.count_tokens(test_text)
    print(f"\n原始文本: {len(test_text)} 字符, {original_tokens} tokens")
    
    available_tokens = 150
    print(f"可用token: {available_tokens}")
    
    print("\n使用改进后的 _compress_narrative 方法:")
    compressed = compressor._compress_narrative(test_text, available_tokens)
    
    compressed_tokens = tokenizer.count_tokens(compressed)
    print(f"\n压缩后: {len(compressed)} 字符, {compressed_tokens} tokens")
    print(f"压缩比: {compressed_tokens / original_tokens * 100:.1f}%")
    
    print(f"\n压缩后内容:\n{compressed}")

def test_comparison():
    """比较新旧去重逻辑"""
    print("\n" + "=" * 60)
    print("比较新旧去重逻辑")
    print("=" * 60)
    
    test_text = """
段落一：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落二：人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。

段落一（几乎相同）：人工智能在当今社会中发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落三：这是最后一段重要内容，应该被保留。
"""
    
    segments = compressor.split_text(test_text)
    print(f"\n原始段落数: {len(segments)}")
    
    print("\n使用哈希去重（旧方法）:")
    seen_hashes = set()
    hash_unique = []
    for seg in segments:
        seg_hash = hash(seg.strip())
        if seg_hash not in seen_hashes:
            hash_unique.append(seg)
            seen_hashes.add(seg_hash)
    print(f"  去重后: {len(hash_unique)} 个段落")
    
    print("\n使用相似度去重（新方法）:")
    similarity_unique = compressor._deduplicate_with_similarity(segments, similarity_threshold=0.75)
    print(f"  去重后: {len(similarity_unique)} 个段落")
    
    improvement = len(segments) - len(similarity_unique)
    print(f"\n改进效果: 额外移除 {improvement} 个近似重复段落")

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    edge_cases = [
        ("", "这是一个测试文本。"),
        ("这是一个测试文本。", ""),
        ("AI人工智能ML机器学习DL深度学习NLP自然语言处理。", 
         "人工智能机器学习深度学习自然语言处理。"),
    ]
    
    print("\n边界情况测试:")
    for text1, text2 in edge_cases:
        similarity = compressor._calculate_text_similarity(text1, text2)
        print(f"\n文本1: '{text1}'")
        print(f"文本2: '{text2}'")
        print(f"相似度: {similarity:.4f}")

if __name__ == "__main__":
    test_similarity_calculation()
    test_enhanced_deduplication()
    test_comparison()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
