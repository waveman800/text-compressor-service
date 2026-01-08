#!/usr/bin/env python3
"""
详细诊断：split_text 和去重逻辑
"""

import sys
sys.path.insert(0, '/home/ai/dev/text_compressor_service')

from core.tokenizer_wrapper import TokenizerWrapper
from core.dynamic_compressor import DynamicContextCompressor

tokenizer = TokenizerWrapper()
compressor = DynamicContextCompressor(session_len=4096, tokenizer=tokenizer)

def diagnose_split_text():
    """诊断 split_text 方法的行为"""
    print("=" * 60)
    print("诊断 split_text 方法")
    print("=" * 60)
    
    test_text = """
人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。

人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。

人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。

重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。
重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。
"""
    
    print(f"\n测试文本 ({len(test_text)} 字符):\n{test_text}")
    
    print("\n调用 split_text 方法:")
    segments = compressor.split_text(test_text)
    
    print(f"\n分割结果: {len(segments)} 个段落")
    for i, seg in enumerate(segments):
        print(f"\n--- 段落 {i+1} ({len(seg)} 字符, {tokenizer.count_tokens(seg)} tokens) ---")
        print(seg[:200] + "..." if len(seg) > 200 else seg)

def test_actual_deduplication():
    """测试实际去重逻辑（使用正确的多段落文本）"""
    print("\n" + "=" * 60)
    print("测试实际去重逻辑")
    print("=" * 60)
    
    test_text = """
段落一：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落二：人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。

段落三：人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。

段落一：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落四：人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。

段落二：人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。

段落五：人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。

段落三：人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。

段落六：重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。

段落一：人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。

段落七：这是最后一段重要内容，应该被保留。
"""
    
    original_tokens = tokenizer.count_tokens(test_text)
    print(f"\n原始文本: {len(test_text)} 字符, {original_tokens} tokens")
    
    segments = compressor.split_text(test_text)
    print(f"分割成 {len(segments)} 个段落")
    
    seen_hashes = set()
    unique_segments = []
    
    for i, seg in enumerate(segments):
        seg_hash = hash(seg.strip())
        status = "重复" if seg_hash in seen_hashes else "唯一"
        print(f"  [{status}] 段落 {i+1}: {seg[:50]}... (hash: {seg_hash})")
        
        if seg_hash not in seen_hashes:
            unique_segments.append(seg)
            seen_hashes.add(seg_hash)
    
    print(f"\n去重后: {len(unique_segments)} 个唯一段落")
    print(f"移除: {len(segments) - len(unique_segments)} 个重复段落")

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
    
    print("\n使用 _compress_narrative 方法:")
    compressed = compressor._compress_narrative(test_text, available_tokens)
    
    compressed_tokens = tokenizer.count_tokens(compressed)
    print(f"\n压缩后: {len(compressed)} 字符, {compressed_tokens} tokens")
    print(f"压缩比: {compressed_tokens / original_tokens * 100:.1f}%")
    
    print(f"\n压缩后内容:\n{compressed}")

if __name__ == "__main__":
    diagnose_split_text()
    test_actual_deduplication()
    test_enhanced_deduplication()
