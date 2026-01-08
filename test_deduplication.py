#!/usr/bin/env python3
"""
针对性测试：验证去重逻辑是否正常工作
"""

import sys
sys.path.insert(0, '/home/ai/dev/text_compressor_service')

from core.tokenizer_wrapper import TokenizerWrapper
from core.dynamic_compressor import DynamicContextCompressor

tokenizer = TokenizerWrapper()
compressor = DynamicContextCompressor(session_len=4096, tokenizer=tokenizer)

def test_deduplication_logic():
    """测试去重逻辑"""
    print("=" * 60)
    print("测试去重逻辑")
    print("=" * 60)
    
    test_text = """
人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。
人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。
人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。
人工智能的伦理问题日益受到关注，如AI决策的透明度、责任归属、人机关系等。这些问题需要建立完善的法规体系。
人工智能技术正在快速发展。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。
人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。
人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。
人工智能的伦理问题日益受到关注，如AI决策的透明度、责任归属、人机关系等。这些问题需要建立完善的法规体系。
重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。
重复内容需要被识别和压缩。重复内容需要被识别和压缩。重复内容需要被识别和压缩。
人工智能技术正在快速发展。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。
人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。
人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。
人工智能的伦理问题日益受到关注，如AI决策的透明度、责任归属、人机关系等。这些问题需要建立完善的法规体系。
这是最后一段重要内容，应该被保留。这是最后一段重要内容，应该被保留。
"""
    
    original_tokens = tokenizer.count_tokens(test_text)
    print(f"\n原始文本长度: {len(test_text)} 字符")
    print(f"原始token数: {original_tokens}")
    
    available_tokens = 200
    print(f"\n可用token数: {available_tokens}")
    print(f"压缩比: {available_tokens / original_tokens * 100:.1f}%")
    
    print("\n" + "-" * 60)
    print("调用 _compress_narrative 方法...")
    print("-" * 60)
    
    compressed = compressor._compress_narrative(test_text, available_tokens)
    
    print("\n" + "-" * 60)
    print("压缩结果")
    print("-" * 60)
    
    compressed_tokens = tokenizer.count_tokens(compressed)
    print(f"压缩后长度: {len(compressed)} 字符")
    print(f"压缩后token数: {compressed_tokens}")
    print(f"实际压缩比: {compressed_tokens / original_tokens * 100:.1f}%")
    
    print(f"\n压缩后内容预览:\n{compressed[:500]}...")
    
    return compressed_tokens < original_tokens

def test_compression_decision():
    """测试压缩决策逻辑"""
    print("\n" + "=" * 60)
    print("测试压缩决策逻辑")
    print("=" * 60)
    
    short_text = "你好，请介绍一下你自己。"
    long_text = """
人工智能在当今社会发挥着越来越重要的作用。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。
人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。
人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。
人工智能的伦理问题日益受到关注，如AI决策的透明度、责任归属、人机关系等。这些问题需要建立完善的法规体系。
人工智能技术正在快速发展。人工智能的发展正在深刻改变我们的生活方式。
人工智能已经广泛应用于医疗、金融、教育等多个领域。人工智能为这些领域带来了革命性的变化。
人工智能技术包括机器学习、深度学习、自然语言处理等多个方向。这些技术的融合推动了AI的快速发展。
人工智能在医疗领域的应用包括辅助诊断、药物研发、医学影像分析等。这些应用显著提高了医疗效率。
人工智能在金融领域的应用包括风险评估、欺诈检测、智能投顾等。这些应用增强了金融系统的安全性。
人工智能在教育领域的应用包括个性化学习、智能辅导、知识图谱等。这些应用提升了教育的公平性和质量。
人工智能的发展也面临诸多挑战，如数据隐私、算法偏见、就业影响等。这些挑战需要社会各界共同应对。
人工智能的伦理问题日益受到关注，如AI决策的透明度、责任归属、人机关系等。这些问题需要建立完善的法规体系。
"""
    
    short_tokens = tokenizer.count_tokens(short_text)
    long_tokens = tokenizer.count_tokens(long_text)
    
    print(f"\n短文本token数: {short_tokens}")
    print(f"长文本token数: {long_tokens}")
    
    print("\n短文本压缩测试:")
    short_compressed, short_was_compressed = compressor.dynamic_compress(short_text, "", 50)
    print(f"  是否压缩: {short_was_compressed}")
    
    print("\n长文本压缩测试:")
    long_compressed, long_was_compressed = compressor.dynamic_compress(long_text, "", 100)
    print(f"  是否压缩: {long_was_compressed}")
    print(f"  压缩后token数: {tokenizer.count_tokens(long_compressed)}")

if __name__ == "__main__":
    success1 = test_deduplication_logic()
    test_compression_decision()
    
    print("\n" + "=" * 60)
    print("去重逻辑测试结果")
    print("=" * 60)
    if success1:
        print("✓ 去重和压缩逻辑正常工作")
    else:
        print("✗ 去重或压缩逻辑可能存在问题")
