import pytest
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper

# 初始化测试资源
tokenizer = TokenizerWrapper()
compressor = DynamicContextCompressor(session_len=100, tokenizer=tokenizer)

# 测试文本
long_text = """
这是一段很长的测试文本。这段文本包含了多个句子和段落，用于测试动态压缩功能。

在这段文本中，我们会包含一些关键词，例如结论、重要、关键等，以测试重要性评分功能。

结论：动态文本压缩可以有效地减少文本长度，同时保留重要信息。这对于处理长文本和聊天历史非常有用。
"""

# 测试聊天历史
chat_history = [
    {"role": "user", "content": "你好，我想了解一下文本压缩服务的功能。"},
    {"role": "assistant", "content": "您好！文本压缩服务可以动态压缩文本和聊天历史，保留重要信息，同时减少token数量。"},
    {"role": "user", "content": "它支持哪些压缩模式？"},
    {"role": "assistant", "content": "它支持快速压缩和标准压缩两种模式。快速压缩更高效，标准压缩更精确。"},
    {"role": "user", "content": "压缩后的文本质量如何？"},
    {"role": "assistant", "content": "压缩后的文本会保留关键词和重要信息，例如结论、重要、关键等。这是通过重要性评分算法实现的。"},
    {"role": "user", "content": "非常感谢您的回答！"},
    {"role": "assistant", "content": "不客气！如果您有任何其他问题，随时可以问我。"}
]

class TestTokenizerWrapper:
    """测试分词器包装器"""
    
    def test_encode_decode(self):
        """测试编码和解码功能"""
        text = "测试文本"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_count_tokens(self):
        """测试token计数功能"""
        text = "测试文本"
        tokens = tokenizer.encode(text)
        count = tokenizer.count_tokens(text)
        assert count == len(tokens)
    
    def test_count_tokens_dict(self):
        """测试字典的token计数功能"""
        data = {"key1": "value1", "key2": "value2"}
        count = tokenizer.count_tokens(data)
        assert count > 0
    
    def test_count_tokens_list(self):
        """测试列表的token计数功能"""
        data = ["item1", "item2", "item3"]
        count = tokenizer.count_tokens(data)
        assert count > 0

class TestDynamicContextCompressor:
    """测试动态文本压缩器"""
    
    def test_split_text_paragraph(self):
        """测试按段落分割文本"""
        compressor.segment_strategy = "paragraph"
        segments = compressor.split_text(long_text)
        assert len(segments) == 3  # 原始文本有3个段落
    
    def test_split_text_sentence(self):
        """测试按句子分割文本"""
        compressor.segment_strategy = "sentence"
        segments = compressor.split_text(long_text)
        assert len(segments) > 0
    
    def test_calculate_importance(self):
        """测试重要性评分功能"""
        importance = compressor.calculate_importance("结论：这是一个重要的结论。")
        assert importance > 0.5  # 包含关键词，应该有较高的重要性
        
        importance = compressor.calculate_importance("这是一个普通的句子。")
        assert importance < 0.5  # 没有关键词，应该有较低的重要性
    
    def test_fast_compress_to_fit(self):
        """测试快速压缩功能"""
        compressed_text = compressor.fast_compress_to_fit(long_text, 50)
        compressed_tokens = tokenizer.count_tokens(compressed_text)
        assert compressed_tokens <= 50
        assert len(compressed_text) > 0
    
    def test_compress_segment(self):
        """测试压缩单个段落"""
        segment = "这是一个很长的段落，用于测试压缩单个段落的功能。这个段落包含了多个句子和一些关键词，例如重要、关键等。"
        compressed_segment = compressor.compress_segment(segment, 20)
        compressed_tokens = tokenizer.count_tokens(compressed_segment)
        assert compressed_tokens <= 20
        assert len(compressed_segment) > 0
    
    def test_dynamic_compress(self):
        """测试动态压缩功能"""
        # 设置较小的会话窗口，确保触发压缩
        compressor.session_len = 100
        compressed_text, was_compressed = compressor.dynamic_compress(long_text, current_prompt="", max_new_tokens=20)
        
        original_tokens = tokenizer.count_tokens(long_text)
        compressed_tokens = tokenizer.count_tokens(compressed_text)
        
        assert was_compressed is True
        assert compressed_tokens < original_tokens
        assert len(compressed_text) > 0
        
        # 检查是否保留了关键词
        assert "结论" in compressed_text
        assert "重要" in compressed_text
    
    def test_dynamic_compress_no_need(self):
        """测试不需要压缩的情况"""
        short_text = "这是一段短文本。"
        compressed_text, was_compressed = compressor.dynamic_compress(short_text, current_prompt="", max_new_tokens=20)
        
        assert was_compressed is False
        assert compressed_text == short_text
    
    def test_compress_chat_history(self):
        """测试聊天历史压缩功能"""
        # 设置较小的会话窗口，确保触发压缩
        compressor.session_len = 150
        compressed_history, was_compressed = compressor.compress_chat_history(chat_history, max_new_tokens=20)
        
        original_tokens = tokenizer.count_tokens(chat_history)
        compressed_tokens = tokenizer.count_tokens(compressed_history)
        
        assert was_compressed is True
        assert compressed_tokens < original_tokens
        assert len(compressed_history) > 0
        
        # 检查是否保留了最新的消息
        assert compressed_history[-1]["role"] == "assistant"
        assert "不客气" in compressed_history[-1]["content"]
    
    def test_compress_chat_history_no_need(self):
        """测试不需要压缩的聊天历史"""
        short_history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好！"}
        ]
        
        compressed_history, was_compressed = compressor.compress_chat_history(short_history, max_new_tokens=20)
        
        assert was_compressed is False
        assert compressed_history == short_history
    
    def test_simple_summarize(self):
        """测试简单摘要功能"""
        summary = compressor.simple_summarize(long_text, max_tokens=30)
        summary_tokens = tokenizer.count_tokens(summary)
        
        assert summary_tokens <= 30
        assert len(summary) > 0
        
        # 检查是否保留了关键词
        assert "结论" in summary

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
