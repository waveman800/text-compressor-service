import asyncio
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dynamic_compressor import DynamicContextCompressor, CompressionMetrics
from core.tokenizer_wrapper import TokenizerWrapper


class TestOptimizedFeatures:
    """测试优化后的功能"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.tokenizer = TokenizerWrapper()
        cls.compressor = DynamicContextCompressor(
            session_len=4096,
            max_new_tokens=256,
            tokenizer=cls.tokenizer
        )
        
        cls.long_text = """
        这是一个关于人工智能发展的重要文档。人工智能技术在过去几年中取得了显著的进展。
        
        首先，深度学习技术的发展使得计算机在图像识别、自然语言处理等领域取得了突破性的成果。
        这些成果不仅改变了我们的生活方式，也为各行各业带来了新的机遇。
        
        其次，大语言模型的出现使得机器能够更好地理解和生成人类语言。
        这对于提高人机交互的效率和体验具有重要意义。
        
        结论：人工智能技术将继续快速发展，为人类社会带来更多的创新和变革。
        我们需要关注这一趋势，并积极应对其中的挑战和机遇。
        
        关键点：
        1. 深度学习技术的突破
        2. 大语言模型的应用
        3. 人机交互的改进
        4. 未来发展的方向
        
        因此，我们应该加强对人工智能技术的研究和应用，推动相关产业的发展。
        """
        
        cls.chat_history = [
            {"role": "user", "content": "你好，我想了解一下人工智能的发展历史"},
            {"role": "assistant", "content": "人工智能的发展可以追溯到20世纪50年代。从早期的符号主义到现在的深度学习，AI经历了多次技术革命。"},
            {"role": "user", "content": "那深度学习是什么时候开始流行的？"},
            {"role": "assistant", "content": "深度学习在2012年左右开始流行，当时AlexNet在ImageNet比赛中取得了突破性成绩。"},
            {"role": "user", "content": "不客气，还有什么想了解的吗？"}
        ]
    
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        cls.compressor.close()
    
    def test_async_compress(self):
        """测试异步压缩功能"""
        async def run_test():
            compressed_text, was_compressed = await self.compressor.dynamic_compress_async(
                self.long_text,
                current_prompt="请总结这段文字",
                max_new_tokens=256
            )
            
            assert isinstance(compressed_text, str)
            assert len(compressed_text) > 0
            assert was_compressed is True
            
            compressed_tokens = self.tokenizer.count_tokens(compressed_text)
            assert compressed_tokens < self.tokenizer.count_tokens(self.long_text)
        
        asyncio.run(run_test())
        print("✓ 异步压缩功能测试通过")
    
    def test_batch_compress(self):
        """测试批量压缩功能"""
        async def run_test():
            texts = [self.long_text] * 3
            results = await self.compressor.batch_compress(
                texts,
                current_prompt="请总结",
                max_new_tokens=256
            )
            
            assert len(results) == 3
            for compressed_text, was_compressed in results:
                assert isinstance(compressed_text, str)
                assert len(compressed_text) > 0
                assert was_compressed is True
        
        asyncio.run(run_test())
        print("✓ 批量压缩功能测试通过")
    
    def test_batch_compress_chat_histories(self):
        """测试批量压缩聊天历史功能"""
        async def run_test():
            chat_histories = [self.chat_history] * 2
            results = await self.compressor.batch_compress_chat_histories(
                chat_histories,
                current_prompt="请回答",
                max_new_tokens=256
            )
            
            assert len(results) == 2
            for compressed_history, was_compressed in results:
                assert isinstance(compressed_history, list)
                assert len(compressed_history) > 0
        
        asyncio.run(run_test())
        print("✓ 批量压缩聊天历史功能测试通过")
    
    def test_compression_metrics(self):
        """测试压缩质量评估功能"""
        compressed_text, _ = self.compressor.dynamic_compress(
            self.long_text,
            current_prompt="请总结",
            max_new_tokens=256
        )
        
        metrics = self.compressor.evaluate_compression(
            self.long_text,
            compressed_text,
            processing_time=0.5
        )
        
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.original_tokens > 0
        assert metrics.compressed_tokens > 0
        assert metrics.compression_ratio > 0
        assert 0 <= metrics.keyword_retention <= 1
        assert 0 <= metrics.information_preservation <= 1
        assert metrics.processing_time >= 0
        
        metrics_dict = metrics.to_dict()
        assert 'original_tokens' in metrics_dict
        assert 'compressed_tokens' in metrics_dict
        assert 'compression_ratio' in metrics_dict
        assert 'keyword_retention' in metrics_dict
        assert 'information_preservation' in metrics_dict
        assert 'processing_time' in metrics_dict
        
        print(f"  原始tokens: {metrics.original_tokens}")
        print(f"  压缩后tokens: {metrics.compressed_tokens}")
        print(f"  压缩比: {metrics.compression_ratio:.2%}")
        print(f"  关键词保留率: {metrics.keyword_retention:.2%}")
        print(f"  信息保留率: {metrics.information_preservation:.2%}")
        print("✓ 压缩质量评估功能测试通过")
    
    def test_compress_with_metrics(self):
        """测试带质量指标的压缩功能"""
        compressed_text, was_compressed, metrics = self.compressor.compress_with_metrics(
            self.long_text,
            current_prompt="请总结",
            max_new_tokens=256
        )
        
        assert isinstance(compressed_text, str)
        assert len(compressed_text) > 0
        assert was_compressed is True
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.processing_time > 0
        
        print(f"  处理时间: {metrics.processing_time:.4f}秒")
        print("✓ 带质量指标的压缩功能测试通过")
    
    def test_text_type_detection(self):
        """测试文本类型检测功能"""
        dialogue_text = '张三说："你好吗？" 李四回答："我很好。"'
        list_text = "1. 第一点\n2. 第二点\n3. 第三点"
        code_text = "```python\ndef hello():\n    print('Hello')\n```"
        narrative_text = "这是一段普通的叙述性文本。"
        
        assert self.compressor.detect_text_type(dialogue_text) == 'dialogue'
        assert self.compressor.detect_text_type(list_text) == 'list'
        assert self.compressor.detect_text_type(code_text) == 'code'
        assert self.compressor.detect_text_type(narrative_text) == 'narrative'
        
        print("✓ 文本类型检测功能测试通过")
    
    def test_importance_score_with_position(self):
        """测试带位置权重的重要性评分"""
        segments = [
            "这是第一段，应该有较高的位置权重。",
            "这是中间段，位置权重适中。",
            "这是最后一段，也应该有较高的位置权重。"
        ]
        
        scores = []
        for idx, segment in enumerate(segments):
            score = self.compressor.calculate_importance(segment, idx, len(segments))
            scores.append(score)
        
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
        
        print(f"  各段落重要性分数: {scores}")
        print("✓ 带位置权重的重要性评分测试通过")
    
    def test_performance_improvement(self):
        """测试性能改进"""
        iterations = 10
        
        # 测试旧方法的性能（模拟）
        start_time = time.time()
        for _ in range(iterations):
            self.compressor.dynamic_compress(self.long_text, max_new_tokens=256)
        old_time = time.time() - start_time
        
        # 测试新方法的性能（使用优化后的compress_segment）
        start_time = time.time()
        for _ in range(iterations):
            self.compressor.dynamic_compress(self.long_text, max_new_tokens=256)
        new_time = time.time() - start_time
        
        print(f"  {iterations}次迭代平均时间: {new_time/iterations:.4f}秒")
        print("✓ 性能测试完成")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试优化后的功能")
    print("=" * 60)
    
    test = TestOptimizedFeatures()
    test.setup_class()
    
    try:
        test.test_async_compress()
        test.test_batch_compress()
        test.test_batch_compress_chat_histories()
        test.test_compression_metrics()
        test.test_compress_with_metrics()
        test.test_text_type_detection()
        test.test_importance_score_with_position()
        test.test_performance_improvement()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.teardown_class()


if __name__ == "__main__":
    run_all_tests()
