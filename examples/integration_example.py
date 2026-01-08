"""
推理框架集成示例
================

本示例展示如何将文本压缩服务集成到大模型推理框架中，
实现自动的上下文窗口管理和动态压缩机制。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import List, Dict, Any, Optional
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper


class LLMInferenceEngine:
    """
    大模型推理引擎，集成动态压缩功能
    """
    
    def __init__(
        self,
        model_name: str,
        context_length: int = 4096,
        max_output_tokens: int = 256,
        enable_compression: bool = True,
        use_fast_compression: bool = False
    ):
        """
        初始化推理引擎
        
        Args:
            model_name: 模型名称
            context_length: 模型的上下文窗口长度
            max_output_tokens: 最大输出token数
            enable_compression: 是否启用动态压缩
            use_fast_compression: 是否使用快速压缩模式
        """
        self.model_name = model_name
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.enable_compression = enable_compression
        
        # 初始化压缩器
        self.compressor = DynamicContextCompressor(
            session_len=context_length,
            max_new_tokens=max_output_tokens,
            enable_dynamic_compression=enable_compression,
            use_fast_compression=use_fast_compression
        )
        
        # 模拟模型加载
        print(f"加载模型: {model_name}")
        print(f"上下文窗口长度: {context_length}")
        print(f"最大输出token数: {max_output_tokens}")
        print(f"动态压缩: {'启用' if enable_compression else '禁用'}")
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        生成文本（同步版本）
        
        Args:
            prompt: 提示词
            context: 上下文文本
            max_new_tokens: 最大生成token数（可选）
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        
        # 压缩上下文（如果启用）
        if self.enable_compression and context:
            compressed_context, was_compressed = self.compressor.dynamic_compress(
                text=context,
                current_prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            if was_compressed:
                print(f"[压缩] 上下文已压缩以适应上下文窗口")
            
            context = compressed_context
        
        # 构建完整输入
        full_input = f"{context}\n\n{prompt}" if context else prompt
        
        # 计算token数
        input_tokens = self.compressor.tokenizer.count_tokens(full_input)
        print(f"[推理] 输入token数: {input_tokens}/{self.context_length}")
        
        # 模拟推理过程
        # 实际应用中，这里会调用真实的模型推理
        output = self._mock_inference(full_input, max_new_tokens)
        
        return output
    
    async def generate_async(
        self,
        prompt: str,
        context: str = "",
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        生成文本（异步版本）
        
        Args:
            prompt: 提示词
            context: 上下文文本
            max_new_tokens: 最大生成token数（可选）
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        
        # 异步压缩上下文
        if self.enable_compression and context:
            compressed_context, was_compressed = await self.compressor.dynamic_compress_async(
                text=context,
                current_prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            if was_compressed:
                print(f"[压缩] 上下文已压缩以适应上下文窗口")
            
            context = compressed_context
        
        # 构建完整输入
        full_input = f"{context}\n\n{prompt}" if context else prompt
        
        # 计算token数
        input_tokens = self.compressor.tokenizer.count_tokens(full_input)
        print(f"[推理] 输入token数: {input_tokens}/{self.context_length}")
        
        # 模拟推理过程
        output = await self._mock_inference_async(full_input, max_new_tokens)
        
        return output
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        聊天对话（同步版本）
        
        Args:
            messages: 聊天历史，格式为 [{"role": "user", "content": "..."}]
            max_new_tokens: 最大生成token数（可选）
            
        Returns:
            生成的回复
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        
        # 提取最新的用户消息作为提示词
        current_prompt = messages[-1]["content"] if messages else ""
        
        # 压缩聊天历史（如果启用）
        if self.enable_compression and len(messages) > 1:
            history_messages = messages[:-1]
            compressed_history, was_compressed = self.compressor.compress_chat_history(
                chat_history=history_messages,
                current_prompt=current_prompt,
                max_new_tokens=max_new_tokens
            )
            
            if was_compressed:
                print(f"[压缩] 聊天历史已压缩以适应上下文窗口")
            
            messages = compressed_history + [messages[-1]]
        
        # 构建完整输入
        full_input = self._format_chat_messages(messages)
        
        # 计算token数
        input_tokens = self.compressor.tokenizer.count_tokens(full_input)
        print(f"[推理] 输入token数: {input_tokens}/{self.context_length}")
        
        # 模拟推理过程
        output = self._mock_inference(full_input, max_new_tokens)
        
        return output
    
    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        聊天对话（异步版本）
        
        Args:
            messages: 聊天历史，格式为 [{"role": "user", "content": "..."}]
            max_new_tokens: 最大生成token数（可选）
            
        Returns:
            生成的回复
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_output_tokens
        
        # 提取最新的用户消息作为提示词
        current_prompt = messages[-1]["content"] if messages else ""
        
        # 异步压缩聊天历史
        if self.enable_compression and len(messages) > 1:
            history_messages = messages[:-1]
            compressed_history, was_compressed = await self.compressor.compress_chat_history_async(
                chat_history=history_messages,
                current_prompt=current_prompt,
                max_new_tokens=max_new_tokens
            )
            
            if was_compressed:
                print(f"[压缩] 聊天历史已压缩以适应上下文窗口")
            
            messages = compressed_history + [messages[-1]]
        
        # 构建完整输入
        full_input = self._format_chat_messages(messages)
        
        # 计算token数
        input_tokens = self.compressor.tokenizer.count_tokens(full_input)
        print(f"[推理] 输入token数: {input_tokens}/{self.context_length}")
        
        # 模拟推理过程
        output = await self._mock_inference_async(full_input, max_new_tokens)
        
        return output
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化聊天消息为字符串"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)
    
    def _mock_inference(self, input_text: str, max_tokens: int) -> str:
        """模拟推理过程（实际应用中替换为真实的模型推理）"""
        # 这里只是模拟，实际应用中会调用真实的模型
        return f"[模拟输出] 基于输入: {input_text[:50]}... 生成了 {max_tokens} 个token的回复"
    
    async def _mock_inference_async(self, input_text: str, max_tokens: int) -> str:
        """模拟异步推理过程"""
        await asyncio.sleep(0.1)  # 模拟推理延迟
        return self._mock_inference(input_text, max_tokens)
    
    def get_compression_metrics(
        self,
        original_text: str,
        compressed_text: str
    ) -> Dict[str, Any]:
        """
        获取压缩质量指标
        
        Args:
            original_text: 原始文本
            compressed_text: 压缩后的文本
            
        Returns:
            压缩质量指标字典
        """
        metrics = self.compressor.evaluate_compression(
            original_text=original_text,
            compressed_text=compressed_text
        )
        return metrics.to_dict()
    
    def close(self):
        """关闭推理引擎"""
        self.compressor.close()
        print("推理引擎已关闭")


# 使用示例
def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建推理引擎
    engine = LLMInferenceEngine(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=256,
        enable_compression=True
    )
    
    # 生成文本
    prompt = "请总结以下内容的关键点："
    context = "这是一个很长的上下文文本..." * 100  # 模拟长文本
    
    output = engine.generate(prompt=prompt, context=context)
    print(f"输出: {output}\n")
    
    engine.close()


def example_chat_usage():
    """聊天使用示例"""
    print("=" * 60)
    print("示例2: 聊天对话")
    print("=" * 60)
    
    # 创建推理引擎
    engine = LLMInferenceEngine(
        model_name="gpt-4",
        context_length=8192,
        max_output_tokens=512,
        enable_compression=True
    )
    
    # 聊天历史
    messages = [
        {"role": "user", "content": "你好，我想了解Python编程。"},
        {"role": "assistant", "content": "你好！Python是一种高级编程语言..."},
        {"role": "user", "content": "Python有哪些主要特点？"},
        {"role": "assistant", "content": "Python的主要特点包括：简洁易读..."},
        # ... 更多对话
    ]
    
    # 添加更多历史消息以触发压缩
    for i in range(20):
        messages.append({
            "role": "user",
            "content": f"这是第{i+1}轮对话的问题..." * 10
        })
        messages.append({
            "role": "assistant",
            "content": f"这是第{i+1}轮对话的回答..." * 10
        })
    
    # 添加新的用户消息
    messages.append({
        "role": "user",
        "content": "请总结我们刚才讨论的主要内容。"
    })
    
    # 生成回复
    output = engine.chat(messages=messages)
    print(f"输出: {output}\n")
    
    engine.close()


def example_async_usage():
    """异步使用示例"""
    print("=" * 60)
    print("示例3: 异步处理")
    print("=" * 60)
    
    async def async_main():
        # 创建推理引擎
        engine = LLMInferenceEngine(
            model_name="claude-3",
            context_length=100000,
            max_output_tokens=1024,
            enable_compression=True
        )
        
        # 异步生成文本
        prompt = "请分析以下数据："
        context = "这是一个很长的数据分析文本..." * 100
        
        output = await engine.generate_async(prompt=prompt, context=context)
        print(f"输出: {output}\n")
        
        engine.close()
    
    asyncio.run(async_main())


def example_compression_metrics():
    """压缩质量评估示例"""
    print("=" * 60)
    print("示例4: 压缩质量评估")
    print("=" * 60)
    
    # 创建推理引擎
    engine = LLMInferenceEngine(
        model_name="llama-2-7b",
        context_length=4096,
        max_output_tokens=256,
        enable_compression=True
    )
    
    # 压缩文本
    original_text = "这是一个很长的原始文本..." * 100
    compressed_text, was_compressed = engine.compressor.dynamic_compress(
        text=original_text,
        current_prompt="请总结：",
        max_new_tokens=256
    )
    
    if was_compressed:
        # 获取压缩质量指标
        metrics = engine.get_compression_metrics(original_text, compressed_text)
        
        print("压缩质量指标:")
        print(f"  原始token数: {metrics['original_tokens']}")
        print(f"  压缩后token数: {metrics['compressed_tokens']}")
        print(f"  压缩比: {metrics['compression_ratio']:.2%}")
        print(f"  关键词保留率: {metrics['keyword_retention']:.2%}")
        print(f"  信息保留率: {metrics['information_preservation']:.2%}")
        print(f"  处理时间: {metrics['processing_time']:.4f}秒\n")
    
    engine.close()


def example_batch_processing():
    """批量处理示例"""
    print("=" * 60)
    print("示例5: 批量处理")
    print("=" * 60)
    
    async def batch_main():
        # 创建推理引擎
        engine = LLMInferenceEngine(
            model_name="gpt-3.5-turbo",
            context_length=4096,
            max_output_tokens=256,
            enable_compression=True
        )
        
        # 批量压缩多个文本
        texts = [
            "这是第一个长文本..." * 50,
            "这是第二个长文本..." * 50,
            "这是第三个长文本..." * 50,
        ]
        
        results = await engine.compressor.batch_compress(
            texts=texts,
            current_prompt="请总结：",
            max_new_tokens=256
        )
        
        print(f"批量压缩了 {len(results)} 个文本")
        for i, (compressed, was_compressed) in enumerate(results):
            print(f"  文本{i+1}: {'已压缩' if was_compressed else '无需压缩'}")
        print()
        
        engine.close()
    
    asyncio.run(batch_main())


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_chat_usage()
    example_async_usage()
    example_compression_metrics()
    example_batch_processing()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
