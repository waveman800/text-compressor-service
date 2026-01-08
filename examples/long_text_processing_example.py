"""
超长文本处理完整示例
====================

本示例展示如何对超长文本进行压缩，然后再发起大模型交互请求的完整流程。

适用场景：
- 文档总结：长篇文档需要总结
- 知识库问答：基于大量文档的问答
- 代码分析：分析大型代码库
- 研究论文：处理长篇学术论文
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Dict, Any, Optional
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper


class LongTextProcessor:
    """超长文本处理器，集成压缩和大模型交互"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        context_length: int = 4096,
        max_output_tokens: int = 512,
        enable_compression: bool = True,
        safety_margin: float = 0.1
    ):
        """
        初始化文本处理器
        
        Args:
            model_name: 使用的模型名称
            context_length: 模型的上下文窗口长度
            max_output_tokens: 最大输出token数
            enable_compression: 是否启用压缩
            safety_margin: 安全余量（预留的token比例，默认10%）
        """
        self.model_name = model_name
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.enable_compression = enable_compression
        self.safety_margin = safety_margin
        
        # 计算实际可用的上下文长度（减去安全余量）
        self.available_context = int(context_length * (1 - safety_margin))
        
        # 初始化压缩器
        self.compressor = DynamicContextCompressor(
            session_len=self.available_context,
            max_new_tokens=max_output_tokens,
            enable_dynamic_compression=enable_compression
        )
        
        print(f"初始化文本处理器")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: {context_length} tokens")
        print(f"  可用上下文: {self.available_context} tokens (预留{safety_margin*100:.0f}%)")
        print(f"  最大输出: {max_output_tokens} tokens")
        print(f"  压缩功能: {'启用' if enable_compression else '禁用'}")
        print()
    
    def check_if_compression_needed(
        self,
        text: str,
        prompt: str = ""
    ) -> Dict[str, Any]:
        """
        检查是否需要压缩
        
        Args:
            text: 待处理的文本
            prompt: 提示词
            
        Returns:
            包含检查结果的字典
        """
        # 计算token数
        text_tokens = self.compressor.tokenizer.count_tokens(text)
        prompt_tokens = self.compressor.tokenizer.count_tokens(prompt)
        total_tokens = text_tokens + prompt_tokens + self.max_output_tokens
        
        # 判断是否需要压缩
        needs_compression = total_tokens > self.available_context
        
        result = {
            'text_tokens': text_tokens,
            'prompt_tokens': prompt_tokens,
            'max_output_tokens': self.max_output_tokens,
            'total_tokens': total_tokens,
            'available_context': self.available_context,
            'needs_compression': needs_compression,
            'excess_tokens': max(0, total_tokens - self.available_context)
        }
        
        return result
    
    def process_and_query(
        self,
        text: str,
        prompt: str,
        model_api_func: Optional[callable] = None,
        return_compression_info: bool = False
    ) -> Dict[str, Any]:
        """
        处理文本并发起大模型查询
        
        Args:
            text: 待处理的文本
            prompt: 提示词
            model_api_func: 大模型API调用函数（可选，如果不提供则返回构建的请求）
            return_compression_info: 是否返回压缩信息
            
        Returns:
            包含结果和信息的字典
        """
        print("="*70)
        print("步骤1: 检查文本长度")
        print("="*70)
        
        # 检查是否需要压缩
        check_result = self.check_if_compression_needed(text, prompt)
        
        print(f"文本tokens: {check_result['text_tokens']}")
        print(f"提示词tokens: {check_result['prompt_tokens']}")
        print(f"预留输出tokens: {check_result['max_output_tokens']}")
        print(f"总计tokens: {check_result['total_tokens']}")
        print(f"可用上下文: {check_result['available_context']}")
        
        if check_result['needs_compression']:
            print(f"⚠️  需要压缩！超出 {check_result['excess_tokens']} tokens")
        else:
            print(f"✅ 无需压缩，可以直接处理")
        
        print()
        
        # 压缩文本（如果需要）
        processed_text = text
        compression_info = None
        
        if self.enable_compression and check_result['needs_compression']:
            print("="*70)
            print("步骤2: 执行文本压缩")
            print("="*70)
            
            start_time = time.time()
            processed_text, was_compressed = self.compressor.dynamic_compress(
                text=text,
                current_prompt=prompt,
                max_new_tokens=self.max_output_tokens
            )
            compression_time = time.time() - start_time
            
            if was_compressed:
                compressed_tokens = self.compressor.tokenizer.count_tokens(processed_text)
                compression_ratio = compressed_tokens / check_result['text_tokens']
                
                print(f"✅ 压缩完成")
                print(f"  原始tokens: {check_result['text_tokens']}")
                print(f"  压缩后tokens: {compressed_tokens}")
                print(f"  压缩比: {compression_ratio:.2%}")
                print(f"  压缩耗时: {compression_time:.3f}秒")
                
                compression_info = {
                    'original_tokens': check_result['text_tokens'],
                    'compressed_tokens': compressed_tokens,
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time
                }
            else:
                print("ℹ️  文本已符合要求，无需压缩")
        else:
            print("="*70)
            print("步骤2: 跳过压缩")
            print("="*70)
            print("文本长度在允许范围内，直接使用原始文本")
        
        print()
        
        # 构建最终请求
        print("="*70)
        print("步骤3: 构建大模型请求")
        print("="*70)
        
        final_input = f"{processed_text}\n\n{prompt}" if processed_text else prompt
        final_tokens = self.compressor.tokenizer.count_tokens(final_input)
        
        print(f"最终输入tokens: {final_tokens}")
        print(f"上下文使用率: {final_tokens/self.available_context:.2%}")
        
        # 构建请求
        request = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'system',
                    'content': '你是一个专业的AI助手，擅长分析和总结文本。'
                },
                {
                    'role': 'user',
                    'content': final_input
                }
            ],
            'max_tokens': self.max_output_tokens,
            'temperature': 0.7
        }
        
        print(f"请求已构建，准备发送给模型: {self.model_name}")
        print()
        
        # 调用模型API（如果提供了）
        model_response = None
        
        if model_api_func:
            print("="*70)
            print("步骤4: 调用大模型API")
            print("="*70)
            
            try:
                model_response = model_api_func(request)
                print("✅ 模型调用成功")
            except Exception as e:
                print(f"❌ 模型调用失败: {e}")
                model_response = None
        else:
            print("="*70)
            print("步骤4: 跳过模型调用")
            print("="*70)
            print("未提供模型API函数，仅返回构建的请求")
        
        print()
        
        # 返回结果
        result = {
            'request': request,
            'model_response': model_response,
            'compression_info': compression_info,
            'token_info': {
                'original_text_tokens': check_result['text_tokens'],
                'final_input_tokens': final_tokens,
                'context_usage': final_tokens / self.available_context
            }
        }
        
        if return_compression_info:
            result['check_result'] = check_result
        
        return result
    
    def batch_process(
        self,
        texts: list,
        prompt: str,
        model_api_func: Optional[callable] = None
    ) -> list:
        """
        批量处理多个文本
        
        Args:
            texts: 文本列表
            prompt: 提示词
            model_api_func: 大模型API调用函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"\n{'='*70}")
            print(f"处理文本 {i+1}/{len(texts)}")
            print(f"{'='*70}\n")
            
            result = self.process_and_query(text, prompt, model_api_func)
            results.append(result)
        
        return results


def mock_llm_api(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    模拟大模型API调用（用于演示）
    
    Args:
        request: 请求字典
        
    Returns:
        模拟的响应
    """
    # 模拟API调用延迟
    time.sleep(0.5)
    
    # 返回模拟响应
    return {
        'id': 'chatcmpl-mock',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': request['model'],
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '这是一个模拟的模型响应。在实际使用中，这里会是真实的大模型输出。'
                },
                'finish_reason': 'stop'
            }
        ],
        'usage': {
            'prompt_tokens': request['max_tokens'] // 2,
            'completion_tokens': request['max_tokens'] // 4,
            'total_tokens': request['max_tokens'] * 3 // 4
        }
    }


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*70)
    print("示例1: 基本使用 - 处理超长文本")
    print("="*70 + "\n")
    
    # 创建处理器
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    # 超长文本（模拟）
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，
    但能像人那样思考、也可能超过人的智能。
    
    机器学习是人工智能的核心，是使计算机具有智能的根本途径。
    深度学习是机器学习领域中一种新的方法，它源于人工神经网络的研究。
    
    卷积神经网络（CNN）是一种专门用来处理具有类似网格结构数据的神经网络。
    循环神经网络（RNN）是一种用于处理序列数据的神经网络。
    Transformer是一种基于自注意力机制的神经网络架构。
    
    大语言模型（LLM）是指具有大量参数的语言模型，通常使用Transformer架构。
    GPT（Generative Pre-trained Transformer）是一种生成式预训练Transformer模型。
    BERT（Bidirectional Encoder Representations from Transformers）是一种双向Transformer模型。
    
    自然语言处理（NLP）是人工智能的一个重要分支，它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    计算机视觉是另一个重要分支，它研究如何让计算机"看"和理解图像和视频。
    """ * 10  # 重复10次以创建超长文本
    
    prompt = "请总结这段关于人工智能的文字，提取关键信息。"
    
    # 处理并查询
    result = processor.process_and_query(
        text=long_text,
        prompt=prompt,
        model_api_func=mock_llm_api
    )
    
    # 显示结果
    print("\n" + "="*70)
    print("处理结果")
    print("="*70)
    print(f"压缩信息: {result['compression_info']}")
    print(f"Token使用: {result['token_info']}")
    if result['model_response']:
        print(f"模型响应: {result['model_response']['choices'][0]['message']['content']}")


def example_2_no_compression_needed():
    """示例2: 无需压缩的情况"""
    print("\n" + "="*70)
    print("示例2: 无需压缩 - 短文本处理")
    print("="*70 + "\n")
    
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    short_text = """
    人工智能（AI）是计算机科学的一个分支，它研究如何让计算机模拟人类的智能行为。
    """
    
    prompt = "这句话是什么意思？"
    
    result = processor.process_and_query(
        text=short_text,
        prompt=prompt,
        model_api_func=mock_llm_api
    )
    
    print("\n" + "="*70)
    print("处理结果")
    print("="*70)
    print(f"压缩信息: {result['compression_info']}")
    print(f"Token使用: {result['token_info']}")


def example_3_batch_processing():
    """示例3: 批量处理"""
    print("\n" + "="*70)
    print("示例3: 批量处理多个文档")
    print("="*70 + "\n")
    
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=256,
        enable_compression=True
    )
    
    # 多个文档
    documents = [
        "文档1: 人工智能是计算机科学的一个分支..." * 50,
        "文档2: 机器学习是AI的核心技术..." * 50,
        "文档3: 深度学习是机器学习的一种方法..." * 50
    ]
    
    prompt = "请提取这段文字的核心观点。"
    
    # 批量处理
    results = processor.batch_process(
        texts=documents,
        prompt=prompt,
        model_api_func=mock_llm_api
    )
    
    # 显示汇总
    print("\n" + "="*70)
    print("批量处理汇总")
    print("="*70)
    for i, result in enumerate(results):
        if result['compression_info']:
            print(f"文档{i+1}: 压缩比 {result['compression_info']['compression_ratio']:.2%}")
        else:
            print(f"文档{i+1}: 无需压缩")


def example_4_custom_model_api():
    """示例4: 使用自定义模型API"""
    print("\n" + "="*70)
    print("示例4: 集成真实的大模型API")
    print("="*70 + "\n")
    
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    long_text = "这是一个长文本..." * 100
    prompt = "请总结这段文字。"
    
    # 自定义API函数（示例）
    def custom_api(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        自定义模型API函数
        可以集成OpenAI API、Claude API、本地模型等
        """
        # 这里可以调用真实的API
        # 例如使用openai库、requests库等
        # return openai.ChatCompletion.create(**request)
        
        # 演示使用mock
        return mock_llm_api(request)
    
    result = processor.process_and_query(
        text=long_text,
        prompt=prompt,
        model_api_func=custom_api
    )
    
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("超长文本处理完整示例")
    print("="*70)
    
    # 运行示例
    example_1_basic_usage()
    example_2_no_compression_needed()
    example_3_batch_processing()
    example_4_custom_model_api()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)
