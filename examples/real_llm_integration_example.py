"""
真实大模型API集成示例
======================

本示例展示如何将文本压缩服务与真实的大模型API（如OpenAI、Claude等）集成，
实现自动的文本压缩和模型调用。

支持的大模型API：
- OpenAI API (GPT-3.5, GPT-4)
- Anthropic Claude API
- 本地模型 (vLLM, LM Deploy等)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from typing import Dict, Any, Optional
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper


class LLMClientWithCompression:
    """集成压缩功能的大模型客户端"""
    
    def __init__(
        self,
        api_type: str = "openai",
        api_key: str = "",
        base_url: str = "",
        model_name: str = "gpt-3.5-turbo",
        context_length: int = 4096,
        max_output_tokens: int = 512,
        enable_compression: bool = True,
        safety_margin: float = 0.1
    ):
        """
        初始化客户端
        
        Args:
            api_type: API类型 ('openai', 'anthropic', 'custom')
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            context_length: 上下文窗口长度
            max_output_tokens: 最大输出token数
            enable_compression: 是否启用压缩
            safety_margin: 安全余量
        """
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.enable_compression = enable_compression
        self.safety_margin = safety_margin
        
        # 计算可用上下文
        self.available_context = int(context_length * (1 - safety_margin))
        
        # 初始化压缩器
        self.compressor = DynamicContextCompressor(
            session_len=self.available_context,
            max_new_tokens=max_output_tokens,
            enable_dynamic_compression=enable_compression
        )
        
        print(f"初始化LLM客户端（带压缩功能）")
        print(f"  API类型: {api_type}")
        print(f"  模型: {model_name}")
        print(f"  上下文窗口: {context_length} tokens")
        print(f"  可用上下文: {self.available_context} tokens")
        print(f"  压缩功能: {'启用' if enable_compression else '禁用'}")
        print()
    
    def chat_with_long_context(
        self,
        messages: list,
        system_prompt: str = "你是一个专业的AI助手。",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用长上下文进行对话
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Returns:
            包含响应和元数据的字典
        """
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        
        print("="*70)
        print("处理长上下文对话请求")
        print("="*70)
        
        # 提取用户消息中的长文本
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content += msg['content'] + "\n"
        
        # 计算token数
        system_tokens = self.compressor.tokenizer.count_tokens(system_prompt)
        user_tokens = self.compressor.tokenizer.count_tokens(user_content)
        total_tokens = system_tokens + user_tokens + max_tokens
        
        print(f"系统提示词tokens: {system_tokens}")
        print(f"用户内容tokens: {user_tokens}")
        print(f"预留输出tokens: {max_tokens}")
        print(f"总计tokens: {total_tokens}")
        print(f"可用上下文: {self.available_context}")
        
        # 判断是否需要压缩
        needs_compression = total_tokens > self.available_context
        
        if needs_compression:
            print(f"⚠️  需要压缩！超出 {total_tokens - self.available_context} tokens")
        else:
            print(f"✅ 无需压缩")
        
        print()
        
        # 压缩用户内容（如果需要）
        processed_user_content = user_content
        compression_info = None
        
        if self.enable_compression and needs_compression:
            print("="*70)
            print("执行文本压缩")
            print("="*70)
            
            start_time = time.time()
            processed_user_content, was_compressed = self.compressor.dynamic_compress(
                text=user_content,
                current_prompt=system_prompt,
                max_new_tokens=max_tokens
            )
            compression_time = time.time() - start_time
            
            if was_compressed:
                compressed_tokens = self.compressor.tokenizer.count_tokens(processed_user_content)
                compression_ratio = compressed_tokens / user_tokens
                
                print(f"✅ 压缩完成")
                print(f"  原始tokens: {user_tokens}")
                print(f"  压缩后tokens: {compressed_tokens}")
                print(f"  压缩比: {compression_ratio:.2%}")
                print(f"  压缩耗时: {compression_time:.3f}秒")
                
                compression_info = {
                    'original_tokens': user_tokens,
                    'compressed_tokens': compressed_tokens,
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time
                }
            else:
                print("ℹ️  文本已符合要求，无需压缩")
        
        # 构建最终消息
        final_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加压缩后的用户消息
        if processed_user_content:
            final_messages.append({"role": "user", "content": processed_user_content})
        
        # 调用API
        print("="*70)
        print("调用大模型API")
        print("="*70)
        
        response = self._call_api(
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 返回结果
        result = {
            'response': response,
            'compression_info': compression_info,
            'token_info': {
                'original_user_tokens': user_tokens,
                'final_input_tokens': self.compressor.tokenizer.count_tokens(
                    system_prompt + processed_user_content
                ),
                'context_usage': (system_tokens + 
                                 self.compressor.tokenizer.count_tokens(processed_user_content) + 
                                 max_tokens) / self.available_context
            }
        }
        
        return result
    
    def _call_api(
        self,
        messages: list,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        调用大模型API
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            API响应
        """
        if self.api_type == "openai":
            return self._call_openai_api(messages, temperature, max_tokens)
        elif self.api_type == "anthropic":
            return self._call_anthropic_api(messages, temperature, max_tokens)
        elif self.api_type == "custom":
            return self._call_custom_api(messages, temperature, max_tokens)
        else:
            raise ValueError(f"不支持的API类型: {self.api_type}")
    
    def _call_openai_api(
        self,
        messages: list,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """调用OpenAI API"""
        try:
            import openai
            
            # 配置客户端
            client_kwargs = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            
            client = openai.OpenAI(**client_kwargs)
            
            # 调用API
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                'content': response.choices[0].message.content,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except ImportError:
            print("⚠️  未安装openai库，使用模拟响应")
            print("   安装命令: pip install openai")
            return self._mock_response(messages, max_tokens)
        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            return self._mock_response(messages, max_tokens)
    
    def _call_anthropic_api(
        self,
        messages: list,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """调用Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # 转换消息格式
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                elif msg['role'] == 'user':
                    user_messages.append(msg)
            
            # 调用API
            response = client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=user_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                'content': response.content[0].text,
                'finish_reason': response.stop_reason,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
        except ImportError:
            print("⚠️  未安装anthropic库，使用模拟响应")
            print("   安装命令: pip install anthropic")
            return self._mock_response(messages, max_tokens)
        except Exception as e:
            print(f"❌ Anthropic API调用失败: {e}")
            return self._mock_response(messages, max_tokens)
    
    def _call_custom_api(
        self,
        messages: list,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """调用自定义API（使用requests）"""
        try:
            import requests
            
            url = f"{self.base_url}/chat/completions"
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                'model': self.model_name,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'content': data['choices'][0]['message']['content'],
                'finish_reason': data['choices'][0]['finish_reason'],
                'usage': data.get('usage', {})
            }
            
        except Exception as e:
            print(f"❌ 自定义API调用失败: {e}")
            return self._mock_response(messages, max_tokens)
    
    def _mock_response(self, messages: list, max_tokens: int) -> Dict[str, Any]:
        """模拟API响应"""
        return {
            'content': '这是一个模拟的API响应。在实际使用中，这里会是真实的大模型输出。',
            'finish_reason': 'stop',
            'usage': {
                'prompt_tokens': max_tokens // 2,
                'completion_tokens': max_tokens // 4,
                'total_tokens': max_tokens * 3 // 4
            }
        }


def example_openai_integration():
    """示例1: 集成OpenAI API"""
    print("\n" + "="*70)
    print("示例1: 集成OpenAI API")
    print("="*70 + "\n")
    
    # 创建客户端
    client = LLMClientWithCompression(
        api_type="openai",
        api_key="your-openai-api-key",  # 替换为真实的API密钥
        base_url="",  # 使用默认的OpenAI API地址
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    # 长文本
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
    """ * 20  # 重复20次以创建超长文本
    
    messages = [
        {"role": "user", "content": long_text}
    ]
    
    # 调用
    result = client.chat_with_long_context(
        messages=messages,
        system_prompt="你是一个专业的AI助手，擅长总结和分析文本。",
        temperature=0.7
    )
    
    # 显示结果
    print("\n" + "="*70)
    print("响应结果")
    print("="*70)
    print(f"压缩信息: {result['compression_info']}")
    print(f"Token使用: {result['token_info']}")
    print(f"\n模型响应:\n{result['response']['content']}")


def example_anthropic_integration():
    """示例2: 集成Anthropic Claude API"""
    print("\n" + "="*70)
    print("示例2: 集成Anthropic Claude API")
    print("="*70 + "\n")
    
    client = LLMClientWithCompression(
        api_type="anthropic",
        api_key="your-anthropic-api-key",  # 替换为真实的API密钥
        model_name="claude-3-sonnet-20240229",
        context_length=200000,  # Claude 3支持200K上下文
        max_output_tokens=4096,
        enable_compression=True
    )
    
    long_text = "这是一个长文本..." * 1000
    messages = [{"role": "user", "content": long_text}]
    
    result = client.chat_with_long_context(
        messages=messages,
        system_prompt="你是一个专业的AI助手。",
        temperature=0.7
    )
    
    print("\n响应结果:")
    print(result['response']['content'])


def example_custom_api():
    """示例3: 集成自定义API（如本地模型）"""
    print("\n" + "="*70)
    print("示例3: 集成自定义API（本地模型）")
    print("="*70 + "\n")
    
    client = LLMClientWithCompression(
        api_type="custom",
        api_key="dummy-key",
        base_url="http://localhost:8000/v1",  # 本地模型服务地址
        model_name="local-model",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    long_text = "这是一个长文本..." * 100
    messages = [{"role": "user", "content": long_text}]
    
    result = client.chat_with_long_context(
        messages=messages,
        system_prompt="你是一个专业的AI助手。",
        temperature=0.7
    )
    
    print("\n响应结果:")
    print(result['response']['content'])


def example_document_qa():
    """示例4: 文档问答场景"""
    print("\n" + "="*70)
    print("示例4: 文档问答场景")
    print("="*70 + "\n")
    
    client = LLMClientWithCompression(
        api_type="openai",
        api_key="your-openai-api-key",
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=1024,
        enable_compression=True
    )
    
    # 模拟长文档
    document = """
    # 项目报告：人工智能在医疗领域的应用

    ## 1. 引言
    人工智能（AI）技术在医疗健康领域的应用日益广泛，从辅助诊断到个性化治疗，
    AI正在改变传统的医疗模式。本报告旨在分析AI在医疗领域的现状、挑战和未来趋势。

    ## 2. 主要应用领域

    ### 2.1 医学影像诊断
    AI在医学影像分析方面取得了显著进展。深度学习模型能够准确识别X光、CT、MRI等
    医学影像中的异常，辅助医生进行诊断。研究表明，AI在某些特定疾病的诊断准确率上
    已经达到或超过了人类专家的水平。

    ### 2.2 药物研发
    AI技术大大加速了新药研发的过程。通过机器学习算法，研究人员可以预测分子活性、
    优化药物结构、筛选候选药物，显著降低了研发成本和时间。

    ### 2.3 个性化治疗
    基于患者的基因数据、病史和生活习惯，AI可以为每个患者制定个性化的治疗方案。
    精准医疗的发展使得治疗效果显著提高，副作用减少。

    ### 2.4 健康管理
    可穿戴设备和健康监测APP结合AI算法，可以实时监测用户的健康状态，
    预测疾病风险，提供健康建议，实现预防性医疗。

    ## 3. 技术挑战

    ### 3.1 数据隐私与安全
    医疗数据包含敏感的个人隐私信息，如何在利用AI技术的同时保护患者隐私是一个重要挑战。
    联邦学习、差分隐私等技术为解决这一问题提供了可能。

    ### 3.2 模型可解释性
    AI模型的决策过程往往是黑盒，缺乏可解释性。在医疗领域，医生和患者需要理解
    AI的决策依据，这要求提高模型的透明度和可解释性。

    ### 3.3 法规与伦理
    AI在医疗领域的应用需要遵守严格的法规和伦理标准。如何制定合理的监管框架，
    平衡技术创新和患者安全，是政策制定者面临的重要课题。

    ## 4. 未来展望

    随着技术的不断进步，AI在医疗领域的应用将更加深入和广泛。
    我们期待看到更智能的诊断系统、更高效的药物研发流程、更个性化的治疗方案，
    以及更完善的健康管理体系。AI与医疗的融合将为人类健康事业带来革命性的变化。

    ## 5. 结论
    人工智能在医疗领域的应用前景广阔，但也面临诸多挑战。
    通过技术创新、政策引导和多方协作，我们有望克服这些挑战，
    实现AI技术在医疗健康领域的可持续发展。
    """ * 10  # 重复10次以创建长文档
    
    question = "请总结这份报告中提到的主要应用领域和面临的挑战。"
    
    messages = [
        {
            "role": "user",
            "content": f"请基于以下文档回答问题：\n\n文档内容：\n{document}\n\n问题：{question}"
        }
    ]
    
    result = client.chat_with_long_context(
        messages=messages,
        system_prompt="你是一个专业的文档分析助手，擅长从长文档中提取关键信息。",
        temperature=0.5
    )
    
    print("\n" + "="*70)
    print("问答结果")
    print("="*70)
    print(f"压缩信息: {result['compression_info']}")
    print(f"\n问题: {question}")
    print(f"\n回答:\n{result['response']['content']}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("真实大模型API集成示例")
    print("="*70)
    
    # 运行示例
    example_openai_integration()
    example_anthropic_integration()
    example_custom_api()
    example_document_qa()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)
    print("\n使用说明：")
    print("1. 安装依赖: pip install openai anthropic requests")
    print("2. 设置API密钥: 替换代码中的 'your-api-key'")
    print("3. 运行示例: python3 real_llm_integration_example.py")
