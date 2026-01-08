"""
不同模型的压缩服务配置示例
==========================

本示例展示如何根据不同的大模型配置压缩服务，
确保压缩服务与模型的上下文窗口长度正确关联。
"""

from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper


class ModelConfig:
    """模型配置类"""
    
    def __init__(
        self,
        name: str,
        context_length: int,
        max_output_tokens: int,
        description: str = ""
    ):
        self.name = name
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.description = description
    
    def __repr__(self):
        return f"{self.name} (context: {self.context_length}, max_output: {self.max_output_tokens})"


# 常见模型配置
MODEL_CONFIGS = {
    # OpenAI 模型
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        context_length=4096,
        max_output_tokens=4096,
        description="OpenAI的GPT-3.5 Turbo模型"
    ),
    "gpt-4": ModelConfig(
        name="GPT-4",
        context_length=8192,
        max_output_tokens=4096,
        description="OpenAI的GPT-4模型"
    ),
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        context_length=128000,
        max_output_tokens=4096,
        description="OpenAI的GPT-4 Turbo模型"
    ),
    
    # Anthropic 模型
    "claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Opus模型"
    ),
    "claude-3-sonnet": ModelConfig(
        name="Claude 3 Sonnet",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Sonnet模型"
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Haiku模型"
    ),
    
    # Meta 模型
    "llama-2-7b": ModelConfig(
        name="LLaMA 2 7B",
        context_length=4096,
        max_output_tokens=2048,
        description="Meta的LLaMA 2 7B模型"
    ),
    "llama-2-70b": ModelConfig(
        name="LLaMA 2 70B",
        context_length=4096,
        max_output_tokens=2048,
        description="Meta的LLaMA 2 70B模型"
    ),
    "llama-3-8b": ModelConfig(
        name="LLaMA 3 8B",
        context_length=8192,
        max_output_tokens=4096,
        description="Meta的LLaMA 3 8B模型"
    ),
    "llama-3-70b": ModelConfig(
        name="LLaMA 3 70B",
        context_length=8192,
        max_output_tokens=4096,
        description="Meta的LLaMA 3 70B模型"
    ),
    
    # 其他模型
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        context_length=8192,
        max_output_tokens=2048,
        description="Mistral AI的7B模型"
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        context_length=32768,
        max_output_tokens=4096,
        description="Mistral AI的8x7B模型"
    ),
    "qwen-72b": ModelConfig(
        name="Qwen 72B",
        context_length=32768,
        max_output_tokens=2048,
        description="阿里云的Qwen 72B模型"
    ),
    "baichuan-13b": ModelConfig(
        name="Baichuan 13B",
        context_length=4096,
        max_output_tokens=2048,
        description="百度的Baichuan 13B模型"
    ),
}


class CompressionServiceFactory:
    """压缩服务工厂类"""
    
    @staticmethod
    def create_compressor(
        model_name: str,
        max_new_tokens: int = None,
        enable_compression: bool = True,
        use_fast_compression: bool = False,
        safety_margin: float = 0.1
    ) -> DynamicContextCompressor:
        """
        根据模型名称创建压缩器
        
        Args:
            model_name: 模型名称
            max_new_tokens: 最大生成token数（可选，默认使用模型配置）
            enable_compression: 是否启用压缩
            use_fast_compression: 是否使用快速压缩模式
            safety_margin: 安全余量（0-1），默认10%
            
        Returns:
            配置好的压缩器实例
        """
        # 获取模型配置
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型: {model_name}")
        
        config = MODEL_CONFIGS[model_name]
        
        # 计算实际的session_len（减去安全余量）
        session_len = int(config.context_length * (1 - safety_margin))
        
        # 设置max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = config.max_output_tokens
        
        # 创建压缩器
        compressor = DynamicContextCompressor(
            session_len=session_len,
            max_new_tokens=max_new_tokens,
            enable_dynamic_compression=enable_compression,
            use_fast_compression=use_fast_compression
        )
        
        print(f"创建压缩器:")
        print(f"  模型: {config.name}")
        print(f"  原始上下文长度: {config.context_length}")
        print(f"  安全余量: {safety_margin:.0%}")
        print(f"  实际session_len: {session_len}")
        print(f"  max_new_tokens: {max_new_tokens}")
        print(f"  动态压缩: {'启用' if enable_compression else '禁用'}")
        print(f"  快速压缩: {'启用' if use_fast_compression else '禁用'}")
        print()
        
        return compressor
    
    @staticmethod
    def list_models():
        """列出所有支持的模型"""
        print("支持的模型配置:")
        print("=" * 80)
        print(f"{'模型名称':<20} {'上下文长度':<15} {'最大输出':<15} {'描述'}")
        print("=" * 80)
        
        for model_id, config in MODEL_CONFIGS.items():
            print(f"{model_id:<20} {config.context_length:<15} "
                  f"{config.max_output_tokens:<15} {config.description}")
        
        print("=" * 80)


# 使用示例
def example_gpt35_turbo():
    """GPT-3.5 Turbo 示例"""
    print("=" * 60)
    print("示例1: GPT-3.5 Turbo")
    print("=" * 60)
    
    # 创建压缩器
    compressor = CompressionServiceFactory.create_compressor(
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        enable_compression=True,
        safety_margin=0.1  # 10%安全余量
    )
    
    # 模拟长文本
    long_text = "这是一个很长的文本..." * 500
    
    # 压缩文本
    compressed_text, was_compressed = compressor.dynamic_compress(
        text=long_text,
        current_prompt="请总结这段文字",
        max_new_tokens=256
    )
    
    print(f"原始文本: {len(long_text)} 字符")
    print(f"压缩后: {len(compressed_text)} 字符")
    print(f"是否压缩: {was_compressed}")
    print()


def example_gpt4():
    """GPT-4 示例"""
    print("=" * 60)
    print("示例2: GPT-4")
    print("=" * 60)
    
    # 创建压缩器
    compressor = CompressionServiceFactory.create_compressor(
        model_name="gpt-4",
        max_new_tokens=512,
        enable_compression=True,
        safety_margin=0.15  # 15%安全余量
    )
    
    # 模拟聊天历史
    chat_history = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"},
    ]
    
    # 添加更多历史消息
    for i in range(30):
        chat_history.append({
            "role": "user",
            "content": f"这是第{i+1}轮的问题..." * 20
        })
        chat_history.append({
            "role": "assistant",
            "content": f"这是第{i+1}轮的回答..." * 20
        })
    
    # 压缩聊天历史
    compressed_history, was_compressed = compressor.compress_chat_history(
        chat_history=chat_history,
        current_prompt="请总结我们的讨论",
        max_new_tokens=512
    )
    
    print(f"原始消息数: {len(chat_history)}")
    print(f"压缩后消息数: {len(compressed_history)}")
    print(f"是否压缩: {was_compressed}")
    print()


def example_claude3():
    """Claude 3 示例"""
    print("=" * 60)
    print("示例3: Claude 3 Opus")
    print("=" * 60)
    
    # 创建压缩器
    compressor = CompressionServiceFactory.create_compressor(
        model_name="claude-3-opus",
        max_new_tokens=1024,
        enable_compression=True,
        use_fast_compression=True,  # 使用快速压缩
        safety_margin=0.05  # 5%安全余量
    )
    
    # 模拟超长文档
    long_document = "这是一个超长的文档内容..." * 2000
    
    # 压缩文档
    compressed_doc, was_compressed = compressor.dynamic_compress(
        text=long_document,
        current_prompt="请分析这个文档",
        max_new_tokens=1024
    )
    
    print(f"原始文档: {len(long_document)} 字符")
    print(f"压缩后: {len(compressed_doc)} 字符")
    print(f"是否压缩: {was_compressed}")
    print()


def example_llama3():
    """LLaMA 3 示例"""
    print("=" * 60)
    print("示例4: LLaMA 3 8B")
    print("=" * 60)
    
    # 创建压缩器
    compressor = CompressionServiceFactory.create_compressor(
        model_name="llama-3-8b",
        max_new_tokens=512,
        enable_compression=True,
        safety_margin=0.1
    )
    
    # 模拟代码分析
    code_content = """
def example_function():
    # 这是一个示例函数
    # 包含很多注释
    for i in range(100):
        # 循环处理
        result = process(i)
        # 保存结果
        save_result(result)
    return result
""" * 50
    
    # 压缩代码
    compressed_code, was_compressed = compressor.dynamic_compress(
        text=code_content,
        current_prompt="请分析这段代码",
        max_new_tokens=512
    )
    
    print(f"原始代码: {len(code_content)} 字符")
    print(f"压缩后: {len(compressed_code)} 字符")
    print(f"是否压缩: {was_compressed}")
    print()


def example_custom_config():
    """自定义配置示例"""
    print("=" * 60)
    print("示例5: 自定义配置")
    print("=" * 60)
    
    # 直接创建压缩器（不使用工厂）
    compressor = DynamicContextCompressor(
        session_len=4096,  # 自定义上下文长度
        max_new_tokens=256,  # 自定义最大输出
        enable_dynamic_compression=True,
        use_fast_compression=False
    )
    
    print(f"自定义配置:")
    print(f"  session_len: {compressor.session_len}")
    print(f"  max_new_tokens: {compressor.max_new_tokens}")
    print(f"  enable_dynamic_compression: {compressor.enable_dynamic_compression}")
    print(f"  use_fast_compression: {compressor.use_fast_compression}")
    print()


def example_list_models():
    """列出所有支持的模型"""
    print("=" * 60)
    print("示例6: 列出所有支持的模型")
    print("=" * 60)
    print()
    
    CompressionServiceFactory.list_models()
    print()


def example_compression_trigger():
    """演示压缩触发机制"""
    print("=" * 60)
    print("示例7: 压缩触发机制演示")
    print("=" * 60)
    
    # 创建压缩器
    compressor = CompressionServiceFactory.create_compressor(
        model_name="gpt-3.5-turbo",
        max_new_tokens=256,
        enable_compression=True
    )
    
    # 测试不同长度的文本
    test_cases = [
        ("短文本", "这是一个短文本"),
        ("中等文本", "这是一个中等长度的文本..." * 50),
        ("长文本", "这是一个很长的文本..." * 500),
        ("超长文本", "这是一个超长的文本..." * 2000),
    ]
    
    for name, text in test_cases:
        prompt_tokens = compressor.tokenizer.count_tokens("请总结")
        text_tokens = compressor.tokenizer.count_tokens(text)
        max_new_tokens = 256
        total_tokens = prompt_tokens + text_tokens + max_new_tokens
        
        print(f"\n测试: {name}")
        print(f"  提示词tokens: {prompt_tokens}")
        print(f"  文本tokens: {text_tokens}")
        print(f"  max_new_tokens: {max_new_tokens}")
        print(f"  总tokens: {total_tokens}")
        print(f"  session_len: {compressor.session_len}")
        
        if total_tokens > compressor.session_len:
            print(f"  状态: 需要压缩")
            available_tokens = compressor.session_len - prompt_tokens - max_new_tokens
            print(f"  可用tokens: {available_tokens}")
        else:
            print(f"  状态: 无需压缩")
    
    print()


if __name__ == "__main__":
    # 运行所有示例
    example_gpt35_turbo()
    example_gpt4()
    example_claude3()
    example_llama3()
    example_custom_config()
    example_list_models()
    example_compression_trigger()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
