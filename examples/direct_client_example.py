from core.dynamic_compressor import DynamicContextCompressor
from api.tokenizer_wrapper import TokenizerWrapper

def main():
    print("Dynamic Text Compressor - Direct Client Example")
    print("=" * 50)
    
    # Initialize tokenizer and compressor
    print("Initializing tokenizer and compressor...")
    tokenizer = TokenizerWrapper()
    compressor = DynamicContextCompressor(
        session_len=4096,
        tokenizer=tokenizer,
        enable_dynamic_compression=True,
        use_fast_compression=False,
        segment_strategy="paragraph"
    )
    
    print(f"Tokenizer type: {'Hugging Face' if tokenizer.use_huggingface else 'Character-based fallback'}")
    print("=" * 50)
    
    # Example 1: Text Compression
    print("\n1. Text Compression Example:")
    sample_text = """这是一个关于人工智能的研究报告。

人工智能（AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。AI的应用范围非常广泛，包括自然语言处理、计算机视觉、机器学习等多个领域。

在自然语言处理领域，AI模型可以理解和生成人类语言，实现机器翻译、文本摘要、情感分析等功能。例如，大型语言模型（LLM）如GPT、BERT等能够处理复杂的语言任务。

在计算机视觉领域，AI可以识别图像中的物体、场景和人物，应用于自动驾驶、面部识别、医学影像分析等领域。

机器学习是AI的核心技术之一，它使计算机能够从数据中学习模式，而不需要显式编程。监督学习、无监督学习和强化学习是机器学习的主要类型。

总之，人工智能正在改变我们的生活和工作方式，带来了巨大的机遇和挑战。未来，AI技术将继续发展，为人类社会创造更多价值。"""
    
    print(f"Original text: {tokenizer.count_tokens(sample_text)} tokens")
    
    # Compress with custom parameters
    compressor.session_len = 150
    compressed_text, was_compressed = compressor.dynamic_compress(
        text=sample_text,
        current_prompt="请总结以下关于人工智能的报告：",
        max_new_tokens=100
    )
    
    print(f"Compressed text: {tokenizer.count_tokens(compressed_text)} tokens")
    print(f"Was compressed: {was_compressed}")
    print("Compressed content:")
    print(compressed_text)
    
    # Example 2: Chat History Compression
    print("\n2. Chat History Compression Example:")
    
    chat_messages = [
        {"role": "system", "content": "你是一个AI助手，帮助用户学习编程。"},
        {"role": "user", "content": "你好，我想学习Python编程，应该从哪里开始？"},
        {"role": "assistant", "content": "学习Python编程的最佳起点是了解基本语法，包括变量、数据类型、条件语句、循环等。你可以通过在线教程如Codecademy、Coursera或官方文档来学习。推荐使用Anaconda作为开发环境，它包含了很多常用的库。"},
        {"role": "user", "content": "Python中的列表和字典有什么区别？"},
        {"role": "assistant", "content": "列表和字典是Python中两种常用的数据结构。列表是有序的，可以通过索引访问元素，使用方括号[]定义。字典是无序的，通过键值对存储数据，使用大括号{}定义。列表中的元素可以是任意类型，而字典的键必须是不可变类型（如字符串、数字、元组）。"},
        {"role": "user", "content": "如何安装第三方库？"},
        {"role": "assistant", "content": "在Python中，你可以使用pip来安装第三方库。pip是Python的包管理器，可以从PyPI（Python Package Index）下载和安装库。使用命令'pip install 库名'即可安装。例如，要安装numpy库，可以运行'pip install numpy'。"},
        {"role": "user", "content": "如何处理文件？"},
        {"role": "assistant", "content": "Python提供了内置的open函数来处理文件。你可以使用'with open("文件名", "模式") as f:'的语法来打开文件，这样可以确保文件正确关闭。常见的模式有'r'（只读）、'w'（写入）、'a'（追加）等。例如，要读取文本文件，可以使用'with open("file.txt", "r") as f: content = f.read()'。"},
        {"role": "user", "content": "什么是面向对象编程？"},
        {"role": "assistant", "content": "面向对象编程（OOP）是一种编程范式，它使用对象和类来组织代码。类是对象的蓝图，定义了对象的属性和方法。对象是类的实例。OOP的主要概念包括封装、继承和多态。封装将数据和操作数据的方法绑定在一起，继承允许创建新类继承现有类的属性和方法，多态允许不同类的对象对同一消息做出不同的响应。"},
        {"role": "user", "content": "如何使用Python进行数据分析？"},
        {"role": "assistant", "content": "Python是数据分析的强大工具，常用的库包括numpy（用于数值计算）、pandas（用于数据处理和分析）、matplotlib和seaborn（用于数据可视化）。pandas提供了DataFrame结构，用于处理结构化数据。你可以使用pandas读取CSV、Excel等格式的数据，进行数据清洗、转换和分析，然后使用matplotlib或seaborn创建图表可视化结果。"},
        {"role": "user", "content": "现在我想学习机器学习，应该怎么做？"}
    ]
    
    print(f"Original chat history: {len(chat_messages)} messages")
    
    # Compress chat history
    compressor.session_len = 200
    compressed_messages, was_compressed = compressor.compress_chat_history(
        messages=chat_messages,
        current_prompt="",
        max_new_tokens=100
    )
    
    print(f"Compressed chat history: {len(compressed_messages)} messages")
    print(f"Was compressed: {was_compressed}")
    print("Compressed messages:")
    for i, msg in enumerate(compressed_messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        print(f"  {i+1}. {role}: {content[:100]}{'...' if len(content) > 100 else ''}")
    
    print("\n=" * 50)
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
