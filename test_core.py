from core.dynamic_compressor import DynamicContextCompressor
from api.tokenizer_wrapper import TokenizerWrapper

def test_text_compression():
    print("Testing text compression...")
    
    tokenizer = TokenizerWrapper()
    compressor = DynamicContextCompressor(tokenizer=tokenizer)
    
    sample_text = """这是一个测试文本，用于验证动态文本压缩功能是否正常工作。

在这个测试中，我们将创建一个比较长的文本，包含多个段落和重要信息。

首先，我们需要确认压缩器能够正确识别重要的关键词，比如"重要"、"关键"、"结论"等。

其次，我们需要验证压缩器能够根据会话窗口大小进行动态压缩，保留最重要的信息。

最后，我们需要测试压缩器的性能和准确性，确保压缩后的文本仍然包含核心内容。

这个测试对于验证整个系统的可靠性非常重要，因为动态压缩是大模型上下文管理的关键技术之一。

我们希望压缩器能够在保持信息完整性的同时，有效地减少文本长度，以便在有限的上下文窗口中容纳更多信息。

总之，这个测试将全面验证动态文本压缩的各项功能和性能指标。"""
    
    print(f"Original text length: {tokenizer.count_tokens(sample_text)} tokens")
    
    compressor.session_len = 100
    compressed_text, was_compressed = compressor.dynamic_compress(
        text=sample_text,
        current_prompt="请总结以下文本：",
        max_new_tokens=500
    )
    
    print(f"Compressed text length: {tokenizer.count_tokens(compressed_text)} tokens")
    print(f"Was compressed: {was_compressed}")
    print("\nCompressed text:")
    print(compressed_text)
    print("\n" + "="*50 + "\n")

def test_chat_compression():
    print("Testing chat history compression...")
    
    tokenizer = TokenizerWrapper()
    compressor = DynamicContextCompressor(tokenizer=tokenizer)
    
    sample_messages = [
        {"role": "system", "content": "你是一个智能助手，帮助用户解决问题。"},
        {"role": "user", "content": "你好，我想了解如何使用Python进行文本处理。"},
        {"role": "assistant", "content": "Python有很多用于文本处理的库，比如re、nltk、spacy等。re模块用于正则表达式匹配，nltk是自然语言处理工具包，spacy是更现代的NLP库，提供了很多预训练模型。"},
        {"role": "user", "content": "能详细介绍一下正则表达式的使用吗？"},
        {"role": "assistant", "content": "正则表达式是一种用于匹配字符串模式的工具。在Python中，可以使用re模块来使用正则表达式。常用的函数包括re.match、re.search、re.findall、re.sub等。例如，要匹配所有邮箱地址，可以使用r'\w+@\w+\.\w+'这样的模式。"},
        {"role": "user", "content": "好的，谢谢。那nltk库呢？"},
        {"role": "assistant", "content": "NLTK（Natural Language Toolkit）是Python的一个自然语言处理库，提供了很多用于文本分析的工具。它包含了语料库、词法分析、句法分析、语义分析等功能。要使用NLTK，首先需要安装它，然后下载需要的语料库。例如，可以使用nltk.download('punkt')来下载分词器，使用nltk.tokenize.word_tokenize来分词。"},
        {"role": "user", "content": "非常感谢，最后问一下spacy的优势是什么？"},
        {"role": "assistant", "content": "spaCy是一个工业级的NLP库，相比NLTK，它的速度更快，更适合处理大规模文本。它提供了预训练的词向量、命名实体识别、依存句法分析等功能。spaCy的API设计更现代化，使用起来更方便。例如，加载一个英文模型只需要import spacy然后nlp = spacy.load('en_core_web_sm')，然后就可以使用nlp对象来处理文本了。"},
        {"role": "user", "content": "现在我想了解如何在Python中进行文件操作。"}
    ]
    
    print(f"Original messages count: {len(sample_messages)}")
    
    compressor.session_len = 150
    compressed_messages, was_compressed = compressor.compress_chat_history(
        messages=sample_messages,
        current_prompt="",
        max_new_tokens=1000
    )
    
    print(f"Compressed messages count: {len(compressed_messages)}")
    print(f"Was compressed: {was_compressed}")
    print("\nCompressed messages:")
    for msg in compressed_messages:
        print(f"{msg['role']}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{msg['role']}: {msg['content']}")

if __name__ == "__main__":
    test_text_compression()
    test_chat_compression()
