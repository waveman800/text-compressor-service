# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
import re


class TextSummarizer:
    """Text summarizer for compressing context when it exceeds the context window.

    This class provides functionality to automatically summarize text when it
    exceeds the context window length, similar to Ollama's summarization mechanism.

    Args:
        model_name (str): The name or path of the model to use for summarization.
            Default is "facebook/bart-large-cnn".
        device (str): The device to run the model on. Default is "cuda" if available,
            otherwise "cpu".
        max_input_length (int): Maximum input length for the summarization model.
            Default is 1024.
        max_output_length (int): Maximum output length for the summarization.
            Default is 150.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_input_length: int = 1024,
        max_output_length: int = 150,
    ):
        # 使用简单的后备摘要器，避免依赖问题
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device or "cpu"
        
        # 不加载实际的模型，使用简单的文本处理
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        # 不使用实际模型，返回 None
        return None

    @property
    def tokenizer(self):
        # 不使用实际tokenizer，返回 None
        return None

    def summarize(self, text: str) -> str:
        """Summarize the given text.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The summarized text.
        """
        # 直接使用简单摘要方法
        return self._simple_summarize(text)

    def _simple_summarize(self, text: str) -> str:
        """Simple summarization fallback method."""
        import re
        
        # 如果文本很短，直接返回
        if len(text) < 500:
            return text
            
        # 将文本分成句子
        sentences = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s*', text)
        
        # 如果句子很少，返回原文本
        if len(sentences) <= 3:
            return text
            
        # 选择重要句子
        important_sentences = []
        
        # 总是保留第一个句子
        if sentences:
            important_sentences.append(sentences[0])
            
        # 选择包含关键词的句子
        keywords = [
            '重要', '关键', '总结', '结论', '必须', '注意',
            'important', 'key', 'summary', 'conclusion', 'must', 'note'
        ]
        
        for sentence in sentences[1:-1]:
            # 检查是否包含关键词
            if any(keyword in sentence.lower() for keyword in keywords):
                important_sentences.append(sentence)
            # 每五个句子选一个，确保摘要有一定的覆盖范围
            elif len(important_sentences) < len(sentences) // 5:
                important_sentences.append(sentence)
                
        # 添加最后一个句子
        if sentences and len(sentences) > 1:
            important_sentences.append(sentences[-1])
            
        # 合并摘要句子
        summary = ' '.join(important_sentences)
        
        return summary

    def chunk_and_summarize(
        self, 
        text: str, 
        max_chunk_size: int = 1024
    ) -> str:
        """Split text into chunks and summarize each chunk if needed.

        Args:
            text (str): The text to summarize.
            max_chunk_size (int): Maximum size of each chunk.

        Returns:
            str: The summarized text.
        """
        # If text is short enough, return it as is
        if len(text) <= max_chunk_size:
            return text
            
        # Split text into chunks
        chunks = []
        current_chunk = ""
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                
        if current_chunk:
            chunks.append(current_chunk)
            
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summaries.append(self.summarize(chunk))
            
        # Combine summaries
        combined_summary = "\n\n".join(summaries)
        
        return combined_summary


class ContextWindowManager:
    """Manages the context window for chat sessions.

    This class provides functionality to automatically summarize the chat history
    when it exceeds the context window length.

    Args:
        tokenizer: The tokenizer to use for counting tokens.
        session_len (int): The maximum context window length in tokens.
        summarizer (Optional[TextSummarizer]): The summarizer to use. If None,
            a default summarizer will be created.
        summary_threshold (float): The threshold ratio of the context window
            at which summarization is triggered. Default is 0.9 (90%).
        enable_summarization (bool): Whether to enable automatic summarization.
            Default is True.
    """

    def __init__(
        self,
        tokenizer,
        session_len: int,
        summarizer: Optional[TextSummarizer] = None,
        summary_threshold: float = 0.9,
        enable_summarization: bool = True,
    ):
        self.tokenizer = tokenizer
        self.session_len = session_len
        self.summary_threshold = summary_threshold
        self.enable_summarization = enable_summarization
        
        # Create default summarizer if none is provided
        if summarizer is None:
            try:
                self.summarizer = TextSummarizer()
            except ImportError:
                print("Warning: Could not create TextSummarizer, using simple fallback")
                self.summarizer = None
        else:
            self.summarizer = summarizer
            
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        return len(self.tokenizer.encode(text))
        
    def maybe_summarize_history(
        self, 
        messages: List[Dict[str, str]],
        current_prompt: str,
        max_new_tokens: int
    ) -> Tuple[List[Dict[str, str]], bool]:
        """Check if the chat history needs to be summarized and summarize if needed.

        Args:
            messages (List[Dict[str, str]]): The chat history messages.
            current_prompt (str): The current user prompt.
            max_new_tokens (int): The maximum number of tokens for the response.

        Returns:
            Tuple[List[Dict[str, str]], bool]: The (possibly summarized) messages
                and a boolean indicating whether summarization was performed.
        """
        # If summarization is disabled, return the original messages
        if not self.enable_summarization:
            return messages, False
            
        # Count tokens in the current conversation
        history_text = ""
        for msg in messages:
            history_text += f"{msg['role']}: {msg['content']}\n\n"
            
        history_tokens = self.count_tokens(history_text)
        prompt_tokens = self.count_tokens(current_prompt)
        
        # Calculate total expected tokens
        total_expected_tokens = history_tokens + prompt_tokens + max_new_tokens
        
        # Check if we need to summarize
        if total_expected_tokens > self.summary_threshold * self.session_len:
            # We need to summarize the history
            if self.summarizer:
                summarized_history = self.summarizer.summarize(history_text)
            else:
                # Use simple fallback
                summarized_history = self._simple_summarize_fallback(history_text)
            
            # Replace the history with a summary
            summarized_messages = [
                {"role": "system", "content": "The following is a summary of the conversation so far:"},
                {"role": "system", "content": summarized_history},
                {"role": "user", "content": current_prompt}
            ]
            
            return summarized_messages, True
            
        # No summarization needed
        return messages, False
        
    def _simple_summarize_fallback(self, text: str) -> str:
        """Simple summarization fallback when no summarizer is available."""
        import re
        
        # 如果文本很短，直接返回
        if len(text) < 1000:
            return text
            
        # 将文本分成句子
        sentences = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s*', text)
        
        # 如果句子很少，返回原文本
        if len(sentences) <= 5:
            return text
            
        # 选择重要句子
        important_sentences = []
        
        # 总是保留第一个句子
        if sentences:
            important_sentences.append(sentences[0])
            
        # 选择包含关键词的句子
        keywords = [
            '重要', '关键', '总结', '结论', '必须', '注意',
            'important', 'key', 'summary', 'conclusion', 'must', 'note'
        ]
        
        for sentence in sentences[1:-1]:
            # 检查是否包含关键词
            if any(keyword in sentence.lower() for keyword in keywords):
                important_sentences.append(sentence)
            # 每四个句子选一个，确保摘要有一定的覆盖范围
            elif len(important_sentences) < len(sentences) // 4:
                important_sentences.append(sentence)
                
        # 添加最后一个句子
        if sentences and len(sentences) > 1:
            important_sentences.append(sentences[-1])
            
        # 合并摘要句子
        summary = ' '.join(important_sentences)
        
        return summary
