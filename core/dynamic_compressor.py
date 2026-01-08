import re
import json
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from .tokenizer_wrapper import TokenizerWrapper

@dataclass
class TextSegment:
    """文本段落"""
    content: str
    importance: float
    token_count: int
    is_keyword_segment: bool = False

@dataclass
class CompressionMetrics:
    """压缩质量评估指标"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    keyword_retention: float
    information_preservation: float
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_tokens': self.original_tokens,
            'compressed_tokens': self.compressed_tokens,
            'compression_ratio': self.compression_ratio,
            'keyword_retention': self.keyword_retention,
            'information_preservation': self.information_preservation,
            'processing_time': self.processing_time
        }

class DynamicContextCompressor:
    """
    动态文本压缩器，基于lmdeploy 0.8.0的动态压缩逻辑
    """
    def __init__(self,
                 session_len: int = 4096,
                 max_new_tokens: int = 256,
                 tokenizer: TokenizerWrapper = None,
                 enable_dynamic_compression: bool = True,
                 use_fast_compression: bool = False,
                 segment_strategy: str = "paragraph"):
        """
        初始化动态文本压缩器
        
        Args:
            session_len: 会话窗口的最大token数
            max_new_tokens: 生成新token的最大数量
            tokenizer: 分词器实例
            enable_dynamic_compression: 是否启用动态压缩
            use_fast_compression: 是否使用快速压缩模式
            segment_strategy: 文本分割策略，"paragraph"或"sentence"
        """
        self.session_len = session_len
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer or TokenizerWrapper()
        self.enable_dynamic_compression = enable_dynamic_compression
        self.use_fast_compression = use_fast_compression
        self.segment_strategy = segment_strategy
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 关键关键词列表
        self.keywords = [
            "结论", "总结", "因此", "所以", "重要", "关键", "注意",
            "必须", "需要", "建议", "方案", "解决", "结果", "发现",
            "问题", "原因", "措施", "目标", "计划", "步骤", "实现",
            "important", "key", "note", "must", "need", "should", "suggest",
            "result", "find", "problem", "reason", "solution", "goal", "plan"
        ]
    
    async def dynamic_compress_async(self, text: str, current_prompt: str = "", max_new_tokens: int = 0) -> Tuple[str, bool]:
        """
        异步版本的动态文本压缩
        
        Args:
            text: 要压缩的文本
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            Tuple[str, bool]: (压缩后的文本, 是否进行了压缩)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.dynamic_compress,
            text,
            current_prompt,
            max_new_tokens
        )
    
    async def compress_chat_history_async(self, chat_history: List[Dict[str, Any]], current_prompt: str = "", max_new_tokens: int = 0) -> Tuple[List[Dict[str, Any]], bool]:
        """
        异步版本的聊天历史压缩
        
        Args:
            chat_history: 聊天历史列表
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            Tuple[List[Dict[str, Any]], bool]: (压缩后的聊天历史, 是否进行了压缩)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.compress_chat_history,
            chat_history,
            current_prompt,
            max_new_tokens
        )
    
    async def batch_compress(self, texts: List[str], current_prompt: str = "", max_new_tokens: int = 0) -> List[Tuple[str, bool]]:
        """
        批量压缩多个文本
        
        Args:
            texts: 要压缩的文本列表
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            List[Tuple[str, bool]]: 每个文本的压缩结果
        """
        tasks = [
            self.dynamic_compress_async(text, current_prompt, max_new_tokens)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    async def batch_compress_chat_histories(self, chat_histories: List[List[Dict[str, Any]]], current_prompt: str = "", max_new_tokens: int = 0) -> List[Tuple[List[Dict[str, Any]], bool]]:
        """
        批量压缩多个聊天历史
        
        Args:
            chat_histories: 聊天历史列表
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            List[Tuple[List[Dict[str, Any]], bool]]: 每个聊天历史的压缩结果
        """
        tasks = [
            self.compress_chat_history_async(history, current_prompt, max_new_tokens)
            for history in chat_histories
        ]
        return await asyncio.gather(*tasks)
    
    def close(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
    
    def evaluate_compression(self, original_text: str, compressed_text: str, processing_time: float = 0.0) -> CompressionMetrics:
        """
        评估压缩质量
        
        Args:
            original_text: 原始文本
            compressed_text: 压缩后的文本
            processing_time: 处理时间（秒）
            
        Returns:
            CompressionMetrics: 压缩质量指标
        """
        original_tokens = self.tokenizer.count_tokens(original_text)
        compressed_tokens = self.tokenizer.count_tokens(compressed_text)
        
        # 压缩比
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0.0
        
        # 关键词保留率
        original_keywords = sum(1 for keyword in self.keywords if keyword.lower() in original_text.lower())
        compressed_keywords = sum(1 for keyword in self.keywords if keyword.lower() in compressed_text.lower())
        keyword_retention = compressed_keywords / original_keywords if original_keywords > 0 else 1.0
        
        # 信息保留率（基于重要性评分）
        original_importance = self.calculate_importance(original_text)
        compressed_importance = self.calculate_importance(compressed_text)
        information_preservation = compressed_importance / original_importance if original_importance > 0 else 1.0
        
        return CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            keyword_retention=keyword_retention,
            information_preservation=information_preservation,
            processing_time=processing_time
        )
    
    def compress_with_metrics(self, text: str, current_prompt: str = "", max_new_tokens: int = 0) -> Tuple[str, bool, CompressionMetrics]:
        """
        压缩文本并返回质量指标
        
        Args:
            text: 要压缩的文本
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            Tuple[str, bool, CompressionMetrics]: (压缩后的文本, 是否进行了压缩, 质量指标)
        """
        import time
        start_time = time.time()
        
        compressed_text, was_compressed = self.dynamic_compress(text, current_prompt, max_new_tokens)
        
        processing_time = time.time() - start_time
        metrics = self.evaluate_compression(text, compressed_text, processing_time)
        
        return compressed_text, was_compressed, metrics
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成段落或句子
        """
        if not text:
            return []
        
        if self.segment_strategy == "paragraph":
            # 按段落分割
            paragraphs = re.split(r'\n\s*\n', text)
            return [p.strip() for p in paragraphs if p.strip()]
        else:  # sentence
            # 按句子分割
            sentences = re.split(r'[。！？.!?]', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def detect_text_type(self, text: str) -> str:
        """
        检测文本类型，用于选择最佳压缩策略
        
        Returns:
            'dialogue': 对话类型（包含大量引号、对话标记）
            'list': 列表类型（包含大量编号、项目符号）
            'code': 代码类型（包含代码块、特殊符号）
            'narrative': 叙述类型（普通文本）
        """
        if not text:
            return 'narrative'
        
        # 检测对话类型
        dialogue_patterns = [
            r'["「『][^"」』]*["」』]',  # 引号
            r'说：|道：|问：|答：',  # 对话标记
            r'\n\s*[A-Z][a-z]*:',  # 英文对话格式
        ]
        dialogue_count = sum(len(re.findall(p, text)) for p in dialogue_patterns)
        if dialogue_count > 3:
            return 'dialogue'
        
        # 检测列表类型
        list_patterns = [
            r'^\s*[\d\-\*•]+\s+',  # 编号或项目符号
            r'^\s*\[[^\]]+\]',  # 方括号列表
        ]
        list_count = sum(len(re.findall(p, text, re.MULTILINE)) for p in list_patterns)
        if list_count > 3:
            return 'list'
        
        # 检测代码类型
        code_patterns = [
            r'```[\s\S]*?```',  # 代码块
            r'function\s+\w+\s*\(',  # 函数定义
            r'class\s+\w+\s*:',  # 类定义
            r'def\s+\w+\s*\(',  # Python函数
            r'import\s+\w+',  # 导入语句
        ]
        code_count = sum(len(re.findall(p, text)) for p in code_patterns)
        if code_count > 1:
            return 'code'
        
        return 'narrative'
    
    def calculate_importance(self, text: str, position: int = 0, total_segments: int = 1) -> float:
        """
        计算文本的重要性分数（0-1）
        结合关键词匹配、简化的TF-IDF分析、位置权重和长度权重
        
        Args:
            text: 文本内容
            position: 段落在全文中的位置（0-based）
            total_segments: 总段落数
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # 1. 关键词匹配分数
        matched_keywords = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                matched_keywords += 1
        
        if matched_keywords > 0:
            keyword_score = min(matched_keywords / 2, 1.0)
        else:
            keyword_score = 0.0
        
        # 2. 简化的TF-IDF分数
        words = re.findall(r'\w+', text_lower)
        if not words:
            tf_idf_score = 0.0
        else:
            word_count = {}
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
            
            max_tf = max(word_count.values())
            tf_idf_score = (max_tf / len(words)) * 0.5
        
        # 3. 位置权重：开头和结尾的段落更重要
        if total_segments <= 1:
            position_weight = 1.0
        else:
            # 使用正弦函数给开头和结尾更高的权重
            normalized_pos = position / (total_segments - 1)
            position_weight = 0.7 + 0.3 * (1 - abs(normalized_pos - 0.5) * 2)
        
        # 4. 长度权重：适中的长度更重要，太短或太长都降低权重
        text_length = len(text)
        if text_length < 50:
            length_weight = 0.5  # 太短
        elif text_length < 200:
            length_weight = 1.0  # 适中
        elif text_length < 500:
            length_weight = 0.8  # 较长
        else:
            length_weight = 0.6  # 太长
        
        # 综合评分
        content_score = keyword_score * 0.7 + tf_idf_score * 0.3
        total_score = content_score * 0.6 + position_weight * 0.25 + length_weight * 0.15
        
        # 如果包含关键词，增加基础分数
        if matched_keywords > 0:
            total_score = max(total_score, 0.6)
        
        return min(total_score, 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（使用多种算法混合）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数（0-1），1表示完全相同
        """
        if not text1 or not text2:
            return 0.0
        
        text1_clean = text1.strip()
        text2_clean = text2.strip()
        
        if text1_clean == text2_clean:
            return 1.0
        
        char_similarity = self._calculate_character_similarity(text1_clean, text2_clean)
        
        word_similarity = self._calculate_word_similarity_chinese(text1_clean, text2_clean)
        
        keyword_similarity = self._calculate_keyword_similarity(text1_clean, text2_clean)
        
        combined_similarity = char_similarity * 0.3 + word_similarity * 0.5 + keyword_similarity * 0.2
        
        return combined_similarity
    
    def _calculate_character_similarity(self, text1: str, text2: str) -> float:
        """
        计算字符级别的相似度（适合中文）
        """
        if not text1 or not text2:
            return 0.0
        
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        jaccard_char = intersection / union if union > 0 else 0
        
        len1 = len(text1)
        len2 = len(text2)
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        total_chars = len1 + len2
        if total_chars > 0:
            common_ratio = intersection * 2 / total_chars
        else:
            common_ratio = 0
        
        char_similarity = jaccard_char * 0.5 + len_ratio * 0.3 + common_ratio * 0.2
        
        return char_similarity
    
    def _calculate_word_similarity_chinese(self, text1: str, text2: str) -> float:
        """
        计算词级别的相似度（中文优化版）
        """
        try:
            import jieba
            words1 = set(jieba.cut(text1))
            words2 = set(jieba.cut(text2))
            
            words1 = {w.strip() for w in words1 if w.strip() and len(w.strip()) > 1}
            words2 = {w.strip() for w in words2 if w.strip() and len(w.strip()) > 1}
            
            if not words1 or not words2:
                return self._calculate_char_ngram_similarity(text1, text2)
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            jaccard = intersection / union if union > 0 else 0
            
            if intersection > 0:
                avg_overlap = sum(1 for w in words1 if w in words2) / len(words1)
            else:
                avg_overlap = 0
            
            similarity = jaccard * 0.6 + avg_overlap * 0.4
            
            return similarity
        except ImportError:
            return self._calculate_char_ngram_similarity(text1, text2)
    
    def _calculate_char_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        计算字符n-gram相似度（无jieba时的备选方案）
        """
        if not text1 or not text2:
            return 0.0
        
        ngrams1 = set(text1[i:i+n] for i in range(len(text1) - n + 1))
        ngrams2 = set(text2[i:i+n] for i in range(len(text2) - n + 1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard_ngram = intersection / union if union > 0 else 0
        
        return jaccard_ngram
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        计算关键词重叠度
        """
        if not text1 or not text2:
            return 0.0
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        matched_keywords1 = sum(1 for kw in self.keywords if kw.lower() in text1_lower)
        matched_keywords2 = sum(1 for kw in self.keywords if kw.lower() in text2_lower)
        
        if matched_keywords1 == 0 and matched_keywords2 == 0:
            return 0.5
        
        if matched_keywords1 == 0 or matched_keywords2 == 0:
            return 0.0
        
        common_keywords = sum(1 for kw in self.keywords if kw.lower() in text1_lower and kw.lower() in text2_lower)
        
        min_matched = min(matched_keywords1, matched_keywords2)
        keyword_similarity = common_keywords / min_matched if min_matched > 0 else 0
        
        return keyword_similarity
    
    def _deduplicate_with_similarity(self, segments: List[str], similarity_threshold: float = 0.55) -> List[str]:
        """
        使用相似度检测进行去重
        
        Args:
            segments: 文本段落列表
            similarity_threshold: 相似度阈值（0-1），超过此值认为相似
            
        Returns:
            List[str]: 去重后的段落列表
        """
        if not segments:
            return []
        
        unique_segments = []
        seen_texts = []
        
        for segment in segments:
            is_duplicate = False
            
            for seen_text in seen_texts:
                similarity = self._calculate_text_similarity(segment, seen_text)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_segments.append(segment)
                seen_texts.append(segment)
        
        return unique_segments
    
    def fast_compress_to_fit(self, text: str, available_tokens: int) -> str:
        """
        快速压缩文本以适应可用的token数
        """
        if not text or available_tokens <= 0:
            return ""
        
        # 如果文本已经足够短，直接返回
        current_tokens = self.tokenizer.count_tokens(text)
        if current_tokens <= available_tokens:
            return text
        
        # 计算压缩比例，增加安全系数
        compression_ratio = available_tokens / current_tokens * 0.8  # 80%的安全系数
        
        # 按段落分割
        paragraphs = self.split_text(text)
        if not paragraphs:
            return ""
        
        compressed_text = ""
        total_compressed_tokens = 0
        
        # 对每个段落进行压缩
        for para in paragraphs:
            para_tokens = self.tokenizer.count_tokens(para)
            target_tokens = max(1, int(para_tokens * compression_ratio))
            
            # 压缩段落
            compressed_para = self.compress_segment(para, target_tokens)
            compressed_para_tokens = self.tokenizer.count_tokens(compressed_para)
            
            # 检查是否超出限制
            if total_compressed_tokens + compressed_para_tokens <= available_tokens:
                compressed_text += compressed_para + "\n\n"
                total_compressed_tokens += compressed_para_tokens
            else:
                # 如果添加这个段落后超过限制，尝试压缩到剩余空间
                remaining_tokens = available_tokens - total_compressed_tokens
                if remaining_tokens > 0:
                    compressed_para = self.compress_segment(para, remaining_tokens)
                    compressed_para_tokens = self.tokenizer.count_tokens(compressed_para)
                    if compressed_para_tokens <= remaining_tokens:
                        compressed_text += compressed_para
                        total_compressed_tokens += compressed_para_tokens
                break
        
        # 最终检查，如果还是超出限制，进一步压缩
        if self.tokenizer.count_tokens(compressed_text) > available_tokens:
            # 直接截取到目标token数
            compressed_text = self.compress_segment(compressed_text, available_tokens)
        
        return compressed_text.strip()
    
    def compress_segment(self, text: str, max_tokens: int) -> str:
        """
        压缩单个文本段落到指定的最大token数
        使用二分查找优化压缩过程，减少token计算次数
        """
        if not text or max_tokens <= 0:
            return ""
        
        current_tokens = self.tokenizer.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # 使用二分查找找到合适的截断位置
        left, right = 0, len(text)
        best_text = ""
        
        while left <= right:
            mid = (left + right) // 2
            candidate = text[:mid]
            
            # 如果候选文本为空，继续向右搜索
            if not candidate:
                left = mid + 1
                continue
            
            candidate_tokens = self.tokenizer.count_tokens(candidate)
            
            if candidate_tokens <= max_tokens:
                best_text = candidate
                left = mid + 1
            else:
                right = mid - 1
        
        # 如果没有找到合适的文本，返回空字符串
        if not best_text:
            return ""
        
        # 确保文本以完整的句子结束
        end_idx = max(
            best_text.rfind('。'),
            best_text.rfind('！'),
            best_text.rfind('？'),
            best_text.rfind('.') if '.' in best_text else -1,
            best_text.rfind('!') if '!' in best_text else -1,
            best_text.rfind('?') if '?' in best_text else -1
        )
        
        if end_idx != -1 and end_idx < len(best_text) - 1:
            best_text = best_text[:end_idx + 1]
        else:
            best_text = best_text.strip()
            if best_text and not best_text.endswith(('。', '！', '？', '.', '!', '?')):
                best_text += '。'
        
        # 最终验证token数
        final_tokens = self.tokenizer.count_tokens(best_text)
        if final_tokens > max_tokens:
            # 如果仍然超出，使用更激进的截断
            target_ratio = max_tokens / final_tokens * 0.95
            target_length = int(len(best_text) * target_ratio)
            best_text = best_text[:target_length]
            # 再次确保以完整句子结束
            end_idx = max(
                best_text.rfind('。'),
                best_text.rfind('！'),
                best_text.rfind('？'),
                best_text.rfind('.') if '.' in best_text else -1,
                best_text.rfind('!') if '!' in best_text else -1,
                best_text.rfind('?') if '?' in best_text else -1
            )
            if end_idx != -1:
                best_text = best_text[:end_idx + 1]
        
        return best_text
    
    def dynamic_compress(self, text: str, current_prompt: str = "", max_new_tokens: int = 0) -> Tuple[str, bool]:
        """
        动态压缩文本
        
        Args:
            text: 要压缩的文本
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            Tuple[str, bool]: (压缩后的文本, 是否进行了压缩)
        """
        if not self.enable_dynamic_compression or not text.strip():
            return text, False
        
        # 计算token数
        prompt_tokens = self.tokenizer.count_tokens(current_prompt) if current_prompt else 0
        text_tokens = self.tokenizer.count_tokens(text)
        
        # 如果不指定max_new_tokens，使用默认值
        if max_new_tokens <= 0:
            max_new_tokens = self.max_new_tokens
        
        # 计算总token数（输入+输出）
        total_tokens = prompt_tokens + text_tokens + max_new_tokens
        
        # 如果总token数在限制内，不需要压缩
        if total_tokens <= self.session_len:
            return text, False
        
        # 快速压缩模式
        if self.use_fast_compression:
            available_tokens = self.session_len - prompt_tokens - max_new_tokens
            return self.fast_compress_to_fit(text, available_tokens), True
        
        # 检测文本类型并选择最佳压缩策略
        text_type = self.detect_text_type(text)
        
        # 计算可用的文本token数（上下文窗口 - 提示词 - 预留生成token数）
        available_tokens = self.session_len - prompt_tokens - max_new_tokens
        
        # 根据文本类型选择压缩策略
        if text_type == 'code':
            # 代码类型：保留结构，优先压缩注释和文档字符串
            return self._compress_code(text, available_tokens), True
        elif text_type == 'list':
            # 列表类型：保留列表项，优先压缩描述
            return self._compress_list(text, available_tokens), True
        elif text_type == 'dialogue':
            # 对话类型：保留对话内容，优先压缩描述性文本
            return self._compress_dialogue(text, available_tokens), True
        else:
            # 叙述类型：使用标准压缩
            return self._compress_narrative(text, available_tokens), True
    
    def _compress_narrative(self, text: str, available_tokens: int) -> str:
        """压缩叙述类型文本，使用去重和智能摘要"""
        segments = self.split_text(text)
        
        print(f"[DEBUG] Original segments count: {len(segments)}")
        
        unique_segments = segments
        
        print(f"[DEBUG] Skipping deduplication (no unique segments)")
        
        if len(unique_segments) <= 3:
            print(f"[DEBUG] Using simple_summarize (only {len(unique_segments)} unique segments)")
            return self.simple_summarize(text, available_tokens)
        
        segment_info = []
        for idx, segment in enumerate(unique_segments):
            importance = self.calculate_importance(segment, idx, len(unique_segments))
            token_count = self.tokenizer.count_tokens(segment)
            segment_info.append(TextSegment(
                content=segment,
                importance=importance,
                token_count=token_count,
                is_keyword_segment=any(keyword in segment.lower() for keyword in self.keywords)
            ))
        
        segment_info.sort(key=lambda x: x.importance, reverse=True)
        
        compressed_segments = []
        total_compressed_tokens = 0
        
        for seg in segment_info:
            if total_compressed_tokens + seg.token_count <= available_tokens:
                compressed_segments.append((seg.content, seg.content))
                total_compressed_tokens += seg.token_count
            else:
                remaining_tokens = available_tokens - total_compressed_tokens
                if remaining_tokens <= 0:
                    break
                
                compressed_content = self.compress_segment(seg.content, remaining_tokens)
                compressed_segments.append((seg.content, compressed_content))
                total_compressed_tokens += self.tokenizer.count_tokens(compressed_content)
        
        print(f"[DEBUG] Selected {len(compressed_segments)} segments, total tokens: {total_compressed_tokens}")
        
        # 恢复原始顺序
        compressed_segments.sort(key=lambda x: unique_segments.index(x[0]))
        
        # 重建文本
        compressed_text = "\n\n".join([seg[1] for seg in compressed_segments])
        
        return compressed_text.strip()
    
    def _compress_code(self, text: str, available_tokens: int) -> str:
        """压缩代码类型文本，保留代码结构"""
        # 提取代码块和非代码部分
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        non_code_parts = re.split(r'```[\s\S]*?```', text)
        
        # 计算非代码部分的token
        non_code_text = ''.join(non_code_parts)
        non_code_tokens = self.tokenizer.count_tokens(non_code_text)
        
        # 如果非代码部分已经超过限制，只压缩非代码部分
        if non_code_tokens > available_tokens:
            return self._compress_narrative(text, available_tokens)
        
        # 保留所有代码块，压缩非代码部分
        remaining_tokens = available_tokens - non_code_tokens
        compressed_non_code = self._compress_narrative(non_code_text, remaining_tokens)
        
        # 重建文本
        result = []
        code_idx = 0
        for part in non_code_parts:
            if part:
                result.append(compressed_non_code[:len(part)])
                compressed_non_code = compressed_non_code[len(part):]
            if code_idx < len(code_blocks):
                result.append(code_blocks[code_idx])
                code_idx += 1
        
        return ''.join(result)
    
    def _compress_list(self, text: str, available_tokens: int) -> str:
        """压缩列表类型文本，保留列表项"""
        lines = text.split('\n')
        list_items = []
        non_list_lines = []
        
        for line in lines:
            if re.match(r'^\s*[\d\-\*•]+\s+', line):
                list_items.append(line)
            else:
                non_list_lines.append(line)
        
        # 优先保留列表项
        compressed_list = []
        total_tokens = 0
        
        for item in list_items:
            item_tokens = self.tokenizer.count_tokens(item)
            if total_tokens + item_tokens <= available_tokens:
                compressed_list.append(item)
                total_tokens += item_tokens
            else:
                remaining = available_tokens - total_tokens
                if remaining > 0:
                    compressed_item = self.compress_segment(item, remaining)
                    compressed_list.append(compressed_item)
                    total_tokens += self.tokenizer.count_tokens(compressed_item)
                break
        
        # 如果还有空间，添加非列表行
        if total_tokens < available_tokens:
            remaining = available_tokens - total_tokens
            compressed_non_list = self._compress_narrative('\n'.join(non_list_lines), remaining)
            compressed_list.append(compressed_non_list)
        
        return '\n'.join(compressed_list)
    
    def _compress_dialogue(self, text: str, available_tokens: int) -> str:
        """压缩对话类型文本，保留对话内容"""
        # 提取对话内容
        dialogues = re.findall(r'["「『][^"」』]*["」』]', text)
        non_dialogue = re.sub(r'["「『][^"」』]*["」』]', '', text)
        
        # 优先保留对话
        compressed_dialogues = []
        total_tokens = 0
        
        for dialogue in dialogues:
            dialogue_tokens = self.tokenizer.count_tokens(dialogue)
            if total_tokens + dialogue_tokens <= available_tokens:
                compressed_dialogues.append(dialogue)
                total_tokens += dialogue_tokens
            else:
                remaining = available_tokens - total_tokens
                if remaining > 0:
                    compressed_dialogue = self.compress_segment(dialogue, remaining)
                    compressed_dialogues.append(compressed_dialogue)
                    total_tokens += self.tokenizer.count_tokens(compressed_dialogue)
                break
        
        # 如果还有空间，添加非对话内容
        if total_tokens < available_tokens:
            remaining = available_tokens - total_tokens
            compressed_non_dialogue = self._compress_narrative(non_dialogue, remaining)
            compressed_dialogues.append(compressed_non_dialogue)
        
        return ' '.join(compressed_dialogues)
    
    def compress_chat_history(self, chat_history: List[Dict[str, Any]], current_prompt: str = "", max_new_tokens: int = 0) -> Tuple[List[Dict[str, Any]], bool]:
        """
        压缩聊天历史
        
        Args:
            chat_history: 聊天历史列表
            current_prompt: 当前的提示词
            max_new_tokens: 生成新token的最大数量
            
        Returns:
            Tuple[List[Dict[str, Any]], bool]: (压缩后的聊天历史, 是否进行了压缩)
        """
        if not self.enable_dynamic_compression or not chat_history:
            return chat_history, False
        
        # 计算当前token数
        prompt_tokens = self.tokenizer.count_tokens(current_prompt) if current_prompt else 0
        history_tokens = self.tokenizer.count_tokens(chat_history)
        
        # 如果不指定max_new_tokens，使用默认值
        if max_new_tokens <= 0:
            max_new_tokens = self.max_new_tokens
        
        total_tokens = prompt_tokens + history_tokens + max_new_tokens
        
        # 如果总token数在限制内，不需要压缩
        if total_tokens <= self.session_len:
            return chat_history, False
        
        available_tokens = self.session_len - prompt_tokens - max_new_tokens
        
        if available_tokens <= 0:
            return chat_history, False
        
        # 计算每个消息的token数
        message_tokens = []
        for msg in chat_history:
            message_tokens.append(self.tokenizer.count_tokens(msg))
        
        # 从最新的消息开始保留，压缩最早的消息
        compressed_history = []
        total_compressed_tokens = 0
        compressed = False
        
        # 倒序处理消息，从最新的开始
        reversed_chat = list(reversed(chat_history))
        reversed_tokens = list(reversed(message_tokens))
        temp_history = []
        temp_total = 0
        
        # 先保留最新的消息
        for msg, tokens in zip(reversed_chat, reversed_tokens):
            if temp_total + tokens <= available_tokens:
                temp_history.append(msg)
                temp_total += tokens
            else:
                break
        
        # 如果所有消息都可以保留，不需要压缩
        if len(temp_history) == len(chat_history):
            return chat_history, False
        
        # 如果没有保留任何消息，至少保留最新的一条（强制压缩）
        if not temp_history:
            latest_msg = reversed_chat[0]
            latest_tokens = reversed_tokens[0]
            
            if "content" in latest_msg and isinstance(latest_msg["content"], str):
                target_tokens = max(1, available_tokens)
                # 使用智能压缩方法而不是简单截断
                compressed_content = self._compress_narrative(latest_msg["content"], target_tokens)
                compressed_msg = latest_msg.copy()
                compressed_msg["content"] = compressed_content
                temp_history.append(compressed_msg)
                temp_total = self.tokenizer.count_tokens(compressed_content)
                compressed = True
        
        # 正序处理剩余的消息（最早的消息）
        remaining = len(chat_history) - len(temp_history)
        remaining_available = available_tokens - temp_total
        
        for i in range(remaining):
            msg = chat_history[i]
            tokens = message_tokens[i]
            
            if tokens <= remaining_available:
                compressed_history.append(msg)
                remaining_available -= tokens
            else:
                # 压缩最早的消息
                if "content" in msg and isinstance(msg["content"], str):
                    # 计算需要压缩到的token数
                    target_tokens = max(1, remaining_available)
                    # 使用智能压缩方法
                    compressed_content = self._compress_narrative(msg["content"], target_tokens)
                    compressed_tokens = self.tokenizer.count_tokens(compressed_content)
                    
                    if compressed_tokens <= remaining_available:
                        compressed_msg = msg.copy()
                        compressed_msg["content"] = compressed_content
                        compressed_history.append(compressed_msg)
                        remaining_available -= compressed_tokens
                        compressed = True
            
            if remaining_available <= 0:
                break
        
        # 合并压缩后的最早消息和保留的最新消息
        compressed_history.extend(reversed(temp_history))
        
        # 如果压缩后消息数量减少，标记为已压缩
        if len(compressed_history) < len(chat_history):
            compressed = True
        
        return compressed_history, compressed
    
    def simple_summarize(self, text: str, max_tokens: int = 100) -> str:
        """
        简单的文本摘要方法
        优化：减少重复的token计算，使用贪心算法选择句子
        """
        if not text:
            return ""
        
        # 按句子分割
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        # 提取包含关键词的句子，并预计算每个句子的token数
        sentence_data = []
        for sent in sentences:
            tokens = self.tokenizer.count_tokens(sent)
            has_keyword = any(keyword in sent.lower() for keyword in self.keywords)
            sentence_data.append({
                'text': sent,
                'tokens': tokens,
                'has_keyword': has_keyword
            })
        
        # 分离关键词句子和非关键词句子
        key_sentences = [s for s in sentence_data if s['has_keyword']]
        non_key_sentences = [s for s in sentence_data if not s['has_keyword']]
        
        # 如果没有关键词句子，使用前几个句子
        if not key_sentences:
            key_sentences = non_key_sentences[:3]
        
        # 按token数排序关键词句子（短句子优先）
        sorted_key_sentences = sorted(key_sentences, key=lambda x: x['tokens'])
        
        # 使用贪心算法选择句子
        selected_sentences = []
        total_tokens = 0
        
        for sent_data in sorted_key_sentences:
            sent_text = sent_data['text']
            sent_tokens = sent_data['tokens']
            separator_tokens = 1 if selected_sentences else 0  # 句号占1个token
            
            if total_tokens + sent_tokens + separator_tokens <= max_tokens:
                selected_sentences.append(sent_text)
                total_tokens += sent_tokens + separator_tokens
            else:
                break
        
        if selected_sentences:
            summary = '。'.join(selected_sentences) + '。'
        else:
            # 如果单个关键词句子都太长，直接压缩第一个关键词句子
            summary = self.compress_segment(sorted_key_sentences[0]['text'] + '。', max_tokens)
        
        # 确保摘要中包含关键词
        if not any(keyword in summary.lower() for keyword in self.keywords) and key_sentences:
            # 如果摘要中没有关键词，尝试压缩关键词句子
            for sent_data in key_sentences:
                compressed = self.compress_segment(sent_data['text'] + '。', max_tokens)
                if any(keyword in compressed.lower() for keyword in self.keywords):
                    return compressed
            
            # 如果还是没有，返回包含关键词的最短文本
            return "结论：保留重要信息。"
        
        # 最后检查token数（只检查一次）
        final_tokens = self.tokenizer.count_tokens(summary)
        if final_tokens > max_tokens:
            summary = self.compress_segment(summary, max_tokens)
        
        return summary
