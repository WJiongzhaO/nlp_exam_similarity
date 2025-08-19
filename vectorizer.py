import os
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.utils import extract_core_content
import sys

# 添加路径以确保正确导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("Vectorizer")

def euclidean_distance(v1, v2):
    """计算欧氏距离（使用numpy实现）"""
    return np.sqrt(np.sum((v1 - v2) ** 2))

class TextVectorizer:
    def __init__(self, model_type: str, device=None, model_dir=None):
        """初始化文本向量化工具"""
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 设置模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(current_dir, "models")
        logger.info(f"使用模型目录: {self.model_dir}")

        # 模型路径映射
        model_paths = {
            "bert": os.path.join(self.model_dir, "bert-base-chinese"),
            "bge": os.path.join(self.model_dir, "bge-small-zh"),
            "sentence-bert": os.path.join(self.model_dir, "paraphrase-multilingual-MiniLM-L12-v2")
        }

        if model_type not in model_paths:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.model_path = model_paths[model_type]

        # 检查模型路径
        if not os.path.exists(self.model_path):
            logger.error(f"模型目录不存在: {self.model_path}")
            raise FileNotFoundError(f"模型目录不存在: {self.model_path}")

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载模型并处理可能的错误"""
        try:
            logger.info(f"加载模型: {self.model_path}")
            if self.model_type == "sentence-bert":
                self.model = SentenceTransformer(self.model_path, device=self.device)
                self.vector_size = self.model.get_sentence_embedding_dimension()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
                self.model.eval()

                # 获取模型向量大小
                sample_input = self.tokenizer("测试文本", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model(**sample_input)
                self.vector_size = output.last_hidden_state.size(-1)

            logger.info(f"模型加载成功! 向量维度: {self.vector_size}")

        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def vectorize_text(self, text: str) -> np.ndarray:
        """
        对文本进行向量化（支持长文本分段和核心内容提取）
        :param text: 输入文本
        :return: 文本向量
        """
        try:
            # 处理空文本
            if not text.strip():
                logger.warning("尝试向量化空文本")
                return np.zeros(self.vector_size) if hasattr(self, 'vector_size') else None

            # 提取核心内容（忽略题号和分值）
            core_text = extract_core_content(text)

            # 长文本分段处理
            segments = self.segment_text(core_text)
            segment_embeddings = []

            # 向量化每个分段
            for seg in segments:
                if self.model_type == "sentence-bert":
                    emb = self.model.encode([seg])[0]
                    segment_embeddings.append(emb)
                else:
                    # 处理BERT/BGE类模型
                    inputs = self.tokenizer(
                        seg,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)

                    # 推理
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    # 获取最后的隐藏状态
                    last_hidden_state = outputs.last_hidden_state

                    # 应用改进的加权平均池化
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                    sum_mask = input_mask_expanded.sum(dim=1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    emb = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()

                    segment_embeddings.append(emb)

            # 合并分段向量
            if not segment_embeddings:
                return None

            # 简单平均融合
            return np.mean(segment_embeddings, axis=0)

        except Exception as e:
            logger.error(f"向量化失败: {text[:20]}... - {str(e)}")
            return None

    def segment_text(self, text: str, max_length=500) -> list:
        """长文本分段策略"""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        segmented = []
        for para in paragraphs:
            if len(para) <= max_length:
                segmented.append(para)
                continue
            punctuation_segments = []
            current_segment = ""
            for char in para:
                current_segment += char
                if char in {'。', '？', '！', '；'} and len(current_segment) >= max_length * 0.3:
                    punctuation_segments.append(current_segment.strip())
                    current_segment = ""
            if current_segment:
                punctuation_segments.append(current_segment.strip())
            for seg in punctuation_segments:
                if len(seg) <= max_length:
                    segmented.append(seg)
                else:
                    words = seg.split()
                    current_fragment = []
                    current_length = 0
                    for word in words:
                        if current_length + len(word) + 1 > max_length and current_fragment:
                            segmented.append(" ".join(current_fragment))
                            current_fragment = []
                            current_length = 0
                        current_fragment.append(word)
                        current_length += len(word) + 1
                    if current_fragment:
                        segmented.append(" ".join(current_fragment))
        return segmented