import os
import json
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import sys

# 添加路径以确保正确导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .vectorizer import euclidean_distance  # 使用vectorizer中的实现

logger = logging.getLogger("SimilarityCalculator")

def fused_similarity(v_a: np.ndarray,
                     v_b: np.ndarray,
                     w_cos: float = 0.6) -> float:
    """
    余弦 + 欧氏 融合距离 → 相似度
    w_cos 越大，余弦占比越高
    """
    v_a = v_a.reshape(1, -1)
    v_b = v_b.reshape(1, -1)

    cos_sim = cosine_similarity(v_a, v_b)[0, 0]
    euc_dist = euclidean_distance(v_a.flatten(), v_b.flatten())
    euc_sim = 1 / (1 + euc_dist)  # 转成 0~1

    return w_cos * cos_sim + (1 - w_cos) * euc_sim

def calculate_similarity(
        paper_a_file: str,
        paper_b_file: str,
        threshold: float = 0.7,
        type_sensitive: bool = True,
        fusion_weight: float = 0.6,
        deduplicate: bool = True
) -> dict:
    logger.info(f"计算试卷相似度: {paper_a_file} vs {paper_b_file}")

    with open(paper_a_file, "r", encoding="utf-8") as f:
        paper_a = json.load(f)
    with open(paper_b_file, "r", encoding="utf-8") as f:
        paper_b = json.load(f)

    # 收集向量 & 题号 & 文本
    vectors_a, info_a = [], []
    for q in paper_a["questions"]:
        if "vector" in q and q["vector"] is not None:
            vectors_a.append(np.array(q["vector"]))
            info_a.append({
                "id": q["id"],
                "type": q["type"],
                "text": q["text"]
            })

    vectors_b, info_b = [], []
    for q in paper_b["questions"]:
        if "vector" in q and q["vector"] is not None:
            vectors_b.append(np.array(q["vector"]))
            info_b.append({
                "id": q["id"],
                "type": q["type"],
                "text": q["text"]
            })

    if not vectors_a or not vectors_b:
        return {
            "paper_a": paper_a_file,
            "paper_b": paper_b_file,
            "method": "fused",
            "threshold": threshold,
            "type_sensitive": type_sensitive,
            "fusion_weight": fusion_weight,
            "total_questions_a": len(vectors_a),
            "total_questions_b": len(vectors_b),
            "total_pairs": 0,
            "similar_pairs": []
        }

    # 使用矩阵运算优化计算效率
    matrix_a = np.array(vectors_a)
    matrix_b = np.array(vectors_b)

    # 计算余弦相似度矩阵
    cos_sim_matrix = cosine_similarity(matrix_a, matrix_b)

    # 优化欧氏距离计算 - 向量化替代循环
    # 计算差值矩阵
    diff = matrix_a[:, np.newaxis, :] - matrix_b[np.newaxis, :, :]
    # 计算平方和
    squared_diff = np.sum(diff**2, axis=-1)
    # 计算欧氏距离
    euc_dist_matrix = np.sqrt(squared_diff)
    # 转换为相似度
    euc_sim_matrix = 1 / (1 + euc_dist_matrix)

    # 融合相似度
    fused_sim_matrix = fusion_weight * cos_sim_matrix + (1 - fusion_weight) * euc_sim_matrix

    # 找出满足条件的相似对
    similar_pairs = []
    for i in range(len(info_a)):
        for j in range(len(info_b)):
            if type_sensitive and info_a[i]["type"] != info_b[j]["type"]:
                continue
            sim = fused_sim_matrix[i][j]
            if sim >= threshold:
                similar_pairs.append({
                    "paper_a": info_a[i],
                    "paper_b": info_b[j],
                    "similarity": float(sim)
                })

    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "paper_a": paper_a_file,
        "paper_b": paper_b_file,
        "method": "fused",
        "threshold": threshold,
        "type_sensitive": type_sensitive,
        "fusion_weight": fusion_weight,
        "total_questions_a": len(vectors_a),
        "total_questions_b": len(vectors_b),
        "total_pairs": len(similar_pairs),
        "deduplicate": deduplicate,
        "similar_pairs": similar_pairs
    }