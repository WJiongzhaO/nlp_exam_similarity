import sys
import os
import json
import time
import logging
import numpy as np
from .vectorizer import TextVectorizer

logger = logging.getLogger("PaperVectorizer")


def vectorize_paper(input_file: str, output_file: str, model_type: str = "sentence-bert", model_dir: str = None):
    logger.info(f"开始向量化试卷: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    vectorizer = TextVectorizer(model_type, model_dir=model_dir)
    start_time = time.time()
    vectorized_questions = []

    for question in exam_data:
        question_text = question["text"]
        segments = question.get("segments", [])
        question_data = {
            "id": question["id"],
            "type": question["type"],
            "score": question.get("score", 0),
            "text": question_text,
            "segments": segments,
            "fingerprint": question.get("fingerprint", ""),
            "vector": None
        }

        # 添加空向量处理
        if segments:
            embeddings = []
            for seg in segments:
                emb = vectorizer.vectorize_text(seg)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                question_data["vector"] = np.mean(embeddings, axis=0).tolist()
        else:
            embedding = vectorizer.vectorize_text(question_text)
            if embedding is not None:
                question_data["vector"] = embedding.tolist()
            else:
                # 添加空向量处理
                logger.warning(f"题目 {question['id']} 向量化失败，使用零向量替代")
                question_data["vector"] = np.zeros(vectorizer.vector_size).tolist()

        vectorized_questions.append(question_data)

    elapsed = time.time() - start_time
    logger.info(f"向量化完成! 耗时: {elapsed:.2f}秒")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_type,
            "questions": vectorized_questions
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到: {output_file}")