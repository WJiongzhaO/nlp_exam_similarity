import os
import json
import logging
import numpy as np
import re
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Utils")


def extract_core_content(text: str) -> str:
    """
    从题目中提取核心文本内容（去除HTML标签、题号、分值等）

    参数:
        text: 原始文本

    返回:
        核心文本内容
    """
    if not text:
        return ""

    try:
        # 使用BeautifulSoup移除HTML标签
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)

        # 移除题号模式 (例如: "1.", "一、", "(1)", "[1]")
        clean_text = re.sub(
            r'^(?:\d+[\.、]?|[一二三四五六七八九十]+[、.]?\s*|\(\d+\)|\[\d+\]\s*)\s*',
            '',
            clean_text
        )

        # 移除分值信息 (例如: "(5分)", "[10分]")
        clean_text = re.sub(r'[\(\[][\d\.]+分[\)\]]', '', clean_text)

        # 移除额外的空白字符
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text
    except Exception as e:
        logger.error(f"提取核心内容失败: {str(e)}")
        return text.strip()


def generate_html_report(results: Dict[str, Any], output_file: str):
    """
    生成HTML格式的相似度报告

    参数:
        results: 相似度计算结果
        output_file: 输出HTML文件路径
    """
    try:
        # 提取结果数据
        paper_a_name = os.path.basename(results.get("paper_a", "试卷A"))
        paper_b_name = os.path.basename(results.get("paper_b", "试卷B"))
        similar_pairs = results.get("similar_pairs", [])
        threshold = results.get("threshold", 0.7)
        type_sensitive = results.get("type_sensitive", True)
        fusion_weight = results.get("fusion_weight", 0.6)
        deduplicate = results.get("deduplicate", True)
        total_pairs = len(similar_pairs)

        # 计算整体相似度（如果有定义的话）
        overall_similarity = results.get("overall_similarity", 0.0)
        if not overall_similarity and total_pairs > 0:
            # 如果没有提供整体相似度，计算平均相似度
            similarities = [pair["similarity"] for pair in similar_pairs]
            overall_similarity = sum(similarities) / len(similarities)

        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>试卷相似度分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ margin-bottom: 30px; }}
                .param {{ margin: 5px 0; }}
                .highlight {{ background-color: #ffeb3b; }}
            </style>
        </head>
        <body>
            <h1>试卷相似度分析报告</h1>

            <div class="summary">
                <h2>分析概览</h2>
                <p><strong>试卷A:</strong> {paper_a_name}</p>
                <p><strong>试卷B:</strong> {paper_b_name}</p>
                <p><strong>整体相似度:</strong> {overall_similarity:.4f}</p>
                <p><strong>相似题目对数:</strong> {total_pairs}</p>
            </div>

            <div class="params">
                <h2>分析参数</h2>
                <p class="param"><strong>相似度阈值:</strong> {threshold}</p>
                <p class="param"><strong>类型敏感:</strong> {'是' if type_sensitive else '否'}</p>
                <p class="param"><strong>融合权重(余弦):</strong> {fusion_weight}</p>
                <p class="param"><strong>题目去重:</strong> {'是' if deduplicate else '否'}</p>
            </div>

            <div class="details">
                <h2>相似题目详情</h2>
                <table>
                    <tr>
                        <th>序号</th>
                        <th>试卷A题目ID</th>
                        <th>试卷A题目类型</th>
                        <th>试卷A题目内容摘要</th>
                        <th>试卷B题目ID</th>
                        <th>试卷B题目类型</th>
                        <th>试卷B题目内容摘要</th>
                        <th>相似度</th>
                    </tr>
        """

        # 添加相似题目对
        for idx, pair in enumerate(similar_pairs, 1):
            q_a = pair.get("paper_a", {})
            q_b = pair.get("paper_b", {})
            similarity = pair.get("similarity", 0.0)

            # 高亮显示高度相似的题目
            row_class = "class='highlight'" if similarity > 0.9 else ""

            # 截取内容摘要
            text_a = extract_core_content(q_a.get('text', ''))[:100] + '...' if len(
                q_a.get('text', '')) > 100 else extract_core_content(q_a.get('text', ''))
            text_b = extract_core_content(q_b.get('text', ''))[:100] + '...' if len(
                q_b.get('text', '')) > 100 else extract_core_content(q_b.get('text', ''))

            html_content += f"""
                    <tr {row_class}>
                        <td>{idx}</td>
                        <td>{q_a.get('id', '')}</td>
                        <td>{q_a.get('type', '')}</td>
                        <td>{text_a}</td>
                        <td>{q_b.get('id', '')}</td>
                        <td>{q_b.get('type', '')}</td>
                        <td>{text_b}</td>
                        <td>{similarity:.4f}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        # 写入文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info(f"HTML报告已生成: {output_file}")
        return True
    except Exception as e:
        logging.error(f"生成HTML报告失败: {str(e)}")
        return False


def normalize_choices(choices: List[str]) -> List[str]:
    """
    标准化选择题选项，将选项格式统一为大写字母加括号，如 "(A)"、"(B)" 等。

    参数:
        choices: 原始选项列表

    返回:
        标准化后的选项列表
    """
    normalized = []
    for choice in choices:
        # 清理选项文本
        cleaned = choice.strip()
        if not cleaned:
            normalized.append(choice)
            continue

        # 尝试匹配标准选项格式
        match = re.match(r'^([(（]?)([A-Za-z])([)）]?)(.*)$', cleaned)
        if match:
            prefix, letter, suffix, content = match.groups()
            # 标准化为 "(A) 内容" 格式
            normalized_choice = f"({letter.upper()}) {content.strip()}"
            normalized.append(normalized_choice)
        else:
            # 其他情况保持原样
            normalized.append(choice)
    return normalized


def calculate_overall_similarity(similar_pairs: List[Dict], total_questions: int) -> float:
    """
    计算整体试卷相似度

    参数:
        similar_pairs: 相似题目对列表
        total_questions: 试卷题目总数（取两试卷中题目数较多者）

    返回:
        整体相似度分数 (0.0-1.0)
    """
    if not similar_pairs or total_questions == 0:
        return 0.0

    # 计算相似题目权重
    similarity_sum = sum(pair["similarity"] for pair in similar_pairs)

    # 考虑相似题目数量占比
    coverage = len(similar_pairs) / total_questions

    # 综合计算整体相似度
    return min(1.0, similarity_sum * coverage / len(similar_pairs))


if __name__ == "__main__":
    # 测试函数
    test_text = "1. (5分) 以下哪个是<strong>正确</strong>的选项?"
    print(f"原始文本: {test_text}")
    print(f"核心内容: {extract_core_content(test_text)}")

    test_choices = ["A. 选项1", "(B) 选项2", "C选项3", "（D） 选项4"]
    print(f"原始选项: {test_choices}")
    print(f"标准化选项: {normalize_choices(test_choices)}")