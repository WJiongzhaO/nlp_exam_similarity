import os
import json
import re
import logging
import hashlib
from bs4 import BeautifulSoup
import bleach
from src.utils import normalize_choices, extract_core_content

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataCleaner")


def clean_html(raw_html: str) -> str:
    math_pattern = re.compile(r'(\$(.*?)\$|\\\[(.*?)\\\])')
    math_blocks = []

    def save_math(match):
        math_blocks.append(match.group(0))
        return f'__MATH_{len(math_blocks) - 1}__'

    protected_html = math_pattern.sub(save_math, raw_html)

    allowed_tags = [
        'p', 'div', 'span', 'br', 'ol', 'ul', 'li',
        'table', 'tr', 'td', 'th', 'strong', 'em', 'b', 'i'
    ]
    cleaned = bleach.clean(protected_html, tags=allowed_tags, strip=True)
    soup = BeautifulSoup(cleaned, 'html.parser')
    for tag in soup.find_all(True):
        tag.attrs = {}
    clean_text = str(soup)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_text)

    def restore_math(match):
        idx = int(match.group(1))
        return math_blocks[idx] if idx < len(math_blocks) else ""

    return re.sub(r'__MATH_(\d+)__', restore_math, clean_text.strip())


def segment_text(text: str, max_length=500) -> list:
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


def get_content_fingerprint(question: dict, paper_id: str) -> str:
    """生成带试卷标识的内容指纹，避免同内容题目被误去重"""
    # 使用核心内容生成指纹
    core_text = extract_core_content(question.get("text", ""))

    # 加入 paper_id，保证不同试卷的相同内容不会被当成重复
    content_with_id = f"{paper_id}::{core_text.strip()}"
    return hashlib.sha256(content_with_id.encode('utf-8')).hexdigest()


def clean_paper_data(input_file: str, output_file: str, paper_id: str = None, deduplicate: bool = True):
    logger.info(f"开始清洗试卷数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        exam_data = json.load(f)

    cleaned_questions = []
    seen_fingerprints = set()

    # 若未指定 paper_id，则用文件名
    if not paper_id:
        paper_id = os.path.basename(input_file)

    for question in exam_data:
        rich_content = question.get("richTextContent", "")
        question_id = question.get("id", "")
        cleaned_content = clean_html(rich_content)
        segments = segment_text(cleaned_content)
        cleaned_question = {
            "id": question_id,
            "type": question.get("type", "未知"),
            "text": cleaned_content,
            "segments": segments
        }

        # 总是生成指纹
        fingerprint = get_content_fingerprint(cleaned_question, paper_id)
        cleaned_question["fingerprint"] = fingerprint

        # 只有在启用去重且指纹未见过时才添加
        if deduplicate and fingerprint in seen_fingerprints:
            logger.info(f"跳过重复题目: {question_id}")
            continue

        seen_fingerprints.add(fingerprint)
        cleaned_questions.append(cleaned_question)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_questions, f, indent=2, ensure_ascii=False)
    logger.info(f"清洗完成! 结果保存到: {output_file}, 题目数量: {len(cleaned_questions)}")