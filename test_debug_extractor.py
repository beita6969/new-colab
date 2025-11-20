#!/usr/bin/env python3
"""调试答案提取器"""
from src.answer_extractor import AnswerExtractor

extractor = AnswerExtractor(use_llm_fallback=False)

# 调试 case 3
text = "Therefore, the final answer is 42."
print(f"原始文本: {text}")
print()

# 模拟提取过程
import re

# 检查各个模式
patterns = [
    r"(?:Final Answer|最终答案|答案)[：:]*\s*([^\n.]+)",
    r"(?:The answer is|Therefore)[：:]*\s*([^\n.]+)",
    r"(?:the final answer is)[：:]*\s*([^\n.]+)",
    r"(?:=|equals to|is)\s*([\d\.,/-]+(?:\s*[a-zA-Z]+)?)",
]

for i, pattern in enumerate(patterns, 1):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        print(f"模式{i}匹配: {pattern}")
        print(f"提取内容: '{match.group(1)}'")
        print()

# 最终提取结果
result = extractor.extract_answer(text, "math", is_ground_truth=False)
print(f"最终提取结果: '{result}'")
