import re

def clean_json_string(text: str) -> str:
    """LLM 응답에서 Markdown JSON 포맷팅 찌꺼기를 안전하게 제거합니다."""
    return re.sub(r'^```json\n|^```|```$', '', text.strip(), flags=re.MULTILINE).strip()
