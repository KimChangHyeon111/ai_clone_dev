import os
import re
import asyncio
import polars as pl
from typing import Tuple, Dict, Any, List, Union, Optional

# =====================================================================
# 1. 출력 통제 계층 (입력과 동일한 DeBERTa 파이프라인 공유)
# =====================================================================
class OutputGuardrail:
    """Input 계층에서 쓰는 DeBERTa 파이프라인을 그대로 주입받아 출력 검열 수행"""
    
    def __init__(
        self, 
        classifier_pipeline,  # 💡 핵심: InputGuardrail에 넣었던 그 모델 그대로 받음!
        forbidden_words_path: str = "forbidden_words.txt"
    ):
        self.classifier = classifier_pipeline
        
        self.forbidden_words_path = forbidden_words_path
        self.forbidden_pattern: Optional[re.Pattern] = None
        self._load_forbidden_words()

    def _load_forbidden_words(self):
        forbidden_words = []
        if os.path.exists(self.forbidden_words_path):
            with open(self.forbidden_words_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        forbidden_words.append(word)
        if forbidden_words:
            sorted_words = sorted(forbidden_words, key=len, reverse=True)
            self.forbidden_pattern = re.compile('|'.join(map(re.escape, sorted_words)), re.IGNORECASE)

    def _filter_forbidden_words(self, text: str) -> str:
        if self.forbidden_pattern:
            return self.forbidden_pattern.sub(lambda m: '*' * len(m.group()), text)
        return text

    def process_text(self, text: str) -> Tuple[bool, str, str]:
        if not isinstance(text, str) or not text.strip():
            return True, text, "EMPTY_OR_NON_STRING"
        
        # 1차: 가벼운 금칙어 마스킹 (정규식 기반)
        clean_text = self._filter_forbidden_words(text)
        
        # 2차: 주입받은 DeBERTa 파이프라인으로 딥러닝 문맥 검열
        try:
            # 텍스트가 너무 길면 DeBERTa max_length(512) 에러가 나므로 자름
            input_text = clean_text[:512] 
            
            # 동기식 파이프라인 추론 (비동기 루프는 Postprocessor에서 to_thread로 빼줌)
            result = self.classifier(input_text)[0]
            
            # 💡 모델의 라벨에 맞게 수정 (예: INJECTION, TOXIC, UNSAFE 등)
            bad_labels = ["INJECTION", "TOXIC", "UNSAFE", "BAD_OUTPUT"]
            if result['label'] in bad_labels and result['score'] > 0.8:
                return False, "[시스템 알림] 안전 정책 위반으로 차단된 출력입니다.", f"POLICY_VIOLATION_{result['label']}"
                
        except Exception as e:
            print(f"⚠️ OutputGuardrail DeBERTa 추론 에러: {e}")
            pass # 모델 장애 시 기본 텍스트는 통과시킴

        return True, clean_text, "SAFE"


# =====================================================================
# 2. 통합 출력 후처리 계층 (OutputPostprocessorAsync)
# =====================================================================
class OutputPostprocessorAsync:
    def __init__(self, guardrail: OutputGuardrail):
        self.guardrail = guardrail

    async def process_output(self, intent: str, raw_result: Union[str, Tuple[pl.DataFrame, pl.DataFrame]]) -> Dict[str, Any]:
        # 💡 to_thread를 통해 DeBERTa의 무거운 추론이 비동기 루프를 멈추지 않게 방어
        if intent == "REALTIME_FGI":
            return await asyncio.to_thread(self._process_fgi_result, raw_result)
        elif intent == "MASS_SIMULATION":
            return await asyncio.to_thread(self._process_batch_result, raw_result)
        return {"status": "ERROR", "message": f"지원하지 않는 Intent 입니다: {intent}"}

    def _process_fgi_result(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text: return {"status": "ERROR", "message": "생성된 답변이 없습니다."}
        
        is_safe, safe_text, reason = self.guardrail.process_text(raw_text)
        
        if not is_safe:
            return {"status": "BLOCKED", "data": {"response": safe_text}, "reason": reason}
        return {"status": "SUCCESS", "data": {"response": safe_text}}

    def _process_batch_result(self, df_tuple: Tuple[pl.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        valid_df, error_df = df_tuple
        if not valid_df.is_empty():
            text_columns = [col for col in ["reasoning", "primary_reason", "improvement_suggestion"] if col in valid_df.columns]
            
            def _apply_guardrail(text: str) -> str:
                if not text: return ""
                _, safe_text, _ = self.guardrail.process_text(text)
                return safe_text

            if text_columns:
                valid_df = valid_df.with_columns([
                    pl.col(col).map_elements(_apply_guardrail, return_dtype=pl.String).alias(col) 
                    for col in text_columns
                ])
                
        valid_dicts = valid_df.to_dicts() if not valid_df.is_empty() else []
        error_dicts = error_df.to_dicts() if not error_df.is_empty() else []
        
        total_count = len(valid_dicts) + len(error_dicts)
        status = "EMPTY_RESULT" if total_count == 0 else "SUCCESS" if len(error_dicts) == 0 else "PARTIAL_SUCCESS"

        return {
            "status": status,
            "meta": {"total_processed": total_count, "success_count": len(valid_dicts), "error_count": len(error_dicts)},
            "data": {"valid_results": valid_dicts, "error_results": error_dicts}
        }
