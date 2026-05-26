import json
from typing import List, Optional, Any
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

class Turn(BaseModel):
    speaker: str
    text: str
    target: str = "All"

class HybridMemoryManager:
    def __init__(
        self,
        max_tokens: int = 2000,
        max_context_items: int = 5,  # 💡 [추가] 동적 정보 요약 임계치 (파라미터)
        project_id: str = None,
        location: str = "global"
    ):
        self.max_tokens = max_tokens
        self.max_context_items = max_context_items

        # --- 기존: 대화(Turn) 메모리 ---
        self.past_summary: str = ""
        self.recent_turns: List[Turn] = []
        self._current_token_count: int = 0
        self._summary_token_count: int = 0

        # --- 💡 [신규] 동적 컨텍스트(ML/PIXEL) 메모리 ---
        self.past_ml_summary: str = ""
        self.recent_ml: List[Any] = []

        self.past_pixel_summary: str = ""
        self.recent_pixel: List[Any] = []

        if project_id:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init(location=location)

        self.gemini_model = GenerativeModel("gemini-2.5-flash-lite")

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        return self.gemini_model.count_tokens(text).total_tokens

    # =====================================================================
    # [기존 로직 유지] 대화 기록 관리 (get_formatted_history, add_interaction 등)
    # =====================================================================
    def get_formatted_history(self, current_name: str = None) -> str:
        history_str = ""
        if self.past_summary:
            history_str += f"<PAST_SUMMARY>\n{self.past_summary}\n</PAST_SUMMARY>\n"
        if self.recent_turns:
            history_str += "<RECENT_TURNS_CONTEXT>\n"
            for turn in self.recent_turns:
                if current_name:
                    if turn.speaker == "Moderator":
                        history_str += f"Moderator: {turn.text}\n"
                    elif turn.speaker == current_name:
                        history_str += f"Your Past Answer ({current_name}): {turn.text}\n"
                    else:
                        history_str += f"Other Participant ({turn.speaker}): {turn.text}\n"
                else:
                    history_str += f"{turn.speaker}: {turn.text}\n"
            history_str += "</RECENT_TURNS_CONTEXT>"
        return history_str.strip()

    def add_interaction(self, speaker: str, text: str, target: str = "All"):
        self.recent_turns.append(Turn(speaker=speaker, text=text, target=target))
        self._current_token_count += self._count_tokens(f"{speaker}: {text}\n")
        if (self._summary_token_count + self._current_token_count) > self.max_tokens:
            self._manage_memory_window()

    def _manage_memory_window(self):
        if len(self.recent_turns) <= 2: return
        chunk_size = max(1, len(self.recent_turns) // 2)
        turns_to_summarize = self.recent_turns[:chunk_size]
        self.recent_turns = self.recent_turns[chunk_size:]
        self._update_summary_batch(turns_to_summarize)

        self._summary_token_count = self._count_tokens(self.past_summary) if self.past_summary else 0
        self._current_token_count = self._count_tokens("".join([f"{t.speaker}: {t.text}\n" for t in self.recent_turns]))

    def _update_summary_batch(self, turns: List[Turn]):
        new_dialogue = "\n".join([f"{t.speaker}: {t.text}" for t in turns])
        is_multi = len(set(t.speaker for t in turns if t.speaker != "Moderator")) > 1
        prompt = f"""다음은 대화 내용입니다. 기존 요약본과 새로운 대화를 바탕으로 핵심을 요약해주세요.
[기존 요약]\n{self.past_summary if self.past_summary else '없음'}
[추가 대화]\n{new_dialogue}"""
        try:
            self.past_summary = self.gemini_model.generate_content(prompt).text.strip()
        except Exception:
            self.past_summary += " | [일부 요약 실패]"

    # =====================================================================
    # 💡 [신규 로직] 유관 정보(ML/PIXEL) 관리 및 자동 요약
    # =====================================================================
    def add_dynamic_context(self, dynamic_ml: list, dynamic_pixel: list):
        """새로 검색된 데이터를 누적하고, 파라미터를 넘으면 요약을 트리거합니다."""
        for m in dynamic_ml:
            if m not in self.recent_ml:
                self.recent_ml.append(m)

        for p in dynamic_pixel:
            if p not in self.recent_pixel:
                self.recent_pixel.append(p)

        if len(self.recent_ml) >= self.max_context_items:
            self._summarize_context("ML")
        if len(self.recent_pixel) >= self.max_context_items:
            self._summarize_context("PIXEL")

    def _summarize_context(self, target: str):
        if target == "ML":
            prompt = f"다음 구매 이력을 한 문단으로 요약하세요.\n[기존 요약]: {self.past_ml_summary}\n[추가 이력]: {json.dumps(self.recent_ml, ensure_ascii=False)}"
            try:
                self.past_ml_summary = self.gemini_model.generate_content(prompt).text.strip()
            except Exception:
                pass
            self.recent_ml = [] # 압축 후 비우기

        elif target == "PIXEL":
            prompt = f"다음 성향(PIXEL) 데이터를 한 문단으로 요약하세요.\n[기존 요약]: {self.past_pixel_summary}\n[추가 특성]: {json.dumps(self.recent_pixel, ensure_ascii=False)}"
            try:
                self.past_pixel_summary = self.gemini_model.generate_content(prompt).text.strip()
            except Exception:
                pass
            self.recent_pixel = [] # 압축 후 비우기

    def get_combined_ml(self, base_ml: list) -> list:
        """[Base + 압축된 과거 요약 + 최근 누적] 형태로 병합"""
        result = base_ml.copy() if isinstance(base_ml, list) else []
        if self.past_ml_summary:
            result.append({"[누적 과거 이력 요약]": self.past_ml_summary})
        result.extend(self.recent_ml)
        return result

    def get_combined_pixel(self, base_pixel: list) -> list:
        """[Base + 압축된 과거 요약 + 최근 누적] 형태로 병합"""
        result = base_pixel.copy() if isinstance(base_pixel, list) else []
        if self.past_pixel_summary:
            result.append(f"[누적 과거 특성 요약] {self.past_pixel_summary}")
        result.extend(self.recent_pixel)
        return result
