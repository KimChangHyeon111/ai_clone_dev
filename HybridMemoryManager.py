from typing import List, Optional
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

class Turn(BaseModel):
    speaker: str      # 발화 주체
    text: str         # 실제 발언 내용
    target: str = "All"  # 💡 [하위 호환성] 기존 코드에서 생략해도 에러가 나지 않도록 기본값 부여

class HybridMemoryManager:
    def __init__(
        self,
        max_tokens: int = 2000,
        project_id: str = None,
        location: str = "global",
        gemini_model = GenerativeModel("gemini-2.5-flash-lite")
    ):
        self.max_tokens = max_tokens
        self.past_summary: str = ""
        self.recent_turns: List[Turn] = []

        self._current_token_count: int = 0
        self._summary_token_count: int = 0

        if project_id:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init(location=location)

        self.gemini_model = gemini_model

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        response = self.gemini_model.count_tokens(text)
        return response.total_tokens

    # =====================================================================
    # 💡 [핵심 통합 1] 호출 방식에 따라 1:1과 다자간 포맷팅을 자동 분기
    # =====================================================================
    def get_formatted_history(self, current_name: str = None) -> str:
        """
        - 기존 1:1 챗봇 호출: get_formatted_history() -> 기본 포맷팅
        - 다자간 FGI 호출: get_formatted_history(current_name="Customer_A") -> 격리 포맷팅
        """
        history_str = ""
        if self.past_summary:
            history_str += f"<PAST_SUMMARY>\n{self.past_summary}\n</PAST_SUMMARY>\n"

        if self.recent_turns:
            history_str += "<RECENT_TURNS_CONTEXT>\n"
            for turn in self.recent_turns:
                # 1. 다자간 대화 모드 (current_name 파라미터가 들어온 경우)
                if current_name:
                    if turn.speaker == "Moderator":
                        history_str += f"Moderator: {turn.text}\n"
                    elif turn.speaker == current_name:
                        history_str += f"Your Past Answer ({current_name}): {turn.text}\n"
                    else:
                        history_str += f"Other Participant ({turn.speaker}): {turn.text}\n"
                # 2. 기존 1:1 모드 (current_name 파라미터가 생략된 경우)
                else:
                    history_str += f"{turn.speaker}: {turn.text}\n"
            history_str += "</RECENT_TURNS_CONTEXT>"

        return history_str.strip()

    def add_interaction(self, speaker: str, text: str, target: str = "All"):
        new_turn = Turn(speaker=speaker, text=text, target=target)
        self.recent_turns.append(new_turn)

        turn_text = f"{speaker}: {text}\n"
        self._current_token_count += self._count_tokens(turn_text)

        total_tokens = self._summary_token_count + self._current_token_count
        if total_tokens > self.max_tokens:
            self._manage_memory_window()

    def _manage_memory_window(self):
        if len(self.recent_turns) <= 2:
            return

        chunk_size = max(1, len(self.recent_turns) // 2)
        turns_to_summarize = self.recent_turns[:chunk_size]
        self.recent_turns = self.recent_turns[chunk_size:]

        self._update_summary_batch(turns_to_summarize)
        self._resync_token_counts()

    # =====================================================================
    # 💡 [핵심 통합 2] 참여자 수를 자동 감지하여 요약 프롬프트 스위칭
    # =====================================================================
    def _update_summary_batch(self, turns: List[Turn]):
        new_dialogue_text = "\n".join(
            [f"{t.speaker}: {t.text}" for t in turns]
        )

        # 현재 요약하려는 청크 안에 'Moderator'를 제외하고 몇 명의 화자가 있는지 카운트
        unique_speakers = set(t.speaker for t in turns if t.speaker != "Moderator")
        is_multi_agent = len(unique_speakers) > 1

        if is_multi_agent:
            # 다자간(Multi-Agent) 전용 요약 프롬프트
            prompt = f"""
다음은 다자간 FGI 대화 내용입니다. 참여자 개개인의 독립된 성향, 가치관, 주요 의견이 서로 뒤섞이거나 유실되지 않도록 화자별 특징을 명확히 구분하여 요약본을 갱신해주세요.
답변은 오직 요약된 텍스트만 출력하세요.

[기존 요약]
{self.past_summary if self.past_summary else "없음"}

[추가된 대화 묶음]
{new_dialogue_text}
"""
        else:
            # 기존 1:1(Single-Agent) 전용 요약 프롬프트
            prompt = f"""
다음은 대화 내용입니다. 기존 요약본과 새로운 대화를 바탕으로, 대화의 핵심 맥락과 발화자의 주요 의견이 유실되지 않도록 요약본을 갱신해주세요.
답변은 오직 요약된 텍스트만 출력하세요.

[기존 요약]
{self.past_summary if self.past_summary else "없음"}

[추가된 대화 묶음]
{new_dialogue_text}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            self.past_summary = response.text.strip()
        except Exception as e:
            print(f"요약 모델 호출 중 에러 발생: {e}")
            self.past_summary += " | [일부 요약 실패]"

    def _resync_token_counts(self):
        self._summary_token_count = self._count_tokens(self.past_summary) if self.past_summary else 0
        recent_text = "".join([f"{t.speaker}: {t.text}\n" for t in self.recent_turns])
        self._current_token_count = self._count_tokens(recent_text)
