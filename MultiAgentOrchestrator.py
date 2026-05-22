import re
import json
import numpy as np
from typing import List, Tuple
from google.genai import types

# 💡 깃허브 원본 클래스 임포트
from SimulationEngine import PersonaAgent
from ContextManager import FGIDataSchema, ContextManager

class _SharedMemoryRouter:
    """메인 실행 루프에서 .shared_memory.add_interaction 호출 시 호환성을 맞추기 위한 프록시"""
    def __init__(self, orchestrator):
        self.orch = orchestrator
    def add_interaction(self, speaker: str, text: str, target: str = "All"):
        self.orch.broadcast(speaker=speaker, text=text, target=target)


class MultiAgentOrchestrator:
    def __init__(
        self,
        agents,
        srg_keys: List[str],
        fgi_profiles: List[dict],
        sys_tmpl_path: str,
        user_tmpl_path: str,
        free_talk_sys_tmpl_path: str,
        free_talk_user_tmpl_path: str,
        project_id: str,
        bucket_name: str,
        context_manager=None
    ):
        self.names = []
        self.agents = agents
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.free_talk_sys_tmpl_path = free_talk_sys_tmpl_path
        self.free_talk_user_tmpl_path = free_talk_user_tmpl_path
        self.ctx = context_manager or ContextManager()

        for i, srg_key in enumerate(srg_keys):
            name = f"Customer_{chr(65+i)}"
            self.names.append(name)

            # 💡 [핵심 추가] PersonaAgent 초기화 시 누락될 수 있는 초기 PIXEL 리스트 확보
            base_pixel = fgi_profiles[i].get("pixel", [])
            if isinstance(base_pixel, str):
                base_pixel = [base_pixel] # 구조화된 리스트 형태 강제 보장
            self.agents[name].profile["pixel"] = base_pixel

        self.chat_mode = "SEQUENTIAL"
        self.targeted_agent = None
        self.last_agent = self.names[-1] if self.names else None
        self.last_moderator_msg = ""
        self.shared_memory = _SharedMemoryRouter(self)

    def broadcast(self, speaker: str, text: str, target: str = "All", exclude: str = None):
        """방 안의 모든 에이전트 메모리에 대화를 전파합니다."""
        for agent_name, agent in self.agents.items():
            if agent_name != exclude:
                try:
                    agent.memory.add_interaction(speaker=speaker, text=text, target=target)
                except TypeError:
                    agent.memory.add_interaction(interviewer_input=speaker, customer_response=text)

    def _get_next_agent(self) -> str:
        if not self.names: return ""
        idx = self.names.index(self.last_agent)
        return self.names[(idx + 1) % len(self.names)]

    def process_turn(self, user_input: str, promo_info: str, DEBUG = False, temperature = 0.5) -> Tuple[str, str, str]:
        target = None
        display_msg = ""
        input_upper = user_input.upper()

        # =====================================================================
        # 1. 3-Way 라우팅 로직
        # =====================================================================
        target_candidate = f"Customer_{input_upper}"

        if target_candidate in self.names and len(user_input.strip()) == 1:
            self.chat_mode = "TARGETED"
            self.targeted_agent = target_candidate
            target = target_candidate

        elif user_input.strip() != '':
            if '@FREE_TALK' in input_upper:
                self.chat_mode = "FREE_TALK"
                self.targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@FREE_TALK', '[자유 토론]', user_input, flags=re.IGNORECASE).strip()
            elif '@ALL' in input_upper or '@모두' in input_upper:
                self.chat_mode = "SEQUENTIAL"
                self.targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@ALL|@모두', '', user_input, flags=re.IGNORECASE).strip()
            else:
                mentioned = False
                for name in self.names:
                    char = name.split('_')[1]
                    if re.search(fr'@{char}(?![a-zA-Z])', user_input, re.IGNORECASE):
                        self.chat_mode = "TARGETED"
                        self.targeted_agent = name
                        target = name
                        display_msg = re.sub(fr'@{char}(?![a-zA-Z])', f'@{name}', user_input, flags=re.IGNORECASE).strip()
                        mentioned = True
                        break
                if not mentioned:
                    if self.chat_mode == "TARGETED" and self.targeted_agent:
                        target = self.targeted_agent
                        display_msg = user_input
                    else:
                        target = self._get_next_agent()
                        display_msg = user_input
        else:
            if self.chat_mode == "TARGETED" and self.targeted_agent:
                target = self.targeted_agent
            else:
                target = self._get_next_agent()

        if display_msg:
            self.last_moderator_msg = display_msg

        # =====================================================================
        # 2. 동적 컨텍스트 세팅 및 질문 브로드캐스팅
        # =====================================================================
        current_name = target
        active_agent = self.agents[current_name]
        is_free_talk = (self.chat_mode == "FREE_TALK")

        active_sys_path = self.free_talk_sys_tmpl_path if is_free_talk else self.sys_tmpl_path
        active_user_path = self.free_talk_user_tmpl_path if is_free_talk else self.user_tmpl_path

        anti_parroting_tag = "\n[SYSTEM RULE (CRITICAL): 다른 참여자의 발언 구조, 단어, 표현을 절대 복사하거나 동조하지 마세요. 의견이 비슷하다면 짧게 동의하고, 같은 단어를 반복하지 마세요.]"

        if display_msg:
            current_context_input = display_msg + anti_parroting_tag
            self.broadcast(speaker="Moderator", text=display_msg, exclude=current_name)
        else:
            if is_free_talk:
                current_context_input = "[자유 토론 진행 중. 앞 사람의 의견에 동의/반박/질문 하세요.]" + anti_parroting_tag
            else:
                virtual_prompt = f"{current_name}님, 앞서 드린 질문('{self.last_moderator_msg}')에 대해 응답해주세요 (요청하지 않은 인사, 자기소개 생략, 본론만 즉답 요망)"
                current_context_input = virtual_prompt + anti_parroting_tag
                self.broadcast(speaker="Moderator", text=virtual_prompt, exclude=current_name)

        # =====================================================================
        # 3. 모델 발화 및 결과 적재
        # =====================================================================
        query_vector = self.embedder_model.encode([current_context_input], normalize_embeddings=True).astype(np.float32)

        dynamic_ml = self._retrieve_dynamic_context(
            query_vector=query_vector, embeddings=active_agent.ml_embeddings, raw_data=active_agent.full_ml_history, threshold=0.7
        )
        dynamic_pixel = self._retrieve_dynamic_context(
            query_vector=query_vector, embeddings=active_agent.pixel_embeddings, raw_data=active_agent.full_pixel, threshold=0.7
        )

        # 💡 2. [핵심] HybridMemoryManager를 활용한 맥락 누적 및 추출
        active_agent.memory.add_dynamic_context(dynamic_ml, dynamic_pixel)

        base_ml = active_agent.profile.get("ml_transaction_history", [])
        base_pixel = active_agent.profile.get("pixel", [])

        combined_ml = active_agent.memory.get_combined_ml(base_ml)
        combined_pixel = active_agent.memory.get_combined_pixel(base_pixel)

        # 3. 데이터 바인딩
        try:
            history_str = active_agent.memory.get_formatted_history(current_name=current_name)
        except TypeError:
            history_str = active_agent.memory.get_formatted_history()

        valid_data = FGIDataSchema(
            ms=active_agent.profile.get("ms_core_mindset", ""),
            ml=combined_ml,
            pixel=combined_pixel,
            promo_info=promo_info,
            conversation_history=history_str,
            user_input=current_context_input
        )

        sys_inst, user_content = self.ctx.build_prompt(sys_path=active_sys_path, user_path=active_user_path, data=valid_data)

        # =====================================================================
        # 💡 [디버깅 추가] 프롬프트 전체 출력
        # =====================================================================
        if DEBUG:
            print("\n" + "="*40 + f" [DEBUG: {current_name}에게 주입된 프롬프트] " + "="*40)
            print(f"[ML 누적치]: {len(active_agent.memory.recent_ml)}개 (압축 요약: {'있음' if active_agent.memory.past_ml_summary else '없음'})")
            print(f"[PIXEL 누적치]: {len(active_agent.memory.recent_pixel)}개 (압축 요약: {'있음' if active_agent.memory.past_pixel_summary else '없음'})")
            print("-" * 100)
            print("[SYSTEM INSTRUCTION]")
            print(sys_inst)
            print("-" * 100)
            print("[USER CONTENT]")
            print(user_content)
            print("="*105 + "\n")
        # =====================================================================
        # =====================================================================
        response = active_agent.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=temperature,
                response_mime_type="application/json"
            )
        )

        try:
            res_data = json.loads(response.text)
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            response_text = response.text.strip()

        try:
            active_agent.memory.add_interaction(speaker=current_name, text=response_text, target="Group" if is_free_talk else "All")
        except TypeError:
            active_agent.memory.add_interaction(interviewer_input=current_name, customer_response=response_text)

        self.broadcast(speaker=current_name, text=response_text, exclude=current_name)

        self.last_agent = current_name
        return current_name, display_msg, response_text
