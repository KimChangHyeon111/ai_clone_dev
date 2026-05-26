import re
import json
import numpy as np
from typing import List, Tuple
from google.genai import types

# 💡 깃허브 원본 클래스 임포트
from PersonaAgent import PersonaAgent
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
        embedder_model,            # 💡 [포인트 2] 누락되었던 임베딩 모델을 외부에서 명시적으로 주입 (DI)
        context_manager=None
    ):
        self.names = []
        self.agents = agents
        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.free_talk_sys_tmpl_path = free_talk_sys_tmpl_path
        self.free_talk_user_tmpl_path = free_talk_user_tmpl_path
        self.ctx = context_manager or ContextManager()
        self.embedder_model = embedder_model # 💡 명시적 할당

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
    
    def _parse_intent_and_target(self, user_input: str) -> Tuple[str, Optional[str], str, str]:
        """
        사용자 입력을 파싱하여 대화 모드, 고정 타겟, 발언자, 출력 메시지를 결정합니다.
        (순수 함수에 가깝게 동작하여 테스트 및 유지보수가 용이해집니다.)
        """
        input_upper = user_input.upper().strip()
        target_candidate = f"Customer_{input_upper}"
        
        # 기본적으로 현재 상태를 상속받음
        chat_mode = self.chat_mode
        targeted_agent = self.targeted_agent
        display_msg = ""
        target = None

        # A. 단일 알파벳 입력 시 (특정 유저 지목 모드 전환)
        if target_candidate in self.names and len(user_input.strip()) == 1:
            chat_mode = "TARGETED"
            targeted_agent = target_candidate
            target = target_candidate

        # B. 텍스트 내용이 있는 경우
        elif user_input.strip() != '':
            if '@FREE_TALK' in input_upper:
                chat_mode = "FREE_TALK"
                targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@FREE_TALK', '[자유 토론]', user_input, flags=re.IGNORECASE).strip()
            
            elif '@ALL' in input_upper or '@모두' in input_upper:
                chat_mode = "SEQUENTIAL"
                targeted_agent = None
                target = self._get_next_agent()
                display_msg = re.sub(r'@ALL|@모두', '', user_input, flags=re.IGNORECASE).strip()
            
            else:
                mentioned = False
                for name in self.names:
                    char = name.split('_')[1]
                    if re.search(fr'@{char}(?![a-zA-Z])', user_input, re.IGNORECASE):
                        chat_mode = "TARGETED"
                        targeted_agent = name
                        target = name
                        display_msg = re.sub(fr'@{char}(?![a-zA-Z])', f'@{name}', user_input, flags=re.IGNORECASE).strip()
                        mentioned = True
                        break
                
                # 멘션이 없으면 현재 모드 유지
                if not mentioned:
                    if chat_mode == "TARGETED" and targeted_agent:
                        target = targeted_agent
                        display_msg = user_input
                    else:
                        target = self._get_next_agent()
                        display_msg = user_input

        # C. 엔터(공백)만 친 경우
        else:
            if chat_mode == "TARGETED" and targeted_agent:
                target = targeted_agent
            else:
                target = self._get_next_agent()

        return chat_mode, targeted_agent, target, display_msg

    def process_turn(self, user_input: str, promo_info: str, DEBUG = False, temperature = 0.5) -> Tuple[str, str, str]:
        target = None
        display_msg = ""
        input_upper = user_input.upper()

        # =====================================================================
        # 1. # --- Step 1. 라우팅 로직 호출 및 상태 업데이트 ---
        # =====================================================================
        chat_mode, targeted_agent, target, display_msg = self._parse_intent_and_target(user_input)
        
        self.chat_mode = chat_mode
        self.targeted_agent = targeted_agent
        if display_msg:
            self.last_moderator_msg = display_msg

        current_name = target
        active_agent = self.agents[current_name]
        is_free_talk = (self.chat_mode == "FREE_TALK")
        # =====================================================================
        # 2. 동적 컨텍스트 세팅 및 질문 브로드캐스팅
        # =====================================================================
        active_sys_path = self.free_talk_sys_tmpl_path if is_free_talk else self.sys_tmpl_path
        active_user_path = self.free_talk_user_tmpl_path if is_free_talk else self.user_tmpl_path

        if display_msg:
            self.broadcast(speaker="Moderator", text=display_msg, exclude=current_name)
        elif not is_free_talk:
            virtual_prompt = f"{current_name}님, 앞서 드린 질문('{self.last_moderator_msg}')에 대해 응답해주세요."
            self.broadcast(speaker="Moderator", text=virtual_prompt, exclude=current_name)
        # =====================================================================
        # 3. 모델 발화 및 결과 적재
        # =====================================================================
        embed_query = display_msg if display_msg else (self.last_moderator_msg if not is_free_talk else "[자유 토론]")
        query_vector = self.embedder_model.encode([embed_query], normalize_embeddings=True).astype(np.float32)

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


        other_names_str = ", ".join([n for n in self.names if n != current_name])

        valid_data = FGIDataSchema(
            ms=active_agent.profile.get("ms_core_mindset", ""),
            ml=combined_ml,
            pixel=combined_pixel,
            promo_info=promo_info,
            conversation_history=history_str,
            user_input=display_msg,               # 순수 질의/명령어 텍스트만 주입 (결합 없음)
            current_name=current_name,            # 현재 에이전트 이름 전달
            other_names=other_names_str,          # 자유 토론용 상대방 이름 목록 전달
            last_moderator_msg=self.last_moderator_msg # 가상 프롬프트용 백업 질의 전달
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
