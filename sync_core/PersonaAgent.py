import numpy as np
import json
# GCP SDK
from google import genai
from google.genai import types

from sync_core.ContextManager import ContextManager
from sync_core.HybridMemoryManager import HybridMemoryManager
from common.schemas import FGIDataSchema
from common.utils import *

class PersonaAgent:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        srg_key: str,
        fgi_profile: dict,
        sys_tmpl_path: str,
        user_tmpl_path: str,
        location: str = "global",
        model_name: str ="gemini-2.5-flash-lite",
        context_manager = None        
    ):
        self.srg_key = srg_key

        self.sys_tmpl_path = sys_tmpl_path
        self.user_tmpl_path = user_tmpl_path
        self.ctx = context_manager or ContextManager()

        self.memory = HybridMemoryManager()
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

        self.profile = {
            "ms_core_mindset": fgi_profile.get("ms", ""),
            "ml_transaction_history": fgi_profile.get("ml", [])
        }
        self.is_valid = bool(self.profile["ms_core_mindset"])
        self.model_name = model_name
        # =====================================================================
        # 💡 [추가] 에이전트 장기 기억 장치 (Vector DB) 탑재
        # =====================================================================
        self.full_ml_history = fgi_profile.get("full_ml", [])
        self.ml_embeddings = fgi_profile.get("ml_embeddings", np.array([]))

        self.full_pixel = fgi_profile.get("full_pixel", [])
        self.pixel_embeddings = fgi_profile.get("pixel_embeddings", np.array([]))

    def process_interviewer_turn(self, interviewer_input: str, promo_info: str = "프로모션 정보 없음", DEBUG: bool = False) -> str:
        if not self.is_valid:
            return "[시스템 알림] 데이터가 부족합니다."

        valid_data = FGIDataSchema(
            ms=self.profile["ms_core_mindset"],
            ml=self.profile["ml_transaction_history"],
            promo_info=promo_info,
            conversation_history=self.memory.get_formatted_history(),
            user_input=interviewer_input
        )

        sys_inst, user_content = self.ctx.build_prompt(
            sys_path=self.sys_tmpl_path,
            user_path=self.user_tmpl_path,
            data=valid_data
        )

        prompt = f"Interviewer: {user_content}\nCustomer: "
        # =====================================================================
        # 💡 2. [디버깅 추가] 모델 호출 직전에 프롬프트 전체 출력
        # =====================================================================
        if DEBUG:
            print("\n" + "="*40 + f" [DEBUG: {self.srg_key}(PersonaAgent) 프롬프트] " + "="*40)
            print("-" * 100)
            print("[SYSTEM INSTRUCTION]")
            print(sys_inst)
            print("-" * 100)
            print("[USER CONTENT]")
            print(prompt)
            print("="*105 + "\n")
        # =====================================================================
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.6
            )
        )



        try:
            clean_txt = clean_json_string(response.text)
            res_data = json.loads(clean_txt)
            # JSON 스키마에서 실제 발화인 'response'만 추출
            response_text = res_data.get("response", response.text).strip()
        except json.JSONDecodeError:
            # 파싱 실패 시 원본 텍스트라도 반환하도록 폴백
            response_text = response.text.strip()
        # =================================================================

        self.memory.add_interaction(interviewer_input, response_text)
        return response_text
