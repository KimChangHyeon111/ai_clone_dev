import numpy as np

# GCP SDK
from google import genai
from google.genai import types

from ContextManager import ContextManager, FGIDataSchema
from HybridMemoryManager import HybridMemoryManager

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

        # =====================================================================
        # 💡 [추가] 에이전트 장기 기억 장치 (Vector DB) 탑재
        # =====================================================================
        self.full_ml_history = fgi_profile.get("full_ml", [])
        self.ml_embeddings = fgi_profile.get("ml_embeddings", np.array([]))

        self.full_pixel = fgi_profile.get("full_pixel", [])
        self.pixel_embeddings = fgi_profile.get("pixel_embeddings", np.array([]))

    def process_interviewer_turn(self, interviewer_input: str, promo_info: str = "프로모션 정보 없음") -> str:
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

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.6
            )
        )
        response_text = response.text.strip()
        self.memory.add_interaction(interviewer_input, response_text)

        return response_text
