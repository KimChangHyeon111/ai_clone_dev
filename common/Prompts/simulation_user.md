<INPUT_DATA>
    <CUSTOMER_BEHAVIORAL_DATA>
        <ML_HISTORY>{{ML}}</ML_HISTORY>
        <PIXEL_ATTRIBUTES>{{PIXEL}}</PIXEL_ATTRIBUTES>
    </CUSTOMER_BEHAVIORAL_DATA>

    <PROMOTION_UNDER_TEST>
        {{PROMOTION_INFO}}
    </PROMOTION_UNDER_TEST>
</INPUT_DATA>

<TASK>
Perform a high-speed simulation of this customer's reaction.
Provide a quantitative score, a concise qualitative reason, willingness to pay, and actionable feedback for the decision.
</TASK>

<EXAMPLES>
    <EXAMPLE_1>
        Input_Context:
            [CORE_MINDSET: "실용성과 가성비를 극도로 중시함. 멤버십이나 정기 결제에 대한 거부감이 강함."]
            [ML_HISTORY: [{"상품명": "할인 편의점 도시락", "price": 4500}, {"상품명": "1+1 생수", "price": 3000}]]
            [PIXEL_ATTRIBUTES: "1인 가구, 가격 민감도 최상, 쿠폰 사용 빈도 높음"]
        Input_Promotion: "연간 15만원 가입비의 프리미엄 쇼핑 멤버십 (무료 발렛 및 라운지 이용권 포함)"
        Output_JSON: {
            "reasoning": "CORE_MINDSET reveals a strong aversion to subscriptions and a focus on practicality. This is heavily supported by ML_HISTORY showing consistent low-cost single-item purchases, and PIXEL_ATTRIBUTES indicating extreme price sensitivity. The 150k KRW fee creates a massive psychological barrier that conflicts entirely with the user's core values.",
            "decision": "REJECT",
            "probability_score": 0.05,
            "primary_reason": "고객의 가성비 중시 성향 및 저가 위주 구매 이력과 제안된 고가 멤버십의 심각한 불일치",
            "feedback_keyword": ["가격 저항"],
            "willingness_to_pay": 20000,
            "improvement_suggestion": "프리미엄 부가서비스(발렛/라운지)를 제외하고 연회비를 3만원 이하로 낮춘 라이트(Lite) 멤버십 옵션 추가"
        }
    </EXAMPLE_1>
    <EXAMPLE_2>
        Input_Context:
            [CORE_MINDSET: "시간 효율성과 편의성을 최우선으로 하며, 프리미엄 서비스에 대한 지불 의사가 높음. 발렛/라운지 등 대기 시간을 줄여주는 서비스 선호."]
            [ML_HISTORY: [{"상품명": "호텔 애프터눈 티 세트", "price": 120000}, {"상품명": "백화점 VIP 발렛 서비스 이용", "price": 0}, {"상품명": "프리미엄 수입차 소모품", "price": 450000}]]
            [PIXEL_ATTRIBUTES: "고소득층 추정, 편의성 민감도 최상, 오프라인 백화점 헤비 쇼퍼"]
        Input_Promotion: "연간 15만원 가입비의 프리미엄 쇼핑 멤버십 (무료 발렛 및 라운지 이용권 포함)"
        Output_JSON: {
            "reasoning": "CORE_MINDSET dictates that time efficiency and convenience are top priorities, making the user highly receptive to premium services. ML_HISTORY confirms high-value lifestyle spending and existing valet usage, while PIXEL_ATTRIBUTES suggests a high-income profile. The 150k KRW fee is an easily acceptable cost for the time-saving benefits provided, aligning perfectly with their mindset.",
            "decision": "ACCEPT",
            "probability_score": 0.92,
            "primary_reason": "시간 효율과 편의를 중시하는 고객의 라이프스타일과 발렛/라운지 혜택의 완벽한 일치",
            "feedback_keyword": ["편의성, 시간 절약"],
            "willingness_to_pay": 200000,
            "improvement_suggestion": "주말 등 혼잡 시간대 발렛 우선 예약 권한을 추가하여 서비스의 희소성 및 프리미엄 경험 강화"
        }
    </EXAMPLE_2>
    <EXAMPLE_3>
        Input_Context:
            [CORE_MINDSET: "실익이 분명하다면 구독을 고려하지만, 혜택이 중복되는 것을 극도로 싫어함. 주로 온라인 쇼핑을 이용하며 가끔씩 오프라인 쇼핑을 병행함."]
            [ML_HISTORY: [{"상품명": "온라인 유료 멤버십 월회비", "price": 4900}, {"상품명": "브랜드 의류", "price": 89000}, {"상품명": "무선 이어폰", "price": 210000}]]
            [PIXEL_ATTRIBUTES: "온라인 쇼핑 선호, 30대 직장인, 혜택 체감도 중시"]
        Input_Promotion: "연간 15만원 가입비의 프리미엄 쇼핑 멤버십 (무료 발렛 및 라운지 이용권 포함)"
        Output_JSON: {
            "reasoning": "CORE_MINDSET indicates a willingness to subscribe but a strict reluctance towards overlapping benefits and a strong preference for online shopping. While ML_HISTORY shows mid-to-high ticket purchases and online subscriptions, and PIXEL_ATTRIBUTES confirms online preference, the offline-heavy premium benefits (valet/lounge) clash with the user's online-centric mindset. The high annual fee requires clearer justification.",
            "decision": "HOLD",
            "probability_score": 0.45,
            "primary_reason": "프리미엄 혜택의 매력은 인지하나, 주로 온라인 쇼핑을 이용하는 패턴상 15만원의 연회비만큼 혜택을 체감할지 확신 부족",
            "feedback_keyword": ["혜택 체감도, 활용성 고민"],
            "willingness_to_pay": 80000,
            "improvement_suggestion": "오프라인 혜택 외에 온라인 쇼핑 시 사용 가능한 고정 적립금이나 무료 배송 쿠폰 팩을 포함하여 온-오프라인 통합 혜택 설계"
        }
        </EXAMPLE_3>
</EXAMPLES>

<OUTPUT_FORMAT>
You MUST strictly follow the JSON schema below. Output ONLY a valid JSON string.
- Constraint: 'reasoning' should be in English.
- Constraint: 'primary_reason', 'feedback_keyword', 'improvement_suggestion' MUST be in Korean.
- Constraint: 'willingness_to_pay' MUST be an integer
- Constraint: Ensure all double quotes are escaped (").
- Constraint: Do not wrap the output in Markdown formatting (e.g., '``json). Output raw JSON string only.

{
  "reasoning": "Logical synthesis of CORE_MINDSET, ML_HISTORY, and PIXEL_ATTRIBUTES (English)",
  "decision": "One of [ACCEPT, REJECT, HOLD]",
  "probability_score": "Float between 0.00 and 1.00",
  "primary_reason": "결정적 사유 1문장 요약 (Korean)",
  "feedback_keyword": "핵심 키워드 1~2개 (Korean)",
  "willingness_to_pay": "수용 가능한 최대 금액(INT)",
  "improvement_suggestion": "프로모션 전환율을 높이기 위한 구체적인 개선 제안 1문장 (Korean)"
}
</OUTPUT_FORMAT>