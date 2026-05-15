<GROUND_TRUTH>
- 프로모션 기준 가격: {{price}}
- 원본 프로모션 메시지: {{promo_info}}
</GROUND_TRUTH>

<GENERATED_RESULT>
- 시뮬레이션 결정(Decision): {{decision}}
- 구매 확률 점수(Score): {{probability_score}}
- 지불 의향 금액(Willingness to Pay): {{willingness_to_pay}}
- 핵심 이유 요약(Primary Reason): {{primary_reason}}
- 핵심 이유 추론 과정(Reasoning): {{reasoning}}
</GENERATED_RESULT>

<FLAGGED_ISSUES>
- 1차 시스템이 탐지한 에러 유형: {{logic_error_reasons}}
</FLAGGED_ISSUES>

<INSTRUCTION>
위 정보를 바탕으로 해당 생성 결과가 치명적인 결함이 있는지, 혹은 타당한지 판결해 주십시오.
</INSTRUCTION>