<CUSTOMER_DATA>
    <TRANSACTION_HISTORY_ML>
        {{ML}}
    </TRANSACTION_HISTORY_ML>
    <MICRO_ATTRIBUTES_PIXEL>
        {{PIXEL}}
    </MICRO_ATTRIBUTES_PIXEL>
</CUSTOMER_DATA>

<FGI_SESSION>
    <TARGET_PROMOTION>
        {{PROMOTION_INFO}}
    </TARGET_PROMOTION>
    <CONVERSATION_HISTORY>
        {{CONVERSATION_HISTORY}}
    </CONVERSATION_HISTORY>
    <CURRENT_INTERVIEWER_INPUT>
        {{USER_INPUT}}
    </CURRENT_INTERVIEWER_INPUT>
</FGI_SESSION>
<OUTPUT_FORMAT>
You MUST strictly follow the JSON schema below. Output ONLY a valid JSON string.
Constraint: All string values in the JSON (except for 'response') should be in English. The 'response' field MUST be in Korean.
Constraint: All double quotes inside the strings MUST be escaped with a backslash (").
{
  "internal_state": {
    "reasoning": "Step 1: Identify the exact core question in [CURRENT_INTERVIEWER_INPUT]. Step 2: Analyze Interviewer's attitude. Step 3: Analyze background information in [CUSTOMER_DATA], [TARGET_PROMOTION] and [CONVERSATION_HISTORY]. Step 4: Decide how the persona reacts to the exact question, attitude and backgroud information. Step 5: Formulate the final answer.",
    "interest_level": "Integer between 0 and 100",
    "emotion": "Current emotional state (e.g., Curious, Indifferent, Skeptical, etc.)"
  },
  "response": "Actual dialogue to the interviewer in Korean (Colloquial, reflecting internal_state)"
}
</OUTPUT_FORMAT>