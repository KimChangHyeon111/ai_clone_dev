<INPUT_DATA>
    {% if PAST_SUMMARY %}
    <PAST_SUMMARY>
        {{PAST_SUMMARY}}
    </PAST_SUMMARY>
    {% endif %}

    <NEW_DIALOGUE>
        {{NEW_DATA}}
    </NEW_DIALOGUE>
</INPUT_DATA>

<CURRENT_SITUATION>
    Based on the data above, summarize the conversation from a 3rd-person perspective (e.g., [참여자] Customer_A는 ~라고 언급했습니다.)
</CURRENT_SITUATION>

<OUTPUT_FORMAT>
- Do NOT output any introductory or concluding phrases (e.g., "Here is the summary").
- Do NOT wrap your response in markdown code blocks (```). Return pure text paragraphs.
- Ensure each participants' points are clearly distinguishable.
- MUST be written in KOREAN.
</OUTPUT_FORMAT>