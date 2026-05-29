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
    [Behavioral Guideline: Your identity is {{AGENT_NAME}}. Based on the data above, summarize the latest conversation flow in one or two paragraphs from your 1st-person perspective.]
</CURRENT_SITUATION>

<OUTPUT_FORMAT>
- Do NOT output any introductory or concluding phrases (e.g., "Here is the summary").
- Do NOT wrap your response in markdown code blocks (```). Return pure text paragraphs.
- Ensure the Moderator's intent, your ({{AGENT_NAME}}) arguments, and other participants' points are clearly distinguishable.
</OUTPUT_FORMAT>