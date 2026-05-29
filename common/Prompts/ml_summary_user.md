<INPUT_DATA>
    {% if PAST_SUMMARY %}
    <PAST_SUMMARY>
        {{PAST_SUMMARY}}
    </PAST_SUMMARY>
    {% endif %}

    <NEW_TRANSACTIONS>
        {{NEW_DATA}}
    </NEW_TRANSACTIONS>
</INPUT_DATA>

<CURRENT_SITUATION>
    [Behavioral Guideline: Your identity is {{AGENT_NAME}}. Based on the data above, summarize your transaction history and shopping patterns in a single, concise paragraph.]
</CURRENT_SITUATION>

<OUTPUT_FORMAT>
- Do NOT output any introductory or concluding phrases.
- Do NOT wrap your response in markdown code blocks (```). Return pure text paragraphs.
- Keep it highly concentrated on factual purchasing behaviors.
</OUTPUT_FORMAT>