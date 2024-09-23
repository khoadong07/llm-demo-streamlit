import json

import streamlit as st
import time
import pandas as pd
from deep_infa import call_deep_infra

def process_call_llm(context, content):
    result_llm, input_token, output_token = call_deep_infra(context, content)
    return result_llm, input_token, output_token

def normalize_field_name(field):
    field_map = {
        "sentiment": "Sentiment",
        "extend_keyword": "Extend Keyword",
        "severity": "Severity",
        "emotion": "Emotion",
        "polarity": "Polarity",
        "intensity": "Intensity",
        "category": "Category",
        "industry": "Industry",
        "angle": "Angle",
        "intent": "Intent",
        "purpose": "Purpose",
        "tone": "Tone",
        "brand_attribute": "Brand Attribute",
        "spam": "Spam",
        "advertisement": "Advertisement",
        "opinion_expressed": "Opinion Expressed",
        "emotional_tone": "Emotional Tone",
        "feedback_provided": "Feedback Provided",
        "insight_provided": "Insight Provided"
    }
    return field_map.get(field, field)

st.title('LLM Demo API')

context_text = st.text_area("Enter your context here:")
input_text = st.text_area("Enter your content here:")

if st.button('Run'):
    if input_text.strip() == "":
        st.warning("Please input some text before running.")

    if context_text.strip() == "":
        st.warning("Please input some text before running.")

    else:
        st.subheader("Analysis Result")
        with st.spinner('Processing...'):
            start_time = time.time()

            result_llm, input_token, output_token = process_call_llm(context=context_text, content=input_text)

            end_time = time.time()
            processing_time = end_time - start_time

        st.write(f"Processing time: {processing_time:.2f} seconds")
        st.write(f"Input token: {input_token} tokens")
        st.write(f"Output token: {output_token} tokens")
        if isinstance(result_llm, list) and len(result_llm) > 0:
            result_llm = result_llm[0]

        st.text(result_llm)
        st.json(result_llm)