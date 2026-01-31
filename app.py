import streamlit as st
import google.generativeai as genai
from google.generativeai.types import Part, GenerateContentConfig, ThinkingConfig, Tool, GoogleSearch, GoogleMaps
from dotenv import load_dotenv

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="QualityStudio | Gemini 3",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# Gemini Client
# --------------------------------------------------
def get_client():
    api_key = st.secrets.get("API_KEY")  # Read from Streamlit Secrets
    if not api_key:
        st.error("Missing API_KEY in Streamlit Secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    return genai

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚡ QualityStudio")
    st.caption("Gemini 3 Power Interface")
    st.divider()

    persona = st.selectbox(
        "AI Personality",
        ["Helpful Assistant", "Code Expert", "Creative Writer", "Critical Thinker"]
    )

    persona_map = {
        "Helpful Assistant": "You are a friendly, helpful AI assistant. Provide concise and accurate answers.",
        "Code Expert": "You are a senior software engineer. Provide clean, optimized code snippets and architectural advice.",
        "Creative Writer": "You are a Pulitzer-prize winning author. Use evocative language and storytelling.",
        "Critical Thinker": "You are a philosopher and scientist. Break down problems logically and explore multiple perspectives."
    }

    st.subheader("Engine Settings")
    use_pro = st.toggle("Pro Reasoning (Gemini 3 Pro)", value=False)
    grounding = st.selectbox("Grounding", ["None", "Google Search", "Google Maps"])
    st.divider()

    uploaded_file = st.file_uploader(
        "Multimodal Context",
        type=["png", "jpg", "jpeg", "mp3", "wav"]
    )

    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("urls"):
            st.markdown(
                " ".join(
                    f'<a href="{u}" target="_blank" class="source-tag">{u.split("//")[-1].split("/")[0]}</a>'
                    for u in msg["urls"]
                ),
                unsafe_allow_html=True
            )

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
if prompt := st.chat_input("Send a message..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file:
            st.caption(f"📎 Attached: {uploaded_file.name}")

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..." if use_pro else "Generating..."):
            client = get_client()
            model_id = "gemini-3-pro-preview" if use_pro else "gemini-3-flash-preview"

            # Tools
            tools = []
            if grounding == "Google Search":
                tools.append(Tool(google_search=GoogleSearch()))
            elif grounding == "Google Maps":
                tools.append(Tool(google_maps=GoogleMaps()))

            # Build content parts
            parts = []
            if uploaded_file:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                parts.append(
                    Part.from_bytes(
                        data=file_bytes,
                        mime_type=uploaded_file.type
                    )
                )

            parts.append(Part.from_text(text=prompt))

            config = GenerateContentConfig(
                system_instruction=persona_map[persona],
                tools=tools if tools else None,
                thinking_config=ThinkingConfig(thinking_budget=4000)
                if use_pro and grounding == "None"
                else None
            )

            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=parts,
                    config=config
                )

                # -------- Extract text safely --------
                text_output = ""
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            text_output += part.text

                if not text_output:
                    text_output = "_No textual response returned._"

                st.markdown(text_output)

                # -------- Extract grounding links safely --------
                urls = []
                candidate = response.candidates[0]
                grounding_meta = getattr(candidate, "grounding_metadata", None)

                if grounding_meta and grounding_meta.grounding_chunks:
                    for chunk in grounding_meta.grounding_chunks:
                        if chunk.web:
                            urls.append(chunk.web.uri)
                        if chunk.maps:
                            urls.append(chunk.maps.uri)

                final_urls = list(set(urls))

                # -------- Save assistant message ONCE --------
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": text_output,
                    "urls": final_urls if final_urls else None
                })

            except Exception as e:
                st.error(f"Error: {e}")
