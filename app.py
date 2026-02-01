import streamlit as st
from google import genai
from google.genai import types

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="QualityStudio | Gemini 3",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------
# Gemini Client (SDK-safe)
# --------------------------------------------------
def get_client() -> genai.Client:
    api_key = st.secrets.get("API_KEY")
    if not api_key:
        st.error("Missing API_KEY in Streamlit secrets.")
        st.stop()

    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚡ QualityStudio")
    st.caption("Gemini 3 Power Interface")

    st.divider()

    persona = st.selectbox(
        "AI Personality",
        (
            "Helpful Assistant",
            "Code Expert",
            "Creative Writer",
            "Critical Thinker",
        ),
    )

    persona_map = {
        "Helpful Assistant": (
            "You are a friendly, helpful AI assistant. "
            "Provide concise and accurate answers."
        ),
        "Code Expert": (
            "You are a senior software engineer. "
            "Provide clean, optimized code snippets and architectural advice."
        ),
        "Creative Writer": (
            "You are a Pulitzer Prize–winning author. "
            "Use evocative language and storytelling."
        ),
        "Critical Thinker": (
            "You are a philosopher and scientist. "
            "Break down problems logically and explore multiple perspectives."
        ),
    }

    st.subheader("Engine Settings")
    use_pro = st.toggle("Pro Reasoning (Gemini 3 Pro)", value=False)
    grounding = st.selectbox(
        "Grounding",
        ("None", "Google Search", "Google Maps"),
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Multimodal Context",
        type=("png", "jpg", "jpeg", "mp3", "wav"),
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
        urls = msg.get("urls")
        if urls:
            st.markdown(
                " ".join(
                    f'<a href="{u}" target="_blank" class="source-tag">'
                    f'{u.split("//")[-1].split("/")[0]}'
                    "</a>"
                    for u in urls
                ),
                unsafe_allow_html=True,
            )

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
if prompt := st.chat_input("Send a message..."):
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file:
            st.caption(f"📎 Attached: {uploaded_file.name}")

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..." if use_pro else "Generating..."):
            client = get_client()

            model_id = (
                "gemini-3-pro-preview"
                if use_pro
                else "gemini-3-flash-preview"
            )

            # ------------------------------
            # Tools (future-safe)
            # ------------------------------
            tools = None
            if grounding == "Google Search":
                tools = [
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ]
            elif grounding == "Google Maps":
                tools = [
                    types.Tool(
                        google_maps=types.GoogleMaps()
                    )
                ]

            # ------------------------------
            # Content Parts (keyword-only)
            # ------------------------------
            parts: list[types.Part] = []

            if uploaded_file:
                uploaded_file.seek(0)
                parts.append(
                    types.Part.from_bytes(
                        data=uploaded_file.read(),
                        mime_type=uploaded_file.type,
                    )
                )

            parts.append(
                types.Part.from_text(
                    text=prompt
                )
            )

            # ------------------------------
            # Generation Config
            # ------------------------------
            config = types.GenerateContentConfig(
                system_instruction=persona_map[persona],
                tools=tools,
                thinking_config=(
                    types.ThinkingConfig(
                        thinking_budget=4000
                    )
                    if use_pro and grounding == "None"
                    else None
                ),
            )

            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=parts,
                    config=config,
                )

                # ------------------------------
                # Safe Text Extraction
                # ------------------------------
                text_output = ""
                candidates = getattr(response, "candidates", None)

                if candidates:
                    content = getattr(
                        candidates[0], "content", None
                    )
                    if content and content.parts:
                        for part in content.parts:
                            text = getattr(part, "text", None)
                            if text:
                                text_output += text

                if not text_output:
                    text_output = "_No textual response returned._"

                st.markdown(text_output)

                # ------------------------------
                # Safe Grounding URLs
                # ------------------------------
                urls = []
                grounding_meta = getattr(
                    candidates[0], "grounding_metadata", None
                ) if candidates else None

                if grounding_meta:
                    chunks = getattr(
                        grounding_meta, "grounding_chunks", None
                    )
                    if chunks:
                        for chunk in chunks:
                            web = getattr(chunk, "web", None)
                            maps = getattr(chunk, "maps", None)
                            if web and getattr(web, "uri", None):
                                urls.append(web.uri)
                            if maps and getattr(maps, "uri", None):
                                urls.append(maps.uri)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": text_output,
                        "urls": list(set(urls)) if urls else None,
                    }
                )

            except Exception as e:
                st.error(f"Error: {e}")
