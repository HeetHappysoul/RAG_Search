my_key = os.getenv("GEMINI_API_KEY")
if not my_key:
    st.error("GEMINI_API_KEY not found — check .env location and filename")
    st.stop()