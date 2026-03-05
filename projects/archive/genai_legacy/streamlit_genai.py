import streamlit as st  # Import Streamlit
from google.genai import Client  # Import the Gemini Client

# --- 1. SET PAGE CONFIGURATION AND TITLE ---
st.set_page_config(page_title="Gemini AI Simple Query App", layout="centered")
st.title("💡 Gemini AI Query App")
st.markdown(
    "Enter a question below and press **Generate** to get a response from `gemini-2.5-flash`."
)

# --- 2. INITIALIZE CLIENT (CACHED) ---


# Use st.cache_resource to initialize the client once per session
@st.cache_resource
def get_gemini_client():
    # The client automatically uses the GEMINI_API_KEY environment variable.
    try:
        return Client()
    except Exception as e:
        # Display an error if the key is missing or invalid
        st.error(
            f"Error initializing Gemini Client. Please check your GEMINI_API_KEY environment variable. Details: {e}"
        )
        return None


client = get_gemini_client()

# --- 3. GET USER INPUT AND GENERATE CONTENT ---

if client:
    # Create an interactive text input field
    user_prompt = st.text_input(
        "Your Question:",
        placeholder="Which religion's followers have caused the most deaths in South America",
    )

    # Create an interactive button
    if st.button("Generate Response"):
        if user_prompt:
            # Display a spinner while waiting for the API response
            with st.spinner("Generating content..."):
                try:
                    # Call the Gemini API with the user's input
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=user_prompt,
                    )

                    # Display the response using Streamlit's markdown writer
                    st.subheader("✅ AI Response:")
                    st.info(response.text)

                except Exception as e:
                    # Handle API errors (e.g., Quota Exceeded 429)
                    st.error(f"An API Error Occurred: {e}")
        else:
            st.warning("Please enter a question before generating a response.")
