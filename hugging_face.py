import streamlit as st
import requests
import json

def generate_content(topic, api_key, model_id, output_format, tone):
    """
    Calls the Hugging Face Inference API to generate content.
    """
    if not api_key:
        st.error("Please enter your Hugging Face API key in the sidebar to use this feature.")
        return None

    # Hugging Face API URL for a specific model
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Dynamically build the prompt based on user selections
    prompt = f"""
    Act as a world-class market analyst and communication expert.
    Your task is to generate a comprehensive and insightful report on the topic: '{topic}'.
    
    The response should be in the format of a '{output_format}'.
    The tone of the response must be '{tone}'.
    If the requested format is a 'LinkedIn Post', conclude with 3-5 relevant and popular hashtags.
    
    Ensure the output is well-structured, insightful, and ready for a professional audience.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True
        }
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        # Safely extract the generated text from the API response
        generated_text = result[0].get('generated_text')

        if not generated_text:
            st.error("The model did not return a valid response. Please try a different topic or adjust your settings.")
            st.json(result) # Show raw response for debugging
            return None

        return generated_text

    except requests.exceptions.HTTPError as e:
        st.error(f"API request failed with status code: {e.response.status_code}")
        try:
            error_details = e.response.json()
            error_message = error_details.get('error', 'No specific message.')
            st.error(f"API Error Details: {error_message}")
            if e.response.status_code == 401:
                st.warning("This is an authentication error (401). Please check that your API key is correct. "
                           "The API key should be a valid token copied from your Hugging Face account settings.")
            elif e.response.status_code == 403:
                st.warning("This is a permissions error. Please ensure your API key is valid and has permission to access the selected model. "
                           "Some models may require a pro account or be private. Double-check your API key and the model permissions on Hugging Face.")
            elif e.response.status_code == 404:
                st.warning("This is a 'Not Found' error. Please verify that the Hugging Face Model ID is correct and that the model exists.")
        except json.JSONDecodeError:
            st.error(f"Failed to parse error response from the API. Response text: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network request failed: {e}")
        st.warning("Please check your internet connection and if firewalls are blocking the request to the Hugging Face API.")
        return None
    except (IndexError, KeyError) as e:
        st.error(f"Failed to parse AI response. The model's output may be malformed. Error: {e}")
        st.json(locals().get('result', {"Error": "Could not retrieve result from API response."}))
        return None

# --- Initialize session state ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# --- UI Layout ---
st.set_page_config(page_title="Hugging Face News Assistant", page_icon="🤗", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=100) # Hugging Face Logo
with col2:
    st.title("Hugging Face Insight Assistant")
    st.markdown("### Generate Professional Content with Hugging Face Models")

st.markdown("---")

# --- Sidebar for API Key and Controls ---
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "Hugging Face API Key",
    type="password",
    help="Get your free key from your Hugging Face account settings.",
    value=st.session_state.api_key
)
# Update session state when the user types in a new key
if api_key_input:
    st.session_state.api_key = api_key_input

st.sidebar.header("📝 Content Controls")
model_id = st.sidebar.text_input(
    "Hugging Face Model ID",
    value="gpt2",
    help="e.g., gpt2, google/flan-t5-large"
)
output_format = st.sidebar.selectbox(
    "Select Output Format",
    ("LinkedIn Post", "Executive Summary", "Email Briefing", "Key Talking Points")
)
tone = st.sidebar.selectbox(
    "Select Tone of Voice",
    ("Professional", "Analytical", "Casual", "Optimistic", "Cautious")
)

# --- Main Content Area ---

# Only proceed if an API key has been entered
if not st.session_state.api_key:
    st.warning("👋 Welcome! Please enter your Hugging Face API key in the sidebar to begin.")
    st.stop()

st.subheader("Enter a topic to begin your analysis")
topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")


if st.button("Generate Intelligence", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🤖 Generating your content..."):
            generated_content = generate_content(topic, st.session_state.api_key, model_id, output_format, tone)
            if generated_content:
                st.markdown("---")
                st.subheader(f"Your Generated {output_format}")
                
                # Custom CSS for a cleaner, modern look
                st.markdown(
                    """
                    <style>
                    .content-container {
                        border: 1px solid #e1e4e8;
                        border-radius: 12px;
                        padding: 25px;
                        background-color: #f6f8fa;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                        color: #24292e;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )

                formatted_content = generated_content.replace('\n', '<br>')
                st.markdown(f'<div class="content-container">{formatted_content}</div>', unsafe_allow_html=True)
                
    else:
        st.warning("Please enter a topic to generate content.")
