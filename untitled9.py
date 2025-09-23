import streamlit as st
import requests
import json

def generate_content(topic, api_key, output_format, tone, region):
    """
    Calls the Gemini API with Google Search enabled to find news and generate customized content.
    """
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar to use this feature.")
        return None, None

    # Use the more powerful gemini-1.5-flash-latest model
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    # Dynamically build the prompt based on user selections for a more tailored output
    prompt = f"""
    Act as a world-class market analyst and communication expert.
    Your task is to analyze the absolute latest global news and developments about the topic: '{topic}'.

    If a specific region is mentioned, focus your search and analysis there.
    Region/Country Focus: {region if region else 'Global'}

    Based on your web search, perform the following tasks:

    1.  Synthesize the most critical and recent information.
    2.  Generate a response in the format of a '{output_format}'.
    3.  The tone of the response must be '{tone}'.
    4.  If the requested format is a 'LinkedIn Post', conclude with 3-5 relevant and popular hashtags.
    5.  Ensure the output is well-structured, insightful, and ready for a professional audience.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }

    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()

        # Safely extract the generated text and citations from the API response
        candidate = result.get('candidates', [{}])[0]
        content_part = candidate.get('content', {}).get('parts', [{}])[0]
        generated_text = content_part.get('text')

        citations = []
        grounding_metadata = candidate.get('groundingMetadata', {})
        if grounding_metadata and 'groundingAttributions' in grounding_metadata:
            for attribution in grounding_metadata['groundingAttributions']:
                title = attribution.get('web', {}).get('title', 'No Title')
                uri = attribution.get('web', {}).get('uri', '#')
                citations.append({"title": title, "uri": uri})

        if not generated_text:
            st.error("The model did not return a valid response. Please try a different topic or adjust your settings.")
            st.json(result) # Show raw response for debugging
            return None, None

        return generated_text, citations

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        try:
            error_details = response.json()
            st.error(f"API Error Details: {error_details.get('error', {}).get('message', 'No specific message.')}")
        except (ValueError, AttributeError):
            st.error(f"Response Body: {response.text if 'response' in locals() else 'No response'}")
        return None, None
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        st.error(f"Failed to parse AI response. The model's output may be malformed. Error: {e}")
        st.json(result)  # Show the raw response for debugging
        return None, None

# --- UI Layout ---
st.set_page_config(page_title="InsightSphere - AI Global News Assistant", page_icon="🌐", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://i.imgur.com/b720ifp.png", width=100) # New Logo
with col2:
    st.title("InsightSphere")
    st.markdown("### Turn Global News into Actionable Intelligence")

st.markdown("---")

# --- Sidebar for API Key and Controls ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your free key from Google AI Studio.")

st.sidebar.header("📝 Content Controls")
output_format = st.sidebar.selectbox(
    "Select Output Format",
    ("LinkedIn Post", "Executive Summary", "Email Briefing", "Key Talking Points")
)
tone = st.sidebar.selectbox(
    "Select Tone of Voice",
    ("Professional", "Analytical", "Casual", "Optimistic", "Cautious")
)

# --- Main Content Area ---
st.subheader("Enter a topic to begin your analysis")
col_topic, col_region = st.columns(2)
with col_topic:
    topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")
with col_region:
    region = st.text_input("Region/Country Focus (Optional)", placeholder="e.g., 'Europe', 'India'")


if st.button("Generate Intelligence", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🤖 Searching the web and generating your content..."):
            generated_content, citations = generate_content(topic, api_key, output_format, tone, region)
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
                        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                        margin-bottom: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                        color: #24292e;
                    }
                    .citations-expander {
                        border-radius: 12px !important;
                        border: 1px solid #e1e4e8 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )

                formatted_content = generated_content.replace('\n', '<br>')
                st.markdown(f'<div class="content-container">{formatted_content}</div>', unsafe_allow_html=True)

                if citations:
                    with st.expander("📰 View News Sources Used by the AI", expanded=False):
                        for i, source in enumerate(citations):
                            st.markdown(f"[{i+1}. {source['title']}]({source['uri']})")

    else:
        st.warning("Please enter a topic to generate content.")
