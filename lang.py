import streamlit as st
# Removed requests and json as they are no longer needed for a no-API version
# import requests
# import json

def generate_content_no_api(topic, output_format, tone, region):
    """
    Generates placeholder content based on the topic and format, without using any external APIs.
    This is for demonstration/development purposes when API access is not desired or available.
    """
    
    # Simulate some processing time
    import time
    time.sleep(2) 

    # Create static/placeholder content
    generated_text = f"""
    This is a simulated {output_format} about **{topic}**.
    
    **Tone:** {tone}
    **Region Focus:** {region if region else 'Global'}

    **Key Insight (Simulated):** Recent trends indicate a growing interest in {topic} across the globe, with particular activity noted in {region if region else 'various regions'}. Companies and researchers are investing heavily in this area, driven by both market demand and technological advancements.

    **Impact (Simulated):** The long-term implications for {topic} are substantial, potentially reshaping industries and daily life. Continued monitoring of developments in {region if region else 'key markets'} will be crucial.

    #Simulated #AI #PlaceholderContent #NoAPI
    """

    # Simulate some static citations
    citations = [
        {"title": "Simulated Source 1: The Future of " + topic, "uri": "https://example.com/source1"},
        {"title": "Simulated Source 2: Regional Analysis on " + topic, "uri": "https://example.com/source2"},
    ]
    
    # Add format-specific content for LinkedIn Post
    if output_format == "LinkedIn Post":
        generated_text += "\n\n#SimulatedPost #LinkedIn"
    
    return generated_text, citations

# --- Initialize session state ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# --- UI Layout ---
st.set_page_config(page_title="InsightSphere - AI Global News Assistant (No API Demo)", page_icon="🌐", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://i.imgur.com/b720ifp.png", width=100) # New Logo
with col2:
    st.title("InsightSphere (No API Demo)")
    st.markdown("### Illustrating the UI without external API calls")

st.markdown("---")

# --- Sidebar for Controls (API Key input is present but effectively ignored) ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("API Key input is ignored in this 'No API' demonstration.")
api_key_input = st.sidebar.text_input(
    "Gemini API Key (Ignored)", 
    type="password", 
    help="This key would normally be used for Gemini API.", 
    value=st.session_state.api_key
)
# Update session state, though it's not used in generate_content_no_api
if api_key_input:
    st.session_state.api_key = api_key_input

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

# No API key check needed as we're not using an API
st.subheader("Enter a topic to see simulated analysis")
col_topic, col_region = st.columns(2)
with col_topic:
    topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")
with col_region:
    region = st.text_input("Region/Country Focus (Optional)", placeholder="e.g., 'Europe', 'India'")


if st.button("Generate Intelligence (Simulated)", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🤖 Simulating content generation..."):
            # Call the new no-API function
            generated_content, citations = generate_content_no_api(topic, output_format, tone, region)
            
            if generated_content:
                st.markdown("---")
                st.subheader(f"Your Generated {output_format} (Simulated)")
                
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
                    with st.expander("📰 View Simulated News Sources", expanded=False):
                        for i, source in enumerate(citations):
                            st.markdown(f"[{i+1}. {source['title']}]({source['uri']})")

    else:
        st.warning("Please enter a topic to simulate content.")
