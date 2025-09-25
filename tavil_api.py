import streamlit as st
import requests
import json

def generate_content(topic, api_key, output_format, tone, region):
    """
    Calls the Tavily API to find news and uses the results to generate content via the Gemini API.
    """
    if not api_key:
        st.error("Please enter your Tavily API key in the sidebar to use this feature.")
        return None, None

    # Step 1: Call the Tavily API to perform a web search
    tavily_api_url = "https://api.tavily.com/search"
    query = f"{topic} news in {region}" if region else f"{topic} news"
    
    tavily_payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_raw_content": True,
        "include_images": False
    }

    try:
        tavily_response = requests.post(tavily_api_url, json=tavily_payload)
        tavily_response.raise_for_status()
        tavily_result = tavily_response.json()
        
        search_results = tavily_result.get('results', [])
        
        if not search_results:
            st.warning("No search results found for the given topic. Please try a different topic.")
            return None, None
            
        # Extract content and citations from Tavily results
        context = ""
        citations = []
        for result in search_results:
            context += f"Title: {result['title']}\nContent: {result['content']}\n\n"
            citations.append({"title": result['title'], "uri": result['url']})

    except requests.exceptions.HTTPError as e:
        st.error(f"Tavily API request failed with status code: {e.response.status_code}")
        st.warning("Please check your API key and ensure it is valid.")
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Network request failed: {e}")
        st.warning("Please check your internet connection.")
        return None, None

    # Step 2: Call the Gemini API to generate content based on the search results
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={st.secrets['GEMINI_API_KEY']}"

    prompt = f"""
    Act as a world-class market analyst and communication expert.
    Analyze the following information and developments:

    ---
    {context}
    ---

    Your task is to:
    1. Synthesize the most critical and recent information from the provided context.
    2. Generate a response in the format of a '{output_format}'.
    3. The tone of the response must be '{tone}'.
    4. If the requested format is a 'LinkedIn Post', conclude with 3-5 relevant and popular hashtags.
    5. Ensure the output is well-structured, insightful, and ready for a professional audience.
    """

    gemini_payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    try:
        gemini_response = requests.post(gemini_api_url, json=gemini_payload, headers={'Content-Type': 'application/json'})
        gemini_response.raise_for_status()
        result = gemini_response.json()

        generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if not generated_text:
            st.error("The Gemini model did not return a valid response. Please try a different topic or adjust your settings.")
            return None, None
        
        return generated_text, citations

    except requests.exceptions.HTTPError as e:
        st.error(f"Gemini API request failed with status code: {e.response.status_code}")
        st.warning("Please check your Gemini API key and try again.")
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Network request failed: {e}")
        st.warning("Please check your internet connection.")
        return None, None


# --- Initialize session state ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# --- UI Layout ---
st.set_page_config(page_title="InsightSphere - News Assistant", page_icon="🌐", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("🌐")
with col2:
    st.title("InsightSphere")
    st.markdown("### Turn Global News into Actionable Intelligence")

st.markdown("---")

# --- Sidebar for API Key and Controls ---
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "Tavily API Key",
    type="password",
    help="Get your free key from Tavily AI.",
    value=st.session_state.api_key
)
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

if not st.session_state.api_key:
    st.warning("👋 Welcome to InsightSphere! Please enter your Tavily API key in the sidebar to begin.")
    st.stop()

st.subheader("Enter a topic to begin your analysis")
col_topic, col_region = st.columns(2)
with col_topic:
    topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")
with col_region:
    region = st.text_input("Region/Country Focus (Optional)", placeholder="e.g., 'Europe', 'India'")

if st.button("Generate Intelligence", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🌐 Searching the web and generating your content..."):
            generated_content, citations = generate_content(topic, st.session_state.api_key, output_format, tone, region)
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
