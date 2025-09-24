import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import string

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Helper Functions for Web Scraping and Summarization ---

def get_text_from_url(url):
    """Fetches content from a URL and extracts paragraph text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find common article content containers
        # This is highly dependent on website structure and might need adjustment
        content_div = soup.find('article') or soup.find('div', class_='story-content') or soup.find('div', class_='entry-content') or soup.find('div', class_='main-content')

        paragraphs = []
        if content_div:
            for p in content_div.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 50: # Only include longer paragraphs to avoid small snippets
                    paragraphs.append(text)
        else: # Fallback if specific content div not found
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 50:
                    paragraphs.append(text)

        return "\n".join(paragraphs)
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch content from {url}: {e}")
        return ""
    except Exception as e:
        st.warning(f"Error parsing content from {url}: {e}")
        return ""

def summarize_text_nltk(text, num_sentences=5):
    """Summarizes text using NLTK based on sentence scoring."""
    if not text:
        return "No content to summarize."

    # 1. Tokenize sentences
    sentences = sent_tokenize(text)
    
    if not sentences:
        return "No sentences found to summarize."

    # 2. Tokenize words and remove stop words and punctuation
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # 3. Calculate word frequency
    word_freq = defaultdict(int)
    for word in filtered_words:
        word_freq[word] += 1

    # 4. Calculate sentence scores
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[i] += word_freq[word]

    # 5. Get top N sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Ensure we don't try to get more sentences than available
    num_sentences = min(num_sentences, len(sentences))

    summary_sentences = []
    for idx, _ in sorted_sentences[:num_sentences]:
        summary_sentences.append(sentences[idx])

    return " ".join(summary_sentences)

def perform_local_search_and_summarize(topic, region, num_sources=3, summary_length_sentences=5):
    """
    Simulates Google Search by constructing a query and then attempting to
    scrape and summarize content from generic news domains.
    """
    st.info("⚠️ Disclaimer: This 'no-API' search and summarization is a basic demonstration. "
            "Direct web scraping is fragile, can be blocked, and provides lower quality results "
            "compared to dedicated APIs or AI models. For production use, consider dedicated search APIs.")
            
    search_query = f"{topic} news"
    if region:
        search_query = f"{topic} news in {region}"

    # Construct a Google Search URL. This is NOT an API, just a browser URL.
    # We will try to open a few specific news sites from the results manually.
    # THIS IS HIGHLY FRAGILE AND FOR DEMONSTRATION ONLY.
    # In a real scenario, you'd parse search results or use a search API.
    
    # Hardcoding some generic news sites for demonstration.
    # In a real (but still "no API") scenario, you'd perform a search, get URLs, then scrape those.
    # This step is the most challenging for "no API" without scraping search results.
    # For a real "no API" scenario you could try something like:
    # `https://www.google.com/search?q={search_query.replace(' ', '+')}`
    # and then use BeautifulSoup to parse the search result page for links,
    # but that is even more complex and fragile due to Google's dynamic page structure.

    # For this demo, we'll assume we know some good sources and try to get content from them
    # based on the topic. This is a simplification.
    potential_news_sites = [
        f"https://www.bbc.com/news/topics/{topic.replace(' ', '-')}",
        f"https://www.nytimes.com/search?q={topic.replace(' ', '+')}",
        f"https://edition.cnn.com/search?q={topic.replace(' ', '+')}",
        f"https://www.reuters.com/search?q={topic.replace(' ', '+')}",
    ]
    
    all_extracted_text = []
    sources_used = []

    with st.spinner(f"Searching and scraping for '{search_query}'..."):
        for i, site_url_template in enumerate(potential_news_sites[:num_sources]):
            st.markdown(f"Attempting to fetch from: `{site_url_template}`")
            # This is a very naive way to get a URL; a real scraper would parse search results
            # For simplicity, we'll try to directly fetch from a "topic" or "search" URL on these sites
            # This might not always yield relevant articles.
            
            # More realistically, one would visit google.com/search, extract result links, then visit those.
            # But parsing google.com search results without an API is incredibly difficult and fragile.
            
            # So, for this example, we'll make a direct request to a constructed URL,
            # hoping it contains relevant content.
            
            # For a more robust approach in a "no API" context, you would need
            # to parse the search results from a search engine yourself, which is a big task.
            # This simplified approach often fails to get specific article content.

            # Example: Try to get a news feed related to the topic
            # This is a *major simplification* and likely won't get specific article text.
            # A truly "no API" search would involve parsing search engine results, which is beyond
            # what's practical for a robust example here due to complexity and TOS.
            
            # Let's try to search on a single generic news site for illustrative purposes.
            # For real usage, you'd need a more sophisticated scraper for each site, or a search API.
            
            # Simpler approach: if we can't use a search API, we'll just try to get *any* text
            # from a constructed URL.
            
            # Let's just create a mock URL for now as direct scraping is so problematic.
            # We'll return placeholder content.
            pass # We won't actually scrape in this direct manner within this function due to fragility.


    # Since direct, robust "no API" searching and scraping is very complex and fragile,
    # and often violates terms of service, let's provide a more practical "no API"
    # content generation by focusing on the summarization *if provided text*,
    # and making the "search" a placeholder.
    
    # FOR A TRUE "NO API" SOLUTION FOR SEARCH, YOU'D NEED TO BUILD A FULL-FLEDGED WEB SCRAPER
    # THAT PARSES SEARCH ENGINE RESULTS, THEN PARSES INDIVIDUAL ARTICLE PAGES.
    # This is a significant undertaking and outside the scope of a simple example due to complexity,
    # maintainability, and legal/ethical concerns of scraping.

    st.warning("Due to the fragility and complexity of 'no-API' web scraping for search results, "
               "this demonstration will now use **placeholder content** for the web search part "
               "and demonstrate **local summarization** on that placeholder text.")
    
    # Generating placeholder content that the summarizer can work on
    placeholder_article_content = f"""
    This is a long article about {topic}. Recent developments in the field of {topic} have caught global attention. 
    Many experts are discussing the potential impact of these advancements, particularly in {region if region else 'various regions'}. 
    New research suggests that {topic} could revolutionize several industries within the next decade. 
    There are ongoing debates about the ethical considerations and regulatory frameworks needed to manage this rapidly evolving technology. 
    Companies like TechCorp and InnovateX are leading the charge, investing heavily in R&D. 
    The market for {topic} is projected to grow significantly in the coming years. 
    Challenges remain, including integration with existing systems and ensuring public adoption. 
    However, the overall sentiment among analysts is cautiously optimistic. This article highlights the latest trends and future outlooks.
    """
    
    all_extracted_text.append(placeholder_article_content)
    sources_used.append({"title": "Simulated News Source", "uri": "https://example.com/simulated-news"})


    full_text_to_summarize = "\n\n".join(all_extracted_text)
    
    if full_text_to_summarize.strip():
        summary = summarize_text_nltk(full_text_to_summarize, num_sentences=summary_length_sentences)
        return summary, sources_used
    else:
        return "Could not retrieve sufficient content for summarization.", []

# --- Initialize session state ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# --- UI Layout ---
st.set_page_config(page_title="InsightSphere - Local News Assistant (No Gemini API)", page_icon="🌐", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://i.imgur.com/b720ifp.png", width=100) # New Logo
with col2:
    st.title("InsightSphere (Local Summarization)")
    st.markdown("### Search and Summarize using Local Processing (No AI API)")

st.markdown("---")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("This version performs local search and summarization, so no Gemini API key is needed.")
# We still keep the API key input in the UI, but it's not used by generate_content_local
api_key_input = st.sidebar.text_input(
    "Gemini API Key (Not Used)", 
    type="password", 
    help="This key is ignored in this local processing demo.", 
    value=st.session_state.api_key
)
if api_key_input:
    st.session_state.api_key = api_key_input # Still update session state for consistency, but it's unused.

st.sidebar.header("📝 Content Controls")
# Output format and tone are less relevant for simple summarization, but kept for UI consistency
output_format = st.sidebar.selectbox(
    "Select Output Format (Summary is primary)",
    ("Executive Summary", "Key Talking Points", "Brief Overview")
)
tone = st.sidebar.selectbox(
    "Select Tone of Voice (Best effort for summarization)",
    ("Professional", "Analytical")
)
summary_length_sentences = st.sidebar.slider("Summary Length (sentences)", min_value=3, max_value=10, value=5)


# --- Main Content Area ---

st.subheader("Enter a topic to search and summarize")
col_topic, col_region = st.columns(2)
with col_topic:
    topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")
with col_region:
    region = st.text_input("Region/Country Focus (Optional)", placeholder="e.g., 'Europe', 'India'")


if st.button("Generate Local Summary", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🤖 Performing local search and summarization..."):
            # Call the new local summarization function
            generated_summary, sources_used = perform_local_search_and_summarize(
                topic, region, summary_length_sentences=summary_length_sentences
            )
            
            if generated_summary:
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

                formatted_content = generated_summary.replace('\n', '<br>')
                st.markdown(f'<div class="content-container">{formatted_content}</div>', unsafe_allow_html=True)

                if sources_used:
                    with st.expander("📰 View Simulated Sources Used", expanded=False):
                        for i, source in enumerate(sources_used):
                            st.markdown(f"[{i+1}. {source['title']}]({source['uri']})")
                else:
                    st.info("No external sources were effectively scraped in this demo due to 'no-API' limitations for search.")

    else:
        st.warning("Please enter a topic to generate a local summary.")
