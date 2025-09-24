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

    with st.spinner(f"Sea
