# --- This is a Streamlit application that runs a local AI agent. ---
# --- It requires a powerful machine (GPU recommended) and libraries installed: ---
# pip install --upgrade streamlit transformers torch accelerate tavily-python

import streamlit as st
import os
from typing import Dict
from tavily import TavilyClient

# --- Core Transformers Import ---
# This library will run the AI logic locally.
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer


# --- 1. Model and Tool Loading (with Caching) ---
# Use Streamlit's caching to load the large model only once.
@st.cache_resource
def load_llm_pipeline(model_name="google/flan-t5-large"):
    """Loads the Hugging Face model and tokenizer into a pipeline."""
    st.info(f"Downloading and loading the local model '{model_name}'. This may take a few minutes on the first run...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    llm_pipeline_instance = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    st.success("✅ Local LLM Loaded Successfully!")
    return llm_pipeline_instance

# --- 2. Define the Agent's Logic in Plain Python ---
# We replace the LangGraph state and nodes with a single, clear function.

def run_agent_flow(topic: str, tavily_client: TavilyClient, llm_pipeline) -> str:
    """
    Orchestrates the agent's workflow using direct library calls instead of LangGraph.
    """
    # --- Node 1: Search ---
    st.write("--- 🔍 Searching for news ---")
    try:
        results = tavily_client.search(query=topic, max_results=3)
        search_results = "\n\n".join([res['content'] for res in results['results']])
    except Exception as e:
        st.error(f"Search failed: {e}. Is your Tavily API key correct?")
        return f"Error: Search failed. {e}"

    # --- Router Logic ---
    if not search_results or len(search_results) < 100:
        st.warning("--- ⚠️ Search results insufficient. Cannot generate a post. ---")
        return "Error: Could not find sufficient information from the web search."
    else:
        st.success("--- ✅ Search results are sufficient. Proceeding to synthesis. ---")

    # --- Node 2: Synthesis ---
    st.write("--- 🧠 Synthesizing information ---")
    synthesis_prompt = f"""
    Based on the following search results, synthesize the most critical and recent information into a few key bullet points.
    If the search results are empty or irrelevant, state that you could not find sufficient information.
    Search Results:
    {search_results}
    """
    # The pipeline returns a list of dictionaries, we extract the generated text.
    synthesis_response = llm_pipeline(synthesis_prompt)[0]['generated_text']

    # --- Node 3: Generate Post ---
    st.write("--- ✍️ Generating LinkedIn Post ---")
    generation_prompt = f"""
    Using the following key points, write a concise, powerful, and professional LinkedIn post.
    Conclude with 3 to 5 relevant and popular hashtags.
    Key Points:
    {synthesis_response}
    """
    final_response = llm_pipeline(generation_prompt)[0]['generated_text']

    return final_response


# --- 3. UI Layout and App Logic ---
st.set_page_config(page_title="InsightSphere Local - Manual Agent", page_icon="🧠", layout="wide")
st.title("🧠 InsightSphere (Local Agent)")
st.markdown("### Powered by Pure Python & Hugging Face")
st.warning("**Note:** This app runs a Large Language Model **locally on your computer**. It requires a powerful machine (preferably with a GPU) and may be slow.")

# --- Sidebar for API Key ---
st.sidebar.header("⚙️ Configuration")
tavily_api_key = st.sidebar.text_input(
    "Tavily API Key",
    type="password",
    help="Get your free key from tavily.com. This is used for web search."
)

# --- Main App ---
if not tavily_api_key:
    st.info("Please enter your Tavily API key in the sidebar to begin.")
else:
    # Load the LLM after the key is provided
    llm_pipeline_instance = load_llm_pipeline()
    # Initialize the Tavily client
    tavily_client = TavilyClient(api_key=tavily_api_key)

    topic = st.text_input("Enter a topic to generate a LinkedIn Post:", placeholder="e.g., 'latest trends in renewable energy'")

    if st.button("Generate Post", type="primary", use_container_width=True):
        if topic:
            with st.spinner("🤖 Agent is thinking..."):
                st.markdown("---")
                st.subheader("Agent Progress")

                try:
                    # Run the agent flow directly
                    final_output = run_agent_flow(topic, tavily_client, llm_pipeline_instance)

                    st.markdown("---")
                    st.subheader("✅ Final Output")

                    if final_output and not final_output.startswith("Error:"):
                        # Format the output for better readability
                        formatted_content = final_output.replace('\n', '<br>')
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #e1e4e8; border-radius: 12px; padding: 25px; background-color: #f6f8fa;">
                                {formatted_content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(final_output)

                except Exception as e:
                    st.error(f"An unexpected error occurred while running the agent: {e}")
                    st.error("Please ensure your Tavily API key is correct and that you have the necessary hardware (GPU recommended).")

        else:
            st.warning("Please enter a topic.")

