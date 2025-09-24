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
    progress_bar = st.progress(0, text="🔍 Searching for news...")
    try:
        results = tavily_client.search(query=topic, max_results=3)
        search_results = "\n\n".join([res['content'] for res in results['results']])
    except Exception as e:
        st.error(f"Search failed: {e}. Is your Tavily API key correct?")
        progress_bar.empty()
        return f"Error: Search failed. {e}"

    # --- Router Logic ---
    if not search_results or len(search_results) < 100:
        st.warning("--- ⚠️ Search results insufficient. Cannot generate a post. ---")
        progress_bar.empty()
        return "Error: Could not find sufficient information from the web search."
    else:
        st.success("--- ✅ Search results are sufficient. ---")

    # --- Node 2: Synthesis ---
    progress_bar.progress(33, text="🧠 Synthesizing information...")
    synthesis_prompt = f"""
    Based on the following search results, synthesize the most critical and recent information into a few key bullet points.
    If the search results are empty or irrelevant, state that you could not find sufficient information.
    Search Results:
    {search_results}
    """
    try:
        # The pipeline returns a list of dictionaries, we extract the generated text.
        synthesis_response = llm_pipeline(synthesis_prompt, max_length=256)[0]['generated_text']
        if not synthesis_response.strip():
             st.error("Synthesis resulted in empty output. The model may be struggling with the content.")
             progress_bar.empty()
             return "Error: Synthesis failed to produce content."
    except Exception as e:
        st.error(f"Error during synthesis: {e}")
        progress_bar.empty()
        return f"Error: The local AI model failed during the synthesis step. This could be a memory issue."


    # --- Node 3: Generate Post ---
    progress_bar.progress(66, text="✍️ Generating LinkedIn Post...")
    generation_prompt = f"""
    Using the following key points, write a concise, powerful, and professional LinkedIn post.
    The post should be engaging and insightful.
    Conclude with 3 to 5 relevant and popular hashtags.
    Key Points:
    {synthesis_response}
    """
    try:
        # Added min_length and max_length to ensure the model generates a response.
        final_response = llm_pipeline(generation_prompt, min_length=50, max_length=300)[0]['generated_text']
        if not final_response.strip():
             st.error("Final post generation resulted in empty output. The model failed to generate the post.")
             progress_bar.empty()
             return "Error: The model failed to generate the final post."
    except Exception as e:
        st.error(f"Error during post generation: {e}")
        progress_bar.empty()
        return f"Error: The local AI model failed during the final generation step."

    progress_bar.progress(100, text="✅ Complete!")
    progress_bar.empty()
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
            # We remove the spinner to see the detailed progress updates below
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

