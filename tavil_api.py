# --- This is a Streamlit application that runs a local AI agent. ---
# --- It requires a powerful machine (GPU recommended) and libraries installed: ---
# pip install --upgrade streamlit langchain langchain-core langchain_huggingface transformers langgraph tavily-python torch accelerate

import streamlit as st
import os
from typing import List, Dict, TypedDict

# --- Core LangGraph and Transformers Imports ---
# These are the libraries that will run the AI logic locally.
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer
from langgraph.graph import StateGraph, END

# --- 1. Model and Tool Loading (with Caching) ---
# Use Streamlit's caching to load the large model only once.
@st.cache_resource
def load_llm_and_tokenizer(model_name="google/flan-t5-large"):
    """Loads the Hugging Face model and tokenizer."""
    st.info(f"Downloading and loading the local model '{model_name}'. This may take a few minutes on the first run...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    llm_pipeline_instance = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    llm = HuggingFacePipeline(pipeline=llm_pipeline_instance)
    st.success("✅ Local LLM Loaded Successfully!")
    return llm

# --- 2. Define the State for the Graph ---
# This is the "memory" of our agent as it moves through the steps.
class AgentState(TypedDict):
    messages: List[BaseMessage]
    topic: str
    search_results: str
    synthesis: str
    # We add these to pass tools into our nodes
    llm: HuggingFacePipeline
    search_tool: TavilySearchResults

# --- 3. Define the Nodes (Steps) of the Graph ---
# These functions will be the steps in our AI's thinking process.

def search_node(state: AgentState) -> Dict:
    """Node to search the web for the latest news on a topic."""
    search_tool = state['search_tool']
    topic = state['topic']
    st.write("--- 🔍 Searching for news ---")
    try:
        results = search_tool.invoke(topic)
        result_contents = "\n\n".join([res['content'] for res in results])
        return {"search_results": result_contents}
    except Exception as e:
        st.error(f"Search failed: {e}. Is your Tavily API key correct?")
        return {"search_results": ""}

def synthesis_node(state: AgentState) -> Dict:
    """Node to synthesize search results into key points."""
    llm = state['llm']
    st.write("--- 🧠 Synthesizing information ---")
    prompt = f"""
    Based on the following search results, synthesize the most critical and recent information into a few key bullet points.
    If the search results are empty or irrelevant, state that you could not find sufficient information.
    Search Results:
    {state['search_results']}
    """
    response = llm.invoke(prompt)
    return {"synthesis": response}

def generate_post_node(state: AgentState) -> Dict:
    """Node to generate the final LinkedIn post."""
    llm = state['llm']
    st.write("--- ✍️ Generating LinkedIn Post ---")
    prompt = f"""
    Using the following key points, write a concise, powerful, and professional LinkedIn post.
    Conclude with 3 to 5 relevant and popular hashtags.
    Key Points:
    {state['synthesis']}
    """
    response = llm.invoke(prompt)
    final_message = HumanMessage(content=response)
    return {"messages": state["messages"] + [final_message]}

# --- 4. Define the Router (Conditional Edge) ---
def should_continue(state: AgentState) -> str:
    """Router node to decide whether to continue based on search results."""
    if not state['search_results'] or len(state['search_results']) < 100:
        st.warning("--- ⚠️ Search results insufficient. Ending graph. ---")
        return "end"
    else:
        st.success("--- ✅ Search results are sufficient. Proceeding to synthesis. ---")
        return "continue"


# --- 5. UI Layout and App Logic ---
st.set_page_config(page_title="InsightSphere Local - LangGraph Agent", page_icon="🧠", layout="wide")
st.title("🧠 InsightSphere (Local Agent)")
st.markdown("### Powered by LangGraph & Hugging Face")
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
    llm_instance = load_llm_and_tokenizer()

    topic = st.text_input("Enter a topic to generate a LinkedIn Post:", placeholder="e.g., 'latest trends in renewable energy'")

    if st.button("Generate Post", type="primary", use_container_width=True):
        if topic:
            with st.spinner("🤖 Agent is thinking..."):
                # --- Build and Run the Graph ---
                search_tool_instance = TavilySearchResults(max_results=3, api_key=tavily_api_key)

                workflow = StateGraph(AgentState)
                workflow.add_node("search", search_node)
                workflow.add_node("synthesis", synthesis_node)
                workflow.add_node("generate_post", generate_post_node)
                workflow.set_entry_point("search")
                workflow.add_conditional_edges(
                    "search",
                    should_continue,
                    {"continue": "synthesis", "end": END}
                )
                workflow.add_edge("synthesis", "generate_post")
                workflow.add_edge("generate_post", END)
                app = workflow.compile()

                # Define the initial state to run the agent
                initial_state = {
                    "topic": topic,
                    "messages": [HumanMessage(content=f"Generate a LinkedIn post about {topic}")],
                    # Pass the tools and llm into the state so nodes can access them
                    "search_tool": search_tool_instance,
                    "llm": llm_instance
                }

                st.markdown("---")
                st.subheader("Agent Progress")
                final_state = None
                try:
                    # Run the agent and get the final state
                    final_state = app.invoke(initial_state)

                    st.markdown("---")
                    st.subheader("✅ Final Output")

                    if final_state and len(final_state['messages']) > 1:
                        # Format the output for better readability
                        formatted_content = final_state['messages'][-1].content.replace('\n', '<br>')
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #e1e4e8; border-radius: 12px; padding: 25px; background-color: #f6f8fa;">
                                {formatted_content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Could not generate a post. The web search might not have found enough information on the topic.")

                except Exception as e:
                    st.error(f"An error occurred while running the agent: {e}")
                    st.error("Please ensure your Tavily API key is correct and that you have the necessary hardware (GPU recommended).")

        else:
            st.warning("Please enter a topic.")
