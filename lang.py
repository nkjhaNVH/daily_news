# --- This is a conceptual script and requires a local environment with ---
# --- a powerful GPU and necessary libraries installed: ---
# pip install --upgrade langchain langchain-core langchain_huggingface transformers langgraph tavily-python beautifulsoup4 python-dotenv

import os
from typing import List, Dict, TypedDict

# --- Setup: Load environment variables for API keys ---
# Create a .env file in the same directory and add your key:
# TAVILY_API_KEY="your_tavily_api_key_here"
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer
from langgraph.graph import StateGraph, END

# --- 1. Define the State for the Graph ---
# This is the "memory" of our agent as it moves through the steps.
class AgentState(TypedDict):
    messages: List[BaseMessage]
    topic: str
    search_results: str
    synthesis: str

# --- 2. Set Up Tools and the LLM ---
# Check if the Tavily API key is available
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Tavily API key not found. Please add it to your .env file.")

# This is a search tool LangGraph can use.
search_tool = TavilySearchResults(max_results=3)

# Load a local model from Hugging Face.
# This downloads a multi-gigabyte model and requires a GPU to run effectively.
# Using a smaller, faster model for demonstration purposes.
# For higher quality, you might use 'meta-llama/Llama-3-8B-Instruct'
print("--- 🚀 Loading local LLM (google/flan-t5-large). This may take a moment... ---")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("--- ✅ LLM Loaded Successfully ---")


# --- 3. Define the Nodes (Steps) of the Graph ---

def search_node(state: AgentState) -> Dict:
    """Node to search the web for the latest news on a topic."""
    print("--- 🔍 Searching for news ---")
    topic = state['topic']
    try:
        results = search_tool.invoke(topic)
        result_contents = "\n\n".join([res['content'] for res in results])
        return {"search_results": result_contents}
    except Exception as e:
        print(f"--- ❌ Search failed: {e} ---")
        return {"search_results": ""}


def synthesis_node(state: AgentState) -> Dict:
    """Node to synthesize search results into key points."""
    print("--- 🧠 Synthesizing information ---")
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
    print("--- ✍️ Generating LinkedIn Post ---")
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
        print("--- ⚠️ Search results insufficient. Ending graph. ---")
        return "end"
    else:
        print("--- ✅ Search results are sufficient. Proceeding to synthesis. ---")
        return "continue"


# --- 5. Build the Graph ---
# This defines the workflow with a self-correction check.
workflow = StateGraph(AgentState)

workflow.add_node("search", search_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("generate_post", generate_post_node)

workflow.set_entry_point("search")

# Add the conditional router. After the "search" node, it will call "should_continue".
# Based on the return value ("continue" or "end"), it will route to the next appropriate node.
workflow.add_conditional_edges(
    "search",
    should_continue,
    {
        "continue": "synthesis",
        "end": END,
    }
)
workflow.add_edge("synthesis", "generate_post")
workflow.add_edge("generate_post", END)

# Compile the graph into a runnable object
app = workflow.compile()


# --- 6. Run the Agent ---
def run_agent(topic: str):
    """Function to run the agent with a specific topic."""
    initial_state = {
        "topic": topic,
        "messages": [HumanMessage(content=f"Generate a LinkedIn post about {topic}")]
    }
    
    print(f"\n--- Running Agent for topic: '{topic}' ---")
    # The 'stream' method shows the output of each step
    for s in app.stream(initial_state):
        print(s)
        print("----")
    
    final_state = app.invoke(initial_state)
    print("\n--- ✅ Final Output ---")
    # Check if a final message was generated before trying to print it
    if len(final_state['messages']) > 1:
        print(final_state['messages'][-1].content)
    else:
        print("Could not generate a post due to insufficient information from the search.")


# Example usage:
if __name__ == "__main__":
    try:
        run_agent("latest trends in renewable energy")
        print("\n" + "="*50 + "\n")
        run_agent("news about fictional company 'GloboCorp'") # Example that might fail search
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

