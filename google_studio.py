import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import operator
from typing import TypedDict, Annotated, List, Union

# --- 1. Load Hugging Face Model ---
@st.cache_resource
def load_hf_model():
    st.write("Loading Hugging Face model (this might take a moment)...")
    try:
        model_name = "google/flan-t5-small" # A small, instruction-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Determine if GPU is available
        device = 0 if torch.cuda.is_available() else -1
        
        # Create a Hugging Face pipeline
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            device=device # Use GPU if available, otherwise CPU
        )
        st.success(f"Hugging Face model '{model_name}' loaded successfully on {'GPU' if device == 0 else 'CPU'}!")
        return hf_pipeline
    except Exception as e:
        st.error(f"Error loading Hugging Face model: {e}")
        st.info("Please ensure you have `transformers` and `torch` installed. "
                "If using GPU, ensure CUDA is set up correctly.")
        return None

hf_pipeline = load_hf_model()

if not hf_pipeline:
    st.stop() # Stop Streamlit app if model loading failed

# Convert Hugging Face pipeline to a LangChain Runnable
llm = hf_pipeline

# --- 2. Define Tools ---

def simulated_search(query: str) -> str:
    """
    Simulates a search for a given query.
    In a real application, this would call a search API or a web scraper.
    """
    st.info(f"Using simulated search for: '{query}'")
    
    # Simple keyword-based simulated responses
    query_lower = query.lower()
    
    if "weather" in query_lower:
        return "The weather in London is currently 15°C and partly cloudy. Expect rain tomorrow."
    elif "capital of france" in query_lower or "paris" in query_lower:
        return "The capital of France is Paris. It's famous for the Eiffel Tower."
    elif "current events" in query_lower:
        return "Recent news includes discussions on global economic trends and climate policy updates."
    elif "ai" in query_lower or "artificial intelligence" in query_lower:
        return "Artificial intelligence is a rapidly advancing field focusing on creating intelligent machines. Recent breakthroughs include large language models and advanced robotics."
    else:
        return f"Simulated search results for '{query}': No specific information found, but general knowledge suggests related topics like technology and global affairs."

# --- 3. Define LangGraph State ---

class AgentState(TypedDict):
    """
    Represents the state of our agent's graph.
    - `input`: The original user query.
    - `chat_history`: List of messages in the conversation.
    - `intermediate_steps`: A list of tuples, where each tuple is (tool_name, tool_output).
    - `final_answer`: The agent's final answer.
    """
    input: str
    chat_history: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    intermediate_steps: Annotated[List[tuple], operator.add]
    final_answer: str

# --- 4. Define Graph Nodes ---

def call_llm(state: AgentState):
    """
    Node to invoke the LLM for a decision or generating an answer.
    """
    messages = [
        HumanMessage(content=state["input"])
    ]
    
    # Include chat history for context
    if state["chat_history"]:
        messages = state["chat_history"] + messages

    # Include intermediate steps if any (tool outputs)
    if state["intermediate_steps"]:
        tool_outputs = "\n".join([f"Tool Output ({name}): {output}" for name, output in state["intermediate_steps"]])
        messages.append(AIMessage(content=f"Previous tool outputs:\n{tool_outputs}"))
    
    # This prompt tells the LLM how to act
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a helpful assistant. Your goal is to answer the user's question.
         You have access to a `simulated_search` tool.
         
         If you need to use the tool, respond ONLY with:
         <tool_use>simulated_search(query)</tool_use>
         Replace 'query' with the actual search term.
         
         If you have enough information, respond with your final answer.
         """
        ),
        *messages
    ])
    
    full_prompt = prompt_template.format(input=state["input"]) # LangGraph handles message formatting for us if we pass messages directly
    
    st.write(f"LLM Input (simplified for display): {state['input']}")
    if state["intermediate_steps"]:
        st.write(f"   (with previous tool outputs)")

    response = llm(full_prompt) # Use the HuggingFace pipeline directly
    
    # The output from the HF pipeline is a list of dicts, get the generated_text
    generated_text = response[0]['generated_text'] if isinstance(response, list) and response else str(response)

    st.write(f"LLM Output: {generated_text}")
    return {"final_answer": generated_text, "chat_history": [HumanMessage(content=state["input"]), AIMessage(content=generated_text)]}


def call_tool(state: AgentState):
    """
    Node to invoke the simulated search tool.
    """
    last_llm_output = state["final_answer"] # Get the last output from LLM, which should be a tool_use tag
    
    # Extract the query from the <tool_use> tag
    if "<tool_use>simulated_search(" in last_llm_output and "</tool_use>" in last_llm_output:
        start_idx = last_llm_output.find("simulated_search(") + len("simulated_search(")
        end_idx = last_llm_output.find(")</tool_use>")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            query = last_llm_output[start_idx:end_idx].strip().strip("'\"")
            st.info(f"Agent decided to use tool with query: '{query}'")
            tool_output = simulated_search(query)
            return {"intermediate_steps": [("simulated_search", tool_output)]}
        else:
            st.warning("Could not parse tool use from LLM output. Returning error.")
            return {"intermediate_steps": [("error", "Failed to parse tool use command.")]}
    else:
        st.warning("LLM output did not contain a tool use command when expected.")
        return {"intermediate_steps": [("error", "LLM did not output tool use command.")]}

# --- 5. Define Graph Edges (Conditional Logic) ---

def should_continue(state: AgentState):
    """
    Determines whether the agent should continue by calling a tool,
    or if it has a final answer.
    """
    last_llm_output = state["final_answer"]
    if "<tool_use>" in last_llm_output and "</tool_use>" in last_llm_output:
        return "call_tool"
    else:
        return "end"

# --- 6. Build the LangGraph ---

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("call_tool", call_tool)

    # Set entry point
    workflow.set_entry_point("call_llm")

    # Define edges
    # If LLM decides to use a tool, call the tool
    # Otherwise, if LLM generates a final answer, end
    workflow.add_conditional_edges(
        "call_llm",    # From node
        should_continue, # Conditional function
        {
            "call_tool": "call_tool", # If should_continue returns "call_tool", go to call_tool node
            "end": END               # If should_continue returns "end", stop the graph
        }
    )
    
    # After calling a tool, always go back to the LLM to process the tool output
    workflow.add_edge("call_tool", "call_llm")

    # Compile the graph
    app = workflow.compile()
    return app

agent_executor = build_graph()

# --- Streamlit UI ---

st.set_page_config(page_title="Hugging Face & LangGraph Agent", page_icon="🤖", layout="wide")

st.title("🤖 Simple Agent with Hugging Face & LangGraph (Local)")
st.markdown("This agent uses a local Hugging Face model (`google/flan-t5-small`) and a simulated search tool.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []

# Display chat history
st.subheader("Conversation")
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Agent:** {msg.content}")

user_input = st.text_input("Ask the agent a question:", key="user_input")

if st.button("Ask Agent", type="primary"):
    if user_input:
        with st.spinner("Agent thinking..."):
            initial_state = {
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "intermediate_steps": st.session_state.intermediate_steps,
                "final_answer": "" # Will be populated by LLM
            }
            
            # Run the agent graph
            try:
                # Iterate through the graph steps to see intermediate states
                final_output = None
                for s in agent_executor.stream(initial_state):
                    final_output = s # Keep track of the last state
                    st.write("--- Graph Step ---")
                    st.write(s) # Display intermediate steps if you want to debug
                
                # Get the final state of the graph
                # The .invoke() method would give you the final state directly.
                # Since we streamed, we'll use the last state from the loop.
                if final_output:
                    agent_response = final_output[END]["final_answer"] # Access the final answer from the END state
                    
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=agent_response))
                    st.session_state.intermediate_steps = [] # Reset intermediate steps for next turn

                    st.markdown("---")
                    st.subheader("Agent's Final Response")
                    st.write(agent_response)
                else:
                    st.error("Agent did not return a final output.")
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                st.error("This often happens if the LLM output is not in the expected format (e.g., missing tool_use tags).")
        
        # Clear input box after submission
        st.session_state.user_input = ""
    else:
        st.warning("Please enter a question.")
