import random
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from transformers import LlamaForCausalLM, LlamaTokenizer

# State definition for tracking user input, mood, and response
class State(TypedDict):
    user_input: str
    mood: str
    response: str

# Initialize the Llama model and tokenizer (replace with your Llama model path)
model_path = r"C:/Users/gurjy/.lmstudio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf"
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Function to decide category based on user input
def decide_mood(state) -> str:
    user_input = state['user_input'].lower()

    if "anxious" in user_input or "stressed" in user_input:
        return "anxiety_support"
    elif "sad" in user_input or "depressed" in user_input:
        return "depression_support"
    elif "happy" in user_input or "cheerful" in user_input:
        return "cheerful_support"
    else:
        return "general_support"

# Function to generate a response using Llama model
def generate_response(user_input: str) -> str:
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Nodes and their functions (reception and mood-specific responses)
def reception_agent(state: State):
    print("---Reception Agent---")
    return {"user_input": state["user_input"], "mood": "", "response": "How are you feeling today?"}

def anxiety_support_agent(state: State):
    print("---Anxiety Support Agent---")
    return {"user_input": state["user_input"], "mood": "anxious", "response": "It sounds like you're feeling anxious. Try some deep breathing exercises."}

def depression_support_agent(state: State):
    print("---Depression Support Agent---")
    return {"user_input": state["user_input"], "mood": "depressed", "response": "I'm really sorry you're feeling this way. Would you like to talk more about it?"}

def cheerful_support_agent(state: State):
    print("---Cheerful Support Agent---")
    return {"user_input": state["user_input"], "mood": "cheerful", "response": "That's great to hear! What has made you feel so cheerful today?"}

def general_support_agent(state: State):
    print("---General Support Agent---")
    return {"user_input": state["user_input"], "mood": "neutral", "response": "I'm here to listen. Can you tell me more about how you're feeling?"}

# Build graph using LangGraph
builder = StateGraph(State)
builder.add_node("reception_agent", reception_agent)
builder.add_node("anxiety_support_agent", anxiety_support_agent)
builder.add_node("depression_support_agent", depression_support_agent)
builder.add_node("cheerful_support_agent", cheerful_support_agent)
builder.add_node("general_support_agent", general_support_agent)

# Define conditional edges
builder.add_edge(START, "reception_agent")
builder.add_conditional_edges("reception_agent", decide_mood, {
    "anxiety_support": "anxiety_support_agent",
    "depression_support": "depression_support_agent",
    "cheerful_support": "cheerful_support_agent",
    "general_support": "general_support_agent"
})

builder.add_edge("anxiety_support_agent", END)
builder.add_edge("depression_support_agent", END)
builder.add_edge("cheerful_support_agent", END)
builder.add_edge("general_support_agent", END)

# Compile the graph
graph = builder.compile()

# Visualize the graph (optional)
# display(Image(graph.get_graph().draw_mermaid_png()))

# Example state and interaction
initial_state = {"user_input": "I am feeling anxious", "mood": "", "response": ""}
result = graph.invoke(initial_state)
print(result)

# Optionally: Using Llama for generating intelligent responses
user_input = "I'm feeling good today!"
generated_response = generate_response(user_input)
print("Llama's Response:", generated_response)
