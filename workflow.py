import random
from typing import Literal
from IPython.display import Image, display
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# State
class State(TypedDict):
    user_input: str  # Stores the user's input
    category: str    # Stores the category (e.g., anxiety, depression, crisis)
    response: str    # Stores the bot's response

# Conditional edge
def decide_category(state) -> Literal["anxiety_support", "depression_support", "crisis_intervention","neutral_state"]:
    user_input = state['user_input'].lower()

    if "anxious" in user_input or "stressed" in user_input:
        return "anxiety_support"
    elif "sad" in user_input or "depressed" in user_input:
        return "depression_support"
    elif "hurt myself" in user_input or "suicide" in user_input:
        return "crisis_intervention"
    elif "happy" in user_input or "good" in user_input:
        return "neutral_state"
    else:
        return "general_support"

# Nodes
def reception_agent(state):
    print("---Reception Agent---")
    return {"user_input": state["user_input"], "category": "", "response": "Hello! How are you feeling today?"}

def anxiety_support_agent(state):
    print("---Anxiety Support Agent---")
    return {"user_input": state["user_input"], "category": "anxiety", "response": "It sounds like you’re feeling anxious. Have you tried deep breathing exercises?"}

def depression_support_agent(state):
    print("---Depression Support Agent---")
    return {"user_input": state["user_input"], "category": "depression", "response": "I’m sorry you’re feeling this way. Remember, it’s okay to ask for help."}

def crisis_intervention_agent(state):
    print("---Crisis Intervention Agent---")
    return {"user_input": state["user_input"], "category": "crisis", "response": "This sounds serious. Please seek support immediately."}

def neutral_state_agent(state):
    print("---Neutral State Agent---")
    return {"user_input": state["user_input"], "category": "neutral", "response": "That's great to hear! Why is that?"}

def general_support_agent(state):
    print("---General Support Agent---")
    return {"user_input": state["user_input"], "category": "general", "response": "I’m here to listen. Can you tell me more about how you’re feeling?"}

# Build graph
builder = StateGraph(State)
builder.add_node("reception_agent", reception_agent)
builder.add_node("anxiety_support_agent", anxiety_support_agent)
builder.add_node("depression_support_agent", depression_support_agent)
builder.add_node("crisis_intervention_agent", crisis_intervention_agent)
builder.add_node("neutral_state_agent", neutral_state_agent)
builder.add_node("general_support_agent", general_support_agent)

# Define edges
builder.add_edge(START, "reception_agent")
builder.add_conditional_edges("reception_agent", decide_category, {
    "anxiety_support": "anxiety_support_agent",
    "depression_support": "depression_support_agent",
    "crisis_intervention": "crisis_intervention_agent","neutral_state": "neutral_state_agent",
    "general_support": "general_support_agent"
})
builder.add_edge("anxiety_support_agent", END)
builder.add_edge("depression_support_agent", END)
builder.add_edge("crisis_intervention_agent", END)
builder.add_edge("neutral_state_agent", END)
builder.add_edge("general_support_agent", END)

# Compile graph
graph = builder.compile()

# Visualize the graph
display(Image(graph.get_graph().draw_mermaid_png()))

# Run the graph
initial_state = {"user_input": "I want to hurt myself, because I keep messing things up.", "category": "", "response": ""}
result = graph.invoke(initial_state)
print(result)