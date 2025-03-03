from typing import Literal
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# State
class State(TypedDict):
    user_input: str  # User's input
    mood: str        # Classified mood (e.g., anxiety, depression)
    response: str    # Bot's response

# Nodes
def reception_agent(state):
    """Ask the user how they are feeling today."""
    user_input = input("Hi! How are you feeling today? ")
    return {"user_input": user_input, "mood": "", "response": f"User says: {user_input}"}

def mood_assessment_agent(state):
    """Assess the user's mood based on input."""
    user_input = state["user_input"].lower()

    if "anxious" in user_input or "stressed" in user_input:
        return {"mood": "anxiety", "response": "It sounds like you're feeling anxious."}
    elif "sad" in user_input or "depressed" in user_input:
        return {"mood": "depression", "response": "It sounds like you're feeling down."}
    elif "fine" in user_input or "okay" in user_input:
        return {"mood": "neutral", "response": "I'm glad to hear you're feeling fine."}
    elif "good" in user_input or "okay" in user_input:
        return {"mood": "cheerful", "response": "I'm glad to hear you're feeling so good!"}
    else:
        return {"mood": "unknown", "response": "Could you tell me more about how you're feeling?"}

def recommendation_agent(state):
    """Recommend an activity or task based on mood."""
    if state["mood"] == "anxiety":
        return {"response": "How about some deep breathing exercises or a walk outside?"}
    elif state["mood"] == "depression":
        return {"response": "Maybe try journaling or a relaxing activity like reading?"}
    elif state["mood"] == "cheerful":
        return {"response": "I am glad to hear that. Why are you feeling cheerful?"}
    elif state["mood"] == "neutral":
        return {"response": "It's good that you're doing fine! Keep up with healthy habits!"}
    else:
        return {"response": "It's important to talk to someone about how you're feeling. Would you like to speak to a counselor?"}

def counselor_recommendation(state):
    """Recommend a counselor if the mood is serious."""
    return {"response": "It might be helpful to talk to a professional. I recommend contacting a counselor."}

# Build graph
builder = StateGraph(State)
builder.add_node("reception_agent", reception_agent)
builder.add_node("mood_assessment_agent", mood_assessment_agent)
builder.add_node("recommendation_agent", recommendation_agent)
builder.add_node("counselor_recommendation", counselor_recommendation)

# Define edges
builder.add_edge(START, "reception_agent")
builder.add_conditional_edges("reception_agent", lambda state: "mood_assessment_agent")
builder.add_edge("mood_assessment_agent", "recommendation_agent")
builder.add_conditional_edges("recommendation_agent", lambda state: "counselor_recommendation" if state["mood"] == "unknown" else END)
builder.add_edge("counselor_recommendation", END)

# Compile graph
graph = builder.compile()

# Running the graph with custom input
def run_graph():
    initial_state = {"user_input": "", "mood": "", "response": ""}
    while True:
        result = graph.invoke(initial_state)
        print(result["response"])

        # Break the loop if the conversation ends
        if "Would you like to speak to a counselor?" in result["response"]:
            break

# Start the conversation
run_graph()
