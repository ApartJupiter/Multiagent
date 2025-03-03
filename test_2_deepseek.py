from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pydantic import BaseModel

# Initialize the OpenAI client.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Define the state schema using Pydantic
class ChatState(BaseModel):
    messages: list = []
    exit: bool = False
    mood: str = ""

# --- Define the Agent Functions (Graph Nodes) ---

def reception_agent_node(state: ChatState) -> ChatState:
    """
    Reception Agent:
      - Greets the user.
      - Asks for a brief introduction about themselves and their day.
    """
    print("\n[Reception Agent]: Hello, welcome to our mental health chat!")
    print("[Reception Agent]: Can you tell me a bit about yourself and how your day is going?")
    
    user_input = input("You: ")
    if user_input.lower() == "exit":
        state.exit = True
        return state
    
    state.messages.append({"role": "user", "content": user_input})
    return state

def analysis_agent_node(state: ChatState) -> ChatState:
    """
    Analysis Agent:
      - Asks the user how they are feeling.
      - If the user expresses negative emotions, asks for further details.
    """
    print("\n[Analysis Agent]: How are you feeling right now? (e.g., anxious, depressed, suicidal, happy, etc.)")
    feeling = input("You: ")
    if feeling.lower() == "exit":
        state.exit = True
        return state
    state.messages.append({"role": "user", "content": feeling})
    
    if feeling.lower() in ["anxious", "depressed", "suicidal"]:
        print("[Analysis Agent]: Can you share more about why you're feeling this way?")
        reason = input("You: ")
        if reason.lower() == "exit":
            state.exit = True
            return state
        state.messages.append({"role": "user", "content": reason})
    
    return state

def assignment_agent_node(state: ChatState) -> ChatState:
    """
    Assignment Agent:
      - Reviews the conversation history to determine the user's emotional state.
      - Asks tailored follow-up questions based on that state.
      - Uses the conversation history to prompt the AI for a conclusion and recommendations.
    """
    print("\n[Assignment Agent]:")
    messages = state.messages
    user_feeling = None

    # Try to determine the user's emotional state from prior inputs.
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"].lower()
            if any(feeling in content for feeling in ["anxious", "depressed", "suicidal", "happy"]):
                if "anxious" in content:
                    user_feeling = "anxious"
                elif "depressed" in content:
                    user_feeling = "depressed"
                elif "suicidal" in content:
                    user_feeling = "suicidal"
                elif "happy" in content:
                    user_feeling = "happy"
                break

    # Store the detected mood in the state.
    if user_feeling is None:
        print("[Assignment Agent]: I'm here to help. Could you clarify how you're feeling right now?")
        user_feeling = input("You: ").lower()
        if user_feeling == "exit":
            state.exit = True
            return state
        messages.append({"role": "user", "content": "Clarified feeling: " + user_feeling})
    state.mood = user_feeling

    # Set tailored follow-up questions.
    if user_feeling == "anxious":
        q1 = "What are the main things causing you stress or anxiety?"
        q2 = "Have you tried any methods to calm yourself? If so, what works for you?"
    elif user_feeling == "depressed":
        q1 = "Are there any particular events or thoughts that are making you feel this way?"
        q2 = "Is there anything that usually makes you feel a little better when you're feeling low?"
    elif user_feeling == "suicidal":
        q1 = "I'm really sorry you're feeling this way. Do you have someone you trust to talk to right now?"
        q2 = "Would you consider reaching out to a mental health professional for support?"
    elif user_feeling == "happy":
        q1 = "That's great to hear! What has been making you feel happy today?"
        q2 = "What can you do to continue nurturing this positive feeling?"
    else:
        q1 = "Could you describe your feelings in more detail?"
        q2 = "Have you noticed any patterns in your emotions recently?"
    
    # Ask the tailored follow-up questions.
    print("[Assignment Agent]:", q1)
    answer1 = input("You: ")
    if answer1.lower() == "exit":
        state.exit = True
        return state
    messages.append({"role": "user", "content": q1 + " " + answer1})
    
    print("[Assignment Agent]:", q2)
    answer2 = input("You: ")
    if answer2.lower() == "exit":
        state.exit = True
        return state
    messages.append({"role": "user", "content": q2 + " " + answer2})
    
    # Append a prompt to generate a conclusion.
    conclusion_prompt = ("Based on our conversation so far, please provide a thoughtful conclusion along with "
                         "tailored recommendations and actionable self-care or professional advice.")
    messages.append({"role": "user", "content": conclusion_prompt})
    
    response = client.chat.completions.create(
        model="llama-3.2-1b-instruct",
        messages=messages
    )
    conclusion = response.choices[0].message.content
    print("\n[Assignment Agent Conclusion]:", conclusion)
    messages.append({"role": "assistant", "content": conclusion})
    
    return state

def support_agent_node(state: ChatState) -> ChatState:
    """
    Support Agent:
      - Asks if the user needs any additional support.
      - If yes, collects further details and prompts the AI for extra recommendations.
    """
    print("\n[Support Agent]:")
    support_query = "Do you feel that you need any additional support or guidance at this moment? (yes/no)"
    print("[Support Agent]:", support_query)
    answer = input("You: ")
    if answer.lower() == "exit":
        state.exit = True
        return state
    state.messages.append({"role": "user", "content": support_query + " " + answer})
    
    if answer.strip().lower() in ["yes", "y"]:
        followup_support = "Could you please elaborate on what kind of support you need or what you're currently struggling with?"
        print("[Support Agent]:", followup_support)
        support_details = input("You: ")
        if support_details.lower() == "exit":
            state.exit = True
            return state
        state.messages.append({"role": "user", "content": followup_support + " " + support_details})
        
        additional_support_prompt = ("Based on our entire conversation, including your recent input about needing additional support, "
                                     "please provide further guidance, self-care tips, or suggestions to help address the issues mentioned.")
        state.messages.append({"role": "user", "content": additional_support_prompt})
        
        response = client.chat.completions.create(
            model="llama-3.2-1b-instruct",
            messages=state.messages
        )
        additional_recommendation = response.choices[0].message.content
        print("\n[Support Agent Additional Recommendations]:", additional_recommendation)
        state.messages.append({"role": "assistant", "content": additional_recommendation})
    else:
        print("[Support Agent]: Alright. Remember, if you ever feel like you need more help, please don't hesitate to reach out.")
    
    print("\nThank you for chatting today. Take care!")
    return state

def counselor_recommendation_node(state: ChatState) -> ChatState:
    """
    Counselor Recommendation Agent:
      - Provides a counselor recommendation if the user's mood is serious (suicidal).
    """
    print("\n[Counselor Recommendation]:")
    rec = "It might be helpful to talk to a professional. I strongly recommend contacting a counselor immediately."
    print("[Counselor Recommendation]:", rec)
    state.messages.append({"role": "assistant", "content": rec})
    return state

# --- Build and Run the StateGraph Workflow ---

def run_workflow():
    # Initialize state with default values
    state = ChatState()
    
    # Create a StateGraph with our Pydantic state schema
    graph = StateGraph(ChatState)
    
    # Define nodes and transitions
    graph.add_node("reception", reception_agent_node)
    graph.add_node("analysis", analysis_agent_node)
    graph.add_node("assignment", assignment_agent_node)
    graph.add_node("support", support_agent_node)
    graph.add_node("counselor_recommendation", counselor_recommendation_node)

    # Set up the workflow
    graph.add_edge(START, "reception")
    graph.add_edge("reception", "analysis")
    graph.add_edge("analysis", "assignment")
    graph.add_edge("assignment", "support")
    graph.add_conditional_edges(
        "support",
        lambda state: "counselor_recommendation" if state.mood == "suicidal" else END
    )
    graph.add_edge("counselor_recommendation", END)

    # Compile and run the graph
    app = graph.compile()
    final_state = app.invoke(state)
    
    if final_state.exit:
        print("Ending chat. Goodbye!")

if __name__ == "__main__":
    print("Chatbot started! Type 'exit' at any prompt to end the conversation.\n")
    run_workflow()