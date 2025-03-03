from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Define the state schema
class ChatState(BaseModel):
    messages: list = []
    exit: bool = False
    mood: str = ""

# --- Analysis Agent (Merged Reception) ---
def analysis_agent_node(state: ChatState) -> ChatState:
    """
    Enhanced Analysis Agent:
      - Greets the user and asks about their day.
      - Dynamically generates follow-up questions.
      - Deduces mood after all responses.
      - Provides AI-powered conclusions.
    """
    print("\n[Analysis Agent]: Hello! Welcome to our mental health chat.")
    print("[Analysis Agent]: Could you tell me a bit about yourself and how your day has been?")

    user_input = input("You: ")
    if user_input.lower() == "exit":
        state.exit = True
        return state

    state.messages.append({"role": "user", "content": user_input})

    # Generate dynamic follow-up questions (4 total)
    for i in range(2):  # Two rounds of 2 questions each
        followup_prompt = """Based on this conversation, generate 2 relevant follow-up questions 
        to better understand the user's situation. Format as:
        1. [Question 1]
        2. [Question 2]"""

        temp_messages = state.messages + [{"role": "user", "content": followup_prompt}]
        
        response = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-7b",
            messages=temp_messages
        )
        generated_questions = response.choices[0].message.content

        # Extract and ask questions
        questions = [line.split('. ', 1)[-1].strip() for line in generated_questions.split('\n') if line.strip().startswith(('1.', '2.', '- '))]

        for question in questions[:2]:
            print(f"\n[Analysis Agent]: {question}")
            answer = input("You: ")
            if answer.lower() == "exit":
                state.exit = True
                return state
            state.messages.append({"role": "user", "content": f"{question} {answer}"})

    # Infer mood based on responses
    full_text = " ".join([msg["content"] for msg in state.messages if msg["role"] == "user"]).lower()
    detected_mood = None
    for emotion in ["anxious", "depressed", "suicidal", "happy", "stressed", "overwhelmed"]:
        if emotion in full_text:
            detected_mood = emotion
            break
    state.mood = detected_mood or "neutral"

    # Generate conclusion
    conclusion_prompt = """Based on our conversation, provide:
    1. Key insights
    2. Personalized recommendations
    3. Actionable steps"""

    state.messages.append({"role": "user", "content": conclusion_prompt})
    
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-7b",
        messages=state.messages
    )
    conclusion = response.choices[0].message.content
    print("\n[Analysis Agent Conclusion]:\n", conclusion)
    state.messages.append({"role": "assistant", "content": conclusion})
    
    return state

# --- Support Agent ---
def support_agent_node(state: ChatState) -> ChatState:
    """
    Support Agent:
      - Offers additional support options
      - Provides final recommendations
    """
    print("\n[Support Agent]: Would you like to explore additional support options or resources? (yes/no)")
    answer = input("You: ")
    if answer.lower() == "exit":
        state.exit = True
        return state
    state.messages.append({"role": "user", "content": answer})

    if answer.strip().lower() in ["yes", "y"]:
        print("[Support Agent]: What specific type of support are you looking for?")
        support_details = input("You: ")
        if support_details.lower() == "exit":
            state.exit = True
            return state
        state.messages.append({"role": "user", "content": support_details})

        support_prompt = f"""Based on the user's request for {support_details}, 
        provide 3-5 specific recommendations with contact information/resources."""

        response = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-7b",
            messages=state.messages
        )
        recommendations = response.choices[0].message.content
        print("\n[Support Agent Recommendations]:\n", recommendations)
        state.messages.append({"role": "assistant", "content": recommendations})
    else:
        print("[Support Agent]: Remember, help is always available when you need it.")

    print("\nThank you for chatting today. Wishing you well!")
    return state

# --- Counselor Recommendation ---
def counselor_recommendation_node(state: ChatState) -> ChatState:
    """
    Counselor Recommendation:
      - Triggers emergency support if needed
    """
    print("\n[Counselor Recommendation]:")
    rec = """Urgent Support Resources:
    - National Suicide Prevention Lifeline: 1-800-273-TALK (8255)
    - Crisis Text Line: Text HOME to 741741
    - Immediate local emergency services: 911"""
    print(rec)
    state.messages.append({"role": "assistant", "content": rec})
    return state

# --- Build and Run the StateGraph Workflow ---
def run_workflow():
    state = ChatState()
    graph = StateGraph(ChatState)
    
    graph.add_node("analysis", analysis_agent_node)
    graph.add_node("support", support_agent_node)
    graph.add_node("counselor", counselor_recommendation_node)

    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "support")
    graph.add_conditional_edges(
        "support",
        lambda state: "counselor" if state.mood == "suicidal" else END
    )
    graph.add_edge("counselor", END)

    app = graph.compile()
    final_state = app.invoke(state)

    if final_state.exit:
        print("Ending chat. Goodbye!")

if __name__ == "__main__":
    print("Chatbot started! Type 'exit' at any prompt to end the conversation.\n")
    run_workflow()
