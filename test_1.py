from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def reception_agent(messages):
    """
    Reception Agent:
      - Greets the user.
      - Asks for a brief introduction about themselves and their day.
    """
    greeting = "Hello, welcome to our mental health chat!"
    prompt = f"{greeting}\nCan you tell me a bit about yourself and how your day is going?"
    print("\nReception Agent:", prompt)
    
    user_input = input("You: ")
    if user_input.lower() == "exit":
        return None

    messages.append({"role": "user", "content": user_input})
    return messages

def analysis_agent(messages):
    """
    Analysis Agent:
      - Asks additional questions to understand the user’s emotional state.
      - If the user expresses negative emotions, it dives deeper.
    """
    analysis_prompt = "How are you feeling right now? (e.g., anxious, depressed, suicidal, happy, etc.)"
    print("\nAnalysis Agent:", analysis_prompt)
    
    feeling = input("You: ")
    if feeling.lower() == "exit":
        return None

    messages.append({"role": "user", "content": feeling})

    # If negative feelings are detected, ask for further details.
    if feeling.lower() in ["anxious", "depressed", "suicidal"]:
        followup_prompt = "Can you share more about why you're feeling this way?"
        print("Analysis Agent:", followup_prompt)
        reason = input("You: ")
        if reason.lower() == "exit":
            return None
        messages.append({"role": "user", "content": reason})
    
    return messages

def assignment_agent(messages):
    """
    Assignment Agent:
      - Analyzes previous responses.
      - Dynamically asks follow-up questions based on the user's emotional state.
      - Provides a conclusion and recommendations based on the entire conversation.
    """
    print("\nAssignment Agent:")

    # Determine the user's emotional state from the conversation history.
    user_feeling = None
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"].lower()
            if any(feeling in content for feeling in ["anxious", "depressed", "suicidal", "happy"]):
                # We assume the first instance of one of these keywords represents the primary state.
                if "anxious" in content:
                    user_feeling = "anxious"
                elif "depressed" in content:
                    user_feeling = "depressed"
                elif "suicidal" in content:
                    user_feeling = "suicidal"
                elif "happy" in content:
                    user_feeling = "happy"
                break

    # If no clear emotional state is detected, ask the user explicitly.
    if user_feeling is None:
        clarification = "I'm here to help. Could you clarify how you're feeling right now?"
        print("Assignment Agent:", clarification)
        user_feeling = input("You: ").lower()
        if user_feeling == "exit":
            return None
        messages.append({"role": "user", "content": clarification + " " + user_feeling})

    # Ask tailored follow-up questions based on the detected emotional state.
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
    print("Assignment Agent:", q1)
    answer1 = input("You: ")
    if answer1.lower() == "exit":
        return None
    messages.append({"role": "user", "content": q1 + " " + answer1})

    print("Assignment Agent:", q2)
    answer2 = input("You: ")
    if answer2.lower() == "exit":
        return None
    messages.append({"role": "user", "content": q2 + " " + answer2})

    # Now, ask the AI for a final conclusion and recommendations.
    conclusion_prompt = ("Based on our conversation so far, please provide a thoughtful conclusion along with "
                         "tailored recommendations and actionable self-care or professional advice.")
    messages.append({"role": "user", "content": conclusion_prompt})

    response = client.chat.completions.create(
        model="llama-3.2-1b-instruct",
        messages=messages
    )

    conclusion = response.choices[0].message.content
    print("\nAssignment Agent Conclusion:", conclusion)
    messages.append({"role": "assistant", "content": conclusion})
    return messages

def support_agent(messages):
    """
    Support Agent:
      - After the conclusion, checks if the user needs any additional support.
      - If the user requires more support, further questions are asked and additional AI-generated advice is provided.
    """
    print("\nSupport Agent:")
    support_query = "Do you feel that you need any additional support or guidance at this moment? (yes/no)"
    print("Support Agent:", support_query)
    answer = input("You: ")
    if answer.lower() == "exit":
        return None
    messages.append({"role": "user", "content": support_query + " " + answer})
    
    if answer.strip().lower() in ["yes", "y"]:
        followup_support = "Could you please elaborate on what kind of support you need or what you're currently struggling with?"
        print("Support Agent:", followup_support)
        support_details = input("You: ")
        if support_details.lower() == "exit":
            return None
        messages.append({"role": "user", "content": followup_support + " " + support_details})
        
        # Now ask the AI for additional support recommendations.
        additional_support_prompt = ("Based on our entire conversation, including your recent input about needing additional support, "
                                     "please provide further guidance, self-care tips, or suggestions to help address the issues mentioned.")
        messages.append({"role": "user", "content": additional_support_prompt})
        
        response = client.chat.completions.create(
            model="llama-3.2-1b-instruct",
            messages=messages
        )
        additional_recommendation = response.choices[0].message.content
        print("\nSupport Agent Additional Recommendations:", additional_recommendation)
        messages.append({"role": "assistant", "content": additional_recommendation})
    else:
        print("Support Agent: Alright. Remember, if you ever feel like you need more help, please don't hesitate to reach out.")

    print("\nThank you for chatting today. Take care!")
    return messages

def run_workflow():
    """
    Runs the multi-agent workflow:
      1. Reception Agent collects an initial message.
      2. Analysis Agent probes deeper into the user's feelings.
      3. Assignment Agent asks tailored follow-up questions and provides a conclusion.
      4. Support Agent checks if further support is needed and, if so, provides additional recommendations.
    """
    messages = []
    print("Chatbot started! Type 'exit' at any prompt to end the conversation.\n")
    
    # Reception Agent
    messages = reception_agent(messages)
    if messages is None:
        print("Ending chat. Goodbye!")
        return
    
    # Analysis Agent
    messages = analysis_agent(messages)
    if messages is None:
        print("Ending chat. Goodbye!")
        return
    
    # Assignment Agent
    messages = assignment_agent(messages)
    if messages is None:
        print("Ending chat. Goodbye!")
        return

    # Support Agent
    messages = support_agent(messages)
    if messages is None:
        print("Ending chat. Goodbye!")
        return

if __name__ == "__main__":
    run_workflow()
