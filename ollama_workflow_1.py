from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Store conversation history
messages = []

print("Chatbot started! Type 'exit' to end the conversation.")

while True:
    user_message = input("You: ")
    
    if user_message.lower() == "exit":
        print("Ending chat. Goodbye!")
        break  # Exit the loop
    
    # Add user's message to the conversation history
    messages.append({"role": "user", "content": user_message})

    # Get AI response
    response = client.chat.completions.create(
        model="llama-3.2-1b-instruct",
        messages=messages
    )

    ai_message = response.choices[0].message.content
    print("AI:", ai_message)

    # Add AI response to conversation history
    messages.append({"role": "assistant", "content": ai_message})
