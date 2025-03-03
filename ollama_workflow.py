import ollama

# Function to interact with the Llama model served by Ollama
def get_llama_response(user_input):
    response = ollama.chat(model="llama-3.2-1b-instruct", messages=[{"role": "user", "content": user_input}])
    return response['text']

user_input = "How can I improve my mental health?"
response = get_llama_response(user_input)
print(response)
