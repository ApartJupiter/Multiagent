from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# --- RAG Setup ---
class MentalHealthRAG:
    def __init__(self):
        # Sample mental health knowledge base (replace with your own data)
        self.documents = [
            "Cognitive Behavioral Therapy (CBT) is effective for anxiety management.",
            "Mindfulness techniques can help reduce stress and improve emotional regulation.",
            "Signs of depression include persistent sadness and loss of interest in activities.",
            "Regular exercise has been shown to improve mood and reduce anxiety symptoms.",
            "The suicide prevention hotline is available 24/7 at 1-800-273-TALK.",
            "Maintaining a consistent sleep schedule helps with emotional stability.",
            "Social connections are crucial for mental health maintenance.",
            "Breathing exercises can help manage acute anxiety attacks."
        ]
        
        # Create vector store
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self):
        texts = self.text_splitter.create_documents(self.documents)
        return FAISS.from_documents(texts, self.embeddings)
    
    def retrieve(self, query: str, k: int = 3):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# Initialize RAG system
mental_health_rag = MentalHealthRAG()

# --- Modified State Class ---
class ChatState(BaseModel):
    messages: list = []
    exit: bool = False
    mood: str = ""
    context: list = []  # Store retrieved context for RAG

# --- Enhanced Analysis Agent with RAG ---
def analysis_agent_node(state: ChatState) -> ChatState:
    print("\n[Analysis Agent]: Hello! Welcome to our mental health chat.")
    print("[Analysis Agent]: Could you tell me a bit about yourself and how your day has been?")

    user_input = input("You: ")
    if user_input.lower() == "exit":
        state.exit = True
        return state

    # Retrieve relevant mental health information
    state.context = mental_health_rag.retrieve(user_input)
    state.messages.append({"role": "user", "content": user_input})

    # Generate dynamic follow-up questions with RAG context
    for i in range(2):
        followup_prompt = f"""Based on this conversation and mental health knowledge: {state.context}
        Generate 2 relevant follow-up questions to better understand the user's situation.
        Format as:
        1. [Question 1]
        2. [Question 2]"""

        temp_messages = state.messages + [{"role": "user", "content": followup_prompt}]
        
        response = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-7b",
            messages=temp_messages
        )
        generated_questions = response.choices[0].message.content

        questions = [line.split('. ', 1)[-1].strip() 
                     for line in generated_questions.split('\n') 
                     if line.strip().startswith(('1.', '2.', '- '))]

        for question in questions[:2]:
            print(f"\n[Analysis Agent]: {question}")
            answer = input("You: ")
            if answer.lower() == "exit":
                state.exit = True
                return state
            state.messages.append({"role": "user", "content": f"{question} {answer}"})

    # Mood detection with RAG-enhanced context
    full_text = " ".join([msg["content"] for msg in state.messages if msg["role"] == "user"]).lower()
    state.context += mental_health_rag.retrieve(full_text)  # Additional context
    
    detected_mood = None
    for emotion in ["anxious", "depressed", "suicidal", "happy", "stressed", "overwhelmed"]:
        if emotion in full_text:
            detected_mood = emotion
            break
    state.mood = detected_mood or "neutral"

    # Generate conclusion with RAG context
    conclusion_prompt = f"""Based on our conversation and mental health knowledge: {state.context}
    Provide:
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

# --- Enhanced Support Agent with RAG ---
def support_agent_node(state: ChatState) -> ChatState:
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
        
        # Retrieve relevant resources
        resources = mental_health_rag.retrieve(support_details)
        state.messages.append({
            "role": "user", 
            "content": f"{support_details} Available resources: {resources}"
        })

        support_prompt = f"""Based on the user's request for {support_details} 
        and available resources: {resources}
        Provide 3-5 specific recommendations with contact information/resources."""

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

# ... (rest of the original code remains the same) ...