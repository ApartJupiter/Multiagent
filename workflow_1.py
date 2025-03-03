import string
import nltk
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

# State
class State(TypedDict):
    user_input: str
    category: str
    response: str
    follow_up: bool = False # New state to track if follow up is needed

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# List of stop words to ignore
STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
              "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
              "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
              "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
              "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
              "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
              "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
              "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
              "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
              "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
              "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
              "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now","want"}

# Enhanced synonym generator with phrase support
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().lower().replace('_', ' ')
            synonyms.add(synonym)
    return list(synonyms)

# Seed words with multi-word support
CATEGORY_SEEDS = {
    "crisis_intervention": ["suicide", "self-harm"],
    "anxiety_support": ["anxiety", "panic attack"],
    "depression_support": ["depression", "hopelessness"]
}

# Generate keywords with manual enhancements
CATEGORY_KEYWORDS = {}
ALL_KEYWORDS = set()

for category, seeds in CATEGORY_SEEDS.items():
    category_keywords = []
    for seed in seeds:
        # Get synonyms and add as both phrases and individual words
        synonyms = get_synonyms(seed)
        category_keywords.extend(synonyms)
        
        # Split multi-word seeds into components
        if ' ' in seed:
            category_keywords.extend(seed.split())
    
    # Manual enhancements
    if category == "crisis_intervention":
        category_keywords += [
            "kill myself", "end it all", "suicidal", "self harm",
            "kill", "myself", "die", "want to die", "cutting"
        ]
    elif category == "anxiety_support":
        category_keywords += [
            "stressed", "worried", "overwhelmed", "nervous",
            "panic", "racing thoughts", "anxious"
        ]
    elif category == "depression_support":
        category_keywords += [
            "sad", "empty", "numb", "lonely", "can't go on",
            "worthless", "depressed"
        ]
    
    # Normalize and deduplicate
    normalized_keywords = set()
    for kw in category_keywords:
        # Handle multi-word phrases
        if ' ' in kw:
            normalized_keywords.add(kw)
        # Add lemmatized versions for single words
        else:
            normalized_keywords.add(kw)  # Keep original word
            normalized_keywords.add(lemmatizer.lemmatize(kw, pos='v'))  # Verb form
            normalized_keywords.add(lemmatizer.lemmatize(kw, pos='n'))  # Noun form
    
    CATEGORY_KEYWORDS[category] = list(normalized_keywords)
    ALL_KEYWORDS.update(normalized_keywords)

# Enhanced decision logic
def decide_category(state) -> Literal["anxiety_support", "depression_support", "crisis_intervention", "general_support"]:
    user_input = state['user_input'].lower()
    
    # Check for crisis phrases first
    crisis_phrases = [p for p in CATEGORY_KEYWORDS["crisis_intervention"] if ' ' in p]
    for phrase in crisis_phrases:
        if phrase in user_input:
            return "crisis_intervention"
    
    # Process individual words with lemmatization
    words = [
        lemmatizer.lemmatize(word.strip(string.punctuation), pos='v')
        for word in user_input.split()
    ]
    
    # Check crisis words
    crisis_words = [w for w in CATEGORY_KEYWORDS["crisis_intervention"] if ' ' not in w]
    if any(word in crisis_words for word in words):
        return "crisis_intervention"
    
    # Check anxiety phrases/words
    anxiety_phrases = [p for p in CATEGORY_KEYWORDS["anxiety_support"] if ' ' in p]
    for phrase in anxiety_phrases:
        if phrase in user_input:
            return "anxiety_support"
    
    anxiety_words = [w for w in CATEGORY_KEYWORDS["anxiety_support"] if ' ' not in w]
    if any(word in anxiety_words for word in words):
        return "anxiety_support"
    
    # Check depression phrases/words
    depression_phrases = [p for p in CATEGORY_KEYWORDS["depression_support"] if ' ' in p]
    for phrase in depression_phrases:
        if phrase in user_input:
            return "depression_support"
    
    depression_words = [w for w in CATEGORY_KEYWORDS["depression_support"] if ' ' not in w]
    if any(word in depression_words for word in words):
        return "depression_support"
    
    # If none of the above, check for positive feelings
    if all(word in ALL_KEYWORDS for word in words) and not any(word in STOP_WORDS for word in words):
        print("What would you like to discuss or explore further?")
        return "general_support"
    
    # Return general support
    return "general_support"


# Agents (updated with better responses)
def reception_agent(state):
    state['follow_up'] = False  # Reset follow-up flag each time
    response = "Hi there, I'm here to listen. How can I support you today?"
    return {
        "response": response,
        "follow_up": True,
    }

def anxiety_support_agent(state):
    if not 'follow_up' in state or state['follow_up']:
        return {
            "response": "I understand anxiety can be overwhelming. Let's try some grounding techniques together. Breathe in for 4 counts, hold for 4, and exhale for 6.",
            "follow_up": True,
        }
    else:
        response = "It sounds like you're feeling something positive! How are you doing today?"
        return {
            "response": response,
            "follow_up": False,
        }


def depression_support_agent(state):
    return {
        "response": "I'm sorry you're feeling this way. You're not alone in this. Would you like to share more about what you're experiencing?",
        **state
    }

def crisis_intervention_agent(state):
    return {
        "response": "I'm deeply concerned about your safety. Please contact the National Suicide Prevention Lifeline at 988 or text HOME to 741741. You're not alone.",
        **state
    }

def general_support_agent(state):
    return {
        "response": "Thank you for sharing. I'm here to listen. Could you tell me more about what's been on your mind?",
        **state
    }


builder = StateGraph(State)
builder.add_node("reception_agent", reception_agent)
builder.add_node("anxiety_support_agent", anxiety_support_agent)
builder.add_node("depression_support_agent", depression_support_agent)
builder.add_node("crisis_intervention_agent", crisis_intervention_agent)
builder.add_node("general_support_agent", general_support_agent)

# Add follow-up state and agent
builder.add_node("follow_up_agent", lambda state: {
    "response": "I notice you're feeling joyous. What would you like to discuss or explore further?",
    "follow_up": False,
})

builder.add_edge(START, "reception_agent")
builder.add_conditional_edges("reception_agent", decide_category, {
    "crisis_intervention": "crisis_intervention_agent",
    "anxiety_support": "anxiety_support_agent",
    "depression_support": "depression_support_agent",
    "general_support": "general_support_agent"
})

# Add edge from follow-up agent to END
builder.add_edge("follow_up_agent", END)

builder.add_edge("crisis_intervention_agent", END)
builder.add_edge("general_support_agent", END)
builder.add_edge("depression_support_agent", END)
builder.add_edge("anxiety_support_agent", END)


graph = builder.compile()

# Enhanced interaction flow
while True:
    print("\n" + "="*40)
    user_input = input("How are you feeling today? (Type 'exit' to quit)\n> ").lower()
    
    if user_input == 'exit':
        break
        
    result = graph.invoke({"user_input": user_input, "category": "", "response": ""})
    
    print("\n" + "-"*20)
    print(result["response"])
    
    # Show unrecognized terms analysis (ignoring stop words)
    input_words = [
        lemmatizer.lemmatize(word.strip(string.punctuation), pos='v')
        for word in user_input.split()
    ]
    unrecognized = [w for w in input_words if w not in ALL_KEYWORDS and w not in STOP_WORDS]
    
    # If positive feeling detected, prompt follow-up
    if all(word in ALL_KEYWORDS for word in input_words) and not any(word in STOP_WORDS for word in input_words):
        print("\nI notice you're feeling joyous. What would you like to discuss or explore further?")
        continue
    
    if unrecognized:
        print("\n/System Note: These terms weren't recognized in our support vocabulary]")
        print(f"Unrecognized: {', '.join(unrecognized)}")
        print("[Consider adding these to our training data]")

# If loop ended, provide final thank you message
print("\nThank you for sharing your feelings. Have a wonderful day!")
