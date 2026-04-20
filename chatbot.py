# chatbot.py

from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 🔹 Initialize model
llm = ChatOllama(
    model="llama3",
    temperature=0.4
)

# 🔹 Memory (stores conversation)
memory = ConversationBufferMemory()

# 🔹 System prompt
SYSTEM_PROMPT = """
You are a friendly AI health assistant for an educational demo.

You can:
- Explain pneumonia types (normal, bacterial, viral)
- Describe symptoms simply
- Suggest general precautions
- Tell when to see a doctor

You must NOT:
- Give medicines or prescriptions
- Act like a real doctor

Keep answers short, clear, and beginner-friendly.
Always end with: "This is educational, not medical advice."
"""

# 🔹 Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


def get_chatbot_response(user_input, predicted_class=None, confidence=None):
    """Main chatbot function"""

    context = ""

    if predicted_class:
        context = f"""
Model Prediction: {predicted_class}
Confidence: {confidence:.2f}%
"""

    full_input = SYSTEM_PROMPT + context + "\nUser: " + user_input

    try:
        response = conversation.predict(input=full_input)
        return response

    except Exception as e:
        return f"⚠️ Error: {str(e)}"