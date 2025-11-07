# app.py — Streamlit front-end for Pizza RAG
import streamlit as st
from huggingface_hub import InferenceClient
import traceback

# Import retriever from your vector module. Make sure vector.py exposes `retriever`.
from vector import retriever

st.set_page_config(page_title="Pizza RAG Chatbot", page_icon="🍕", layout="centered")
st.title("🍕 Pizza Restaurant RAG Chatbot")
st.caption("Ask questions about the restaurant and get answers grounded in real reviews.")

# Load HF token from Streamlit secrets (add via Streamlit Cloud Secrets)
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Add HF_TOKEN in Streamlit Secrets and redeploy.")
    st.stop()

# Create HF Inference client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

def get_top_k_reviews_text(query, k=5, max_chars=3000):
    """
    Retrieve top-k documents from the vector store and return
    a joined text string truncated to max_chars.
    This handles different retriever method names (`get_relevant_documents`, `retrieve`, etc.).
    """
    try:
        # try common LangChain retriever method names
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
        elif callable(retriever):  # some object may be callable
            docs = retriever(query)
        else:
            raise AttributeError("Retriever has no known retrieval method.")
    except Exception as e:
        st.error("Error when calling retriever. See details below.")
        st.write(traceback.format_exc())
        return ""

    # docs might be a list of Document-like objects or strings
    texts = []
    for d in docs:
        # Support different doc shapes
        if isinstance(d, str):
            texts.append(d)
        else:
            # favor .page_content or .content
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            texts.append(text)

    joined = "\n\n".join(texts[:k])
    if len(joined) > max_chars:
        # naive truncation to avoid hitting token limits
        joined = joined[:max_chars] + "\n\n...[truncated]"
    return joined

def ask_model(reviews_text, question, max_tokens=200):
    """
    Call the HF chat completion API and return the assistant content. Defensive parsing.
    """
    if not reviews_text:
        return "No review context available to answer the question."

    messages = [
        {"role": "system", "content": "You are an expert in answering questions about a pizza restaurant. Answer concisely and only use the provided reviews as context."},
        {"role": "user", "content": f"Reviews:\n{reviews_text}\n\nQuestion:\n{question}"}
    ]
    try:
        response = client.chat_completion(messages=messages, max_tokens=max_tokens)
    except Exception as e:
        st.error("Error calling the HuggingFace Inference API.")
        st.write(traceback.format_exc())
        return ""

    # Defensive extraction of text from response
    try:
        # OpenAI-like response shape
        return response.choices[0].message["content"]
    except Exception:
        # Try alternative shapes
        try:
            return response["generated_text"]
        except Exception:
            return str(response)

# Input area
user_input = st.text_input("Ask a question about the restaurant:", value="", placeholder="e.g., Is the pizza authentic?")

if st.button("Ask") or (user_input and st.session_state.get("auto_run", False)):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant reviews..."):
            reviews_text = get_top_k_reviews_text(user_input, k=5)
        if not reviews_text:
            st.info("No relevant reviews found for your query.")
        else:
            st.markdown("**Top-k retrieved reviews (used as context):**")
            st.write(reviews_text)

            with st.spinner("Asking the model..."):
                answer = ask_model(reviews_text, user_input, max_tokens=200)

            st.markdown("### ✅ Answer")
            st.write(answer)
