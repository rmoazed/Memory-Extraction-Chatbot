import streamlit as st
from baseline_memory_chatbot import BaselineMemoryChatbot
from conflict_memory_chatbot import ConflictMemoryChatbot
from config import client, index
from prompts import system_prompt
import joblib
import numpy as np
import pandas as pd

#---

#load datasets
train_df = pd.read_csv("data/memory_dataset_train_df.csv")
train_embeddings = np.load("data/X_train_embeddings.npy")

#load ML models
softmax_model = joblib.load("models/softmax.joblib")
#random_forest_model = joblib.load("models/random_forest.joblib")
# svm_model = joblib.load("models/svm.joblib")  # later

#---

st.set_page_config( #page title
    page_title="Memory Chatbot",
    layout="wide"
)

#---

def initialize_bot(chatbot_type, extraction_method): #initialize chatbot
    memory_namespace = ( #picking the namespace based on architecture and extraction method
        f"{extraction_method}_baseline_app"
        if chatbot_type == "Baseline"
        else f"{extraction_method}_conflict_app"
    )

    if chatbot_type == "Baseline": #picking correct architecture based on user selection
        chatbot_class = BaselineMemoryChatbot
    else:
        chatbot_class = ConflictMemoryChatbot

    classifier = None
    semantic_train_df = None
    semantic_train_embeddings = None

    if extraction_method == "semantic":
        semantic_train_df = train_df
        semantic_train_embeddings = train_embeddings

    elif extraction_method == "softmax":
        classifier = softmax_model

    #elif extraction_method == "random_forest":
        #classifier = random_forest_model

    # elif extraction_method == "svm":
        #classifier = svm_model

    return chatbot_class(
        openai_client=client,
        pinecone_index=index,
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4.1-mini",
        extraction_method=extraction_method,
        memory_namespace=memory_namespace,
        system_prompt=system_prompt,
        top_k=5,
        classifier=classifier,
        semantic_train_df=semantic_train_df,
        semantic_train_embeddings=semantic_train_embeddings
    )

#---

st.title("Memory-Augmented Chatbot")
st.write( #page text
    "Compare different memory extraction and memory management strategies."
)

with st.sidebar: #making the sidebar
    st.header("Chatbot Configuration") #sidebar header

    chatbot_type = st.radio(
        "Memory architecture",
        ["Baseline", "Conflict-aware"] #pick if you want to use the baseline model or the conflict memory chatbot atchitecture for 
        #chatbot interaction
    )

    extraction_method = st.selectbox( #choose which type of extraction method you want to use for the chatbot you're using
        "Memory extraction method",
        ["semantic", "softmax", "llm"] #will add other ml models later
    )

#---


architecture_key = ( #creating unique session keys (streamlit reruns whole script when user interacts w/ bot, so this keeps
    #conversation history and objects available across reruns
    "baseline"
    if chatbot_type == "Baseline"
    else "conflict"
)

bot_key = f"bot_{architecture_key}_{extraction_method}"
messages_key = f"messages_{architecture_key}_{extraction_method}"

#---

#loading bot into session state
if bot_key not in st.session_state:
    st.session_state[bot_key] = initialize_bot(
        chatbot_type,
        extraction_method
    )

#initialize display history
if messages_key not in st.session_state:
    st.session_state[messages_key] = []
bot = st.session_state[bot_key]

#---

#render exisiting messages
for message in st.session_state[messages_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#---

user_message = st.chat_input("Type a message") #accept one user message

if user_message:
    st.session_state[messages_key].append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = bot.respond_once(user_message)

        st.markdown(result["response"])

    st.session_state[messages_key].append(
        {
            "role": "assistant",
            "content": result["response"]
        }
    )