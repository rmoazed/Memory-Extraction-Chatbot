import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import streamlit as st


env_path = Path(__file__).resolve().parent / ".env"

loaded = load_dotenv(
    dotenv_path=env_path,
    override=True
)

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "default"
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"] if "PINECONE_API_KEY" in st.secrets else "default"


#if not env_path.exists():
    #raise FileNotFoundError(f".env file not found at: {env_path}")

#if not loaded:
    #raise RuntimeError(f"python-dotenv could not load: {env_path}")

#if not OPENAI_API_KEY:
    #raise ValueError("OPENAI_API_KEY is missing from the .env file")

#if not PINECONE_API_KEY:
    #raise ValueError("PINECONE_API_KEY is missing from the .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("your_index_name")
