import openai
import os
import json
import uuid
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from schemas import MemoryExtractionResult
from config import client, index
from prompts import system_prompt


class BaselineMemoryChatbot:

    def __init__(
        self,
        openai_client,
        pinecone_index,
        embedding_model,
        chat_model,
        memory_namespace,
        extraction_method,
        system_prompt,
        top_k,
        semantic_train_df=None,
        semantic_train_embeddings=None,
        classifier=None
    ):
        self.client = openai_client
        self.index = pinecone_index
        self.system_prompt = system_prompt #prompt user gives to chatbot
        self.messages = [{"role":"system","content":system_prompt}] #message format
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.memory_namespace = memory_namespace
        self.extraction_method = extraction_method
        self.classifier = classifier #trained ml model
        self.top_k = top_k #so can ask for top k memories
        self.semantic_train_df = semantic_train_df
        self.semantic_train_embeddings = semantic_train_embeddings
        
    def get_embedding(self, text): #function for generating embedding
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
    )
        return response.data[0].embedding
    
    def store_memory(self, memory_text, metadata=None): #function for upserting vectors to pinecone
        embedding = self.get_embedding(memory_text) #get embedding

        memory_id = str(uuid.uuid4()) #create memory id

        self.index.upsert( #upsert the embedding to the pinecone index
            vectors=[
                {
                    "id": memory_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ],
            namespace=self.memory_namespace #rather than have many indicies going to have one index w/ namespaces for
            #simplicities sake
        )

        return memory_id
        
    def extract_memory_semantic(self,user_message,k): 
        message_embedding = self.get_embedding(user_message) #change from when function was written outside of bot class
        #now calling on function defined in class to get the embedding 
        message_embedding = np.array(message_embedding).reshape(1, -1) 
        similarities = cosine_similarity(
        message_embedding,
        self.semantic_train_embeddings)[0] #change from when function was written outside of bot class
        top_k_indices = np.argsort(similarities)[-k:][::-1] 
        top_k_examples = self.semantic_train_df.iloc[top_k_indices].copy() #change from when function written outside of 
        #bot class
        top_k_examples["similarity"] = similarities[top_k_indices] 
        weighted_scores = top_k_examples.groupby("importance_score")["similarity"].sum() 
        predicted_importance = weighted_scores.idxmax() 
        store = predicted_importance > 1 
        return {
        "importance": predicted_importance,
        "store": store,
        "top_k_examples": top_k_examples,
        "weighted_scores": weighted_scores
    }
        
    def extract_memory_ml(self, user_message):
        embedding = self.get_embedding(user_message) #get embedding
        embedding = np.array(embedding).reshape(1, -1) #reshape vector
        predicted_importance = self.classifier.predict(embedding)[0] #model predicts importance score
        store = predicted_importance > 1 #store the memory if importance score > 1

        return {
            "importance": predicted_importance,
            "store": store,
            "method": self.extraction_method 
            }

    def extract_memory_llm(self, user_message): #changed to this method because llm was not consistently producing
        #valid json, so in middle of chats would run into errors
        prompt = f"""
        Classify this user message for long-term memory extraction.

        Importance scale:
        5 = critical constraint
        4 = stable preference or goal
        3 = useful recurring context
        2 = temporary context
        1 = do not store

        Write memory_text as a concise third-person memory.
        If importance is 1, memory_text must be an empty string.

        User message:
        {user_message}
        """

        response = self.client.responses.parse(
        model=self.chat_model,
        input=prompt,
        text_format=MemoryExtractionResult
        )

        parsed = response.output_parsed
        result = parsed.model_dump()

        result["store"] = result["importance"] > 1

        if not result["store"]:
            result["memory_text"] = ""

        return result

    def extract_memory(self, user_message): #function for determining which memory extraction method to use based 
        
        if self.extraction_method == "semantic": #if extraction method designated as semantic use extract_memory_semantic
            #function for extracting memory
            return self.extract_memory_semantic(user_message, self.top_k)

        elif self.extraction_method in ["softmax", "random_forest", "svm"]: #if extraction method is defined as one of the 
            #ml models use the extract_memory_ml function for memory extraction
            return self.extract_memory_ml(user_message)

        elif self.extraction_method == "llm": #if extraction method is defined as llm use extract_memory_llm function as 
            #method for extracting memory
            return self.extract_memory_llm(user_message)

        else:
            raise ValueError("Unknown extraction method") #just in case an extraction method gets input by accident that
            #isn't in the defined list, give an error message

    def process_memory(self, user_message): #this is where the memories are actually being upserted in the pipeline
        result = self.extract_memory(user_message) #bot extracts memory using rules defined in above function/decides if
        #memory should be stored

        if result["store"]: #if store is False, don't store it/skip storing. if store is True, then store_memory function
            #gets called
            memory_text = result.get("memory_text", user_message)

            metadata = { #metadata that will be stored alongside each vector upserted to pinecone
                "source_message": user_message,
                "importance": int(result["importance"]),
                "method": self.extraction_method,
                "memory_text": memory_text,
            }

            memory_id = self.store_memory(memory_text, metadata=metadata)
            result["memory_id"] = memory_id

        return result
        
    def retrieve_memories(self, query_text, top_k=None):
        embedding = self.get_embedding(query_text) #get the embedding using the get_embedding function
        if top_k is None: #top k is user defined (user picks k)
            top_k = self.top_k
        
        #query pinecone index (the below is pinecone api)
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.memory_namespace
        )

        memories = []#empty list to store memories
        
        for match in results["matches"]: #loop through results
            memory = match["metadata"]["memory_text"] #get memory text from metadata
            memories.append(memory) #append memory text to memories
        
        return memories 

    #basically doing the same thing as the retrieve_memories function but for the namespace in the pinecone index that has
    #predefined knowledge pre any conversation with user
    def retrieve_knowledge(self, query_text, top_k=None):
        embedding = self.get_embedding(query_text) #get embedding

        if top_k is None:
            top_k = self.top_k #user chooses k

        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="knowledge" #knowledge namespace!
        )

        knowledge = [] #empty list to add knowledge to

        for match in results["matches"]: #loop through results
            text = match["metadata"]["text"] #get text from metadata
            knowledge.append(text) #append text to knowledge 

        return knowledge
        
    def generate_response(self, question):
        #add user's message to conversation history
        self.messages.append({"role": "user", "content": question})

        #get GPT response
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=self.messages
        )
        #extract assistant's content
        content = response.choices[0].message.content
        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": content})

        return content

        
    def chat(self):
        print("To terminate the conversation, type 'END'.") 
        question = ""

        while question.upper() != "END":
            question = input("\nYou: ").strip()

            if question.upper() == "END": #if the user inputs END the chatbot says goodbye
                print("Goodbye!")
                break #conversation ends

            print("\nBot is typing...\n") #while user is waiting for bot response to generate this prints
            
            self.process_memory(question) #memory storage step
            
            memories = self.retrieve_memories(question) #uses the retrieve_memories function to incorporate any necessary 
            #stored memories into response

            knowledge_results = self.retrieve_knowledge(question) #also use/retrieve knowledge-based context stored in index
            #under separate namespace
            
            memory_context = "\n".join(memories) #joining the relevant memories together
            knowledge_context = "\n".join(knowledge_results) #joining the relevant context together

            #add both contexts to system prompt
            system_message_with_context = f""" 
    {self.system_prompt}

    Relevant long-term memories:
    {memory_context}

    Relevant knowledge base context:
    {knowledge_context}
    """

            self.messages[0]["content"] = system_message_with_context

            response_content = self.generate_response(question) #generate response

            print(f"Bot: {response_content}")

    def respond_once(self, user_message): #adding b/c streamlit can't use things like input(), while, etc.
        memory_result = self.process_memory(user_message)

        memories = self.retrieve_memories(user_message)
        knowledge = self.retrieve_knowledge(user_message)

        memory_context = "\n".join(memories)
        knowledge_context = "\n".join(knowledge)

        self.messages[0]["content"] = f"""
        {self.system_prompt}

        Relevant long-term memories:
        {memory_context}

        Relevant knowledge base context:
        {knowledge_context}
        """

        response_text = self.generate_response(user_message)

        return {
            "response": response_text,
            "memory_result": memory_result,
            "retrieved_memories": memories
        }


