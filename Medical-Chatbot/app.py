from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import Pinecone
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


embeddings = download_hugging_face_embeddings()

from langchain_pinecone import PineconeVectorStore

index_name = "medical-chatbot"

# Load the existing index
docsearch = PineconeVectorStore(
    index_name,
    embeddings
)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model=r"C:\Users\Admin\Documents\Medical-Chatbot\model\tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8,
    }
)

from langchain.chains import LLMChain
llm_chain = LLMChain(llm=llm, prompt=PROMPT)

# 3. Create the retriever. This component is also working correctly.
retriever = docsearch.as_retriever(search_kwargs={'k': 2})

# 4. Create a new function to tie them together. This replaces RetrievalQA.
def manual_qa(query: str):
    """
    This function manually replicates the logic of a RetrievalQA chain.
    """
    # First, get the relevant documents from the retriever
    docs = retriever.get_relevant_documents(query)
    
    # Combine the page content of the documents into a single string
    context_string = "\n\n".join([doc.page_content for doc in docs])
    
    # Run the LLMChain, passing the retrieved context and the original question
    result = llm_chain.run({
        'context': context_string,
        'question': query
    })

@app.route("/")
def index():
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)

