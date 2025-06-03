#!/usr/bin/env python
# coding: utf-8

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import requests
import warnings
warnings.filterwarnings("ignore")

# Create Embeddings from Paragraphs
def create_vectorstore(paragraphs, model_name="all-MiniLM-L6-v2"):
    documents = [Document(page_content=p) for p in paragraphs]
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(documents, embedding_model)

# Define Ollama LLM
class OllamaLLM(LLM):
    model: str = "llama3"

    def _call(self, prompt: str, stop=None):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    @property
    def _llm_type(self):
        return "ollama"

# Build RetrievalQA pipeline
def build_retrievalQA_chain(paragraphs):
    vectorstore = create_vectorstore(paragraphs)
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever), retriever

# Run Query
def answer_query(query, qa_chain, retriever):
    answer = qa_chain.run(query)
    relevant_docs = retriever.get_relevant_documents(query)
    return answer, relevant_docs[0].page_content if relevant_docs else "No relevant paragraph found."

if __name__ == "__main__":
    paragraphs = [
        "Quantum physics is the study of particles at the smallest scales.",
        "Photosynthesis allows plants to convert sunlight into energy.",
        "Gravity is a fundamental force that attracts objects with mass."
    ]

    # Get RetrievalQA pipeline, FAISS retriever
    qa_chain, retriever = build_retrievalQA_chain(paragraphs)

    try:
        while True:
            query = input("Enter your query (Ctrl+C to exit):\n")
            answer, relevant_paragraph = answer_query(query, qa_chain, retriever)
            print("Most Relevant Paragraph:\n", relevant_paragraph)
            print("Answer:\n", answer)
    except KeyboardInterrupt:
        print("\nExited by user.")
