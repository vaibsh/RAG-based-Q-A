# RAG-based-Q-A
One or more short paragraphs are given. Given a question from the user, the task is to answer the question based on the content in
the paragraphs. RAG based LLMs are used for this task. LLM used is llama3 hosted locally using the tool ollama. A vector store of the 
paragraphs is constructed using HuggingFaceEmbeddings and FAISS. The query is converted into embedding, relevant paragraphs are fetched. 
The query and the relevant paragraphs are sent to the LLM which generates an answer based on the relevant paragraphs.

## Usage
1. Run the program using the command:<br>
    python main.py<br>
2. Enter the query of Ctrl+C to exit<br>
3. Most relevant paragraph and the answer is printed
