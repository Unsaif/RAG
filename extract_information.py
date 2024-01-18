from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "PKU"
collectionName = "collection_of_pku_text_blobs"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

def query_data(query):
    embedded_query = embeddings.embed_query(query)
    docs = vectorStore.similarity_search(embedded_query, K=1)
    as_output = docs[0].page_content

    llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature=0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output

with gr.Blocks(theme=Base(), title="Q&A PKU") as demo:
    gr.Markdown(
        """
        # Question Answering App using Atlas Vector Search + RAG Architecture
        """
    )
    textbox = gr.Textbox(label="Enter your Question:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=10, label="Input with just Atlas Vector Search")
            output2 = gr.Textbox(lines=1, max_lines=10, label="Output generated by chaining Atlas Vector Search")

        button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()