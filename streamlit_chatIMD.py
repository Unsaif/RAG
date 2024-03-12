from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import key_param
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import streamlit as st
import time

client = MongoClient(key_param.MONGO_URI)
dbName = "IMD"
collectionName = "collection_of_IMD_text_blobs"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key, disallowed_special=())

vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings, index_name='knn')

template = """
Role and Expertise: You are an AI expert in inherited metabolic diseases. Your primary role is to provide comprehensive and accurate answers to questions related to inherited metabolic disorders. Your expertise is dedicated to understanding, explaining, and offering insights into these conditions.
Diagnostic Proficiency: You are proficient in analyzing diagnostic queries related to inherited metabolic diseases. When presented with symptoms, medical histories, or lab results, use your knowledge to suggest potential diagnoses, always emphasizing the need for professional medical validation.
Scope of Interaction: Focus exclusively on questions about inherited metabolic diseases. Do not engage in or provide answers to queries outside this specific domain. Your purpose is to stay within the boundaries of this medical specialty.
Response Guidelines: Ensure that your responses are informed, clear, and directly relevant to the questions asked. Be concise yet thorough in your explanations, and tailor your language to be accessible to both medical professionals and individuals without specialized medical background.
Ethical Considerations: While you can suggest potential diagnoses based on the information provided, remind users that your guidance is not a substitute for professional medical advice. Avoid making personal judgments or recommendations beyond the scope of inherited metabolic diseases.

Use the following as context for your answer: {context}

Question: {question}

"""
# Don't justify your answers. Don't give information not mentioned in the CONTEXT INFORMATION. If the answer is not in the context, say the words "Sorry, I am unable to answer your question with the information available to me

def query_data(query):
    docs = vectorStore.similarity_search(query, k=1)
    as_output = docs[0].page_content

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key_param.openai_api_key, temperature=0)
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    # qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    # retriever_output = qa.run(query)

    prompt = ChatPromptTemplate.from_template(template)

    def get_relevant_doc_string(_):
        relevent_doc = retriever.invoke(query)
        return relevent_doc[0].page_content
    
    relevent_doc = RunnablePassthrough(get_relevant_doc_string)

    setup_and_retrieval = RunnableParallel(
        {"context": relevent_doc, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | llm
    retriever_output = chain.invoke(query) # , config={'callbacks': [ConsoleCallbackHandler()]}

    return as_output, retriever_output.content

st.title("ChatIMD")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is PKU?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Streamed response emulator
def response_generator(query):
    response = query_data(query)[1]
    for line in response.splitlines():
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        yield '\n'  # Add a newline after each line

# Display assistant response in chat message container
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        response = st.write_stream(response_generator(str(prompt)))
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

# Questions:
# Can you describe PKU?
# What are the clinical features of PKU?
# Is PKU a cause of cataracts?
# Who presented evidence of heterogeneity in PKU?
# Where has brain calcification for PKU been reported?
# Has anyone studied a cohort of women with PKU and if so what was found?
# I have a patient with plasma phenylalanine of 1245 micromoles per litre, hyperactivity, and a low score in general cognitive index. What could this patient have for a disease