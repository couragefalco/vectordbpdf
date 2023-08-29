from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import chromadb
import time

load_dotenv()

persistent_client = chromadb.PersistentClient(path="chroma")

def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0"
    )

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        client=persistent_client,
        collection_name="manus",
        embedding_function=embedding,
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

def display_sources(source):
    for document in source:
        st.write(f"{document.page_content}")

logo_path = "https://drive.google.com/uc?export=view&id=1sl68t7F0fPCw_BzLRPyPhP9iJG3-cvkQ"
st.image(logo_path, use_column_width=True)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about a igus product:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the prompt using your app's logic
    if 'chain' not in st.session_state:
        st.session_state.chain = make_chain()
        st.session_state.chat_history = []

    response = st.session_state.chain({"question": prompt, "chat_history": st.session_state.chat_history})

    answer = response["answer"]
    source = response["source_documents"]
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))

    # Display AI response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "IgusGO: "
        # Simulate stream of response with milliseconds delay
        for chunk in answer.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # with st.expander('Sources'):
        # display_sources(source)

