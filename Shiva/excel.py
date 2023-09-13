import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import pandas as pd
from html_tags import css,bot_template,user_template
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI



# def get_conversation(vector_store):
#     llm = ChatOpenAI() #LLM model in conversation
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) #this helps in keeping storing messages 
#     conversation_chain = ConversationalRetrievalChain.from_llm(  
#         llm=llm,    
#         retriever=vector_store.as_retriever(),  #in order to get quickly find and retrieve relevant information from db
#         memory=memory                           #chat history
#     )#builds chain between
#     return conversation_chain   

def handle_userinput(user_question,df):    #parameter here is the question
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    output = agent.run(user_question)
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True) 
    st.write(bot_template.replace("{{MSG}}", output), unsafe_allow_html=True)



def main():
    load_dotenv()                            #loading evironment variables
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")  #title tags of Streamlit
    st.write(css, unsafe_allow_html=True)    #to write raw HTML results in HTML output where all elements and attributes are set to lowercase. 

    if "conversation" not in st.session_state:      #if conversation is not in session state then its will be intialized to none
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:      #if chat_histoy not in session state we cannot store chat history cuz its not intialized 
        st.session_state.chat_history = None
    with st.sidebar:
        st.subheader("Your Documents")
        docs=st.file_uploader("Upload your excel documents and click on Process")
        df = pd.read_excel(docs)

    st.header("Chat with excel document...") 
    user_question = st.text_input("Ask a question about your documents:") #input tags of Streamlit
    if user_question:                                                    #if user_question exists question will be passed in arguments to handle_userinput 
        handle_userinput(user_question,df)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        output = agent.run(user_question)
        # st.write(output)

                                     
    
        
if __name__ =="__main__":
    main()
