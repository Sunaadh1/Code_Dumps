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

from html_tags import css,bot_template,user_template





def get_document_text(doc_files):  #handles pdf,docx,txt files

    text = ""

    for doc_file in doc_files:

        if doc_file.name.lower().endswith('.pdf'):

            # Handle pdf files

            pdf_reader = PdfReader(doc_file)

            for page in pdf_reader.pages:

                text += page.extract_text()

        elif doc_file.name.lower().endswith('.docx'):

            # Handle DOCX files

            doc = Document(doc_file)

            for paragraph in doc.paragraphs:

                text += paragraph.text + '\n'

        elif doc_file.name.lower().endswith('.txt'):

            # Handle txt files

            text += doc_file.read().decode('utf-8')

        

    return text







def get_text_chunks(text):              #create a chunk

    text_splitter = CharacterTextSplitter(

        separator="\n",

        chunk_size=1000,

        chunk_overlap=200,

        length_function=len

    )

    chunks = text_splitter.split_text(text)     #by using text_splitter style we split it into chunks

    return chunks





def get_vector_store(text_chunks):              

    embeddings = OpenAIEmbeddings()     #OpenAi Embedding style

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #stores textchunks in FAISS vector database

    return vectorstore



def get_conversation(vector_store):

    llm = ChatOpenAI() #LLM model in conversation

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) #this helps in keeping storing messages 

    conversation_chain = ConversationalRetrievalChain.from_llm(  

        llm=llm,    

        retriever=vector_store.as_retriever(),  #in order to get quickly find and retrieve relevant information from db

        memory=memory                           #chat history

    )#builds chain between

    return conversation_chain   



def handle_userinput(user_question):    #parameter here is the question

    response = st.session_state.conversation({'question': user_question})       #it will generate a response to query

    st.session_state.chat_history = response['chat_history'] #response is json object in that there is something called 'chat_history'  that will be stored in the chat



    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:  #1st message is a user input 2nd message is bot output 

            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 

        else:

            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    load_dotenv()                            #loading evironment variables

    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")  #title tags of Streamlit

    st.write(css, unsafe_allow_html=True)    #to write raw HTML results in HTML output where all elements and attributes are set to lowercase. 



    if "conversation" not in st.session_state:      #if conversation is not in session state then its will be intialized to none

        st.session_state.conversation = None

    if "chat_history" not in st.session_state:      #if chat_histoy not in session state we cannot store chat history cuz its not intialized 

        st.session_state.chat_history = None



    st.header("Chat with multiple documents :books:")                          #heading tags of Streamlit

    user_question = st.text_input("Ask a question about your documents:") #input tags of Streamlit

    if user_question:                                                    #if user_question exists question will be passed in arguments to handle_userinput 

        handle_userinput(user_question)                                 





    with st.sidebar:

        st.subheader("Your Documents")

        docs=st.file_uploader("Upload you documents and click on Process",accept_multiple_files=True)

        # url=st.text_input("enter:")

        if st.button("Process"):

            with st.spinner("Processing"):

                #get docx text

                raw_text=get_document_text(docs)

                # st.write(raw_text)    #test

                

                #get text chunks

                text_chunks=get_text_chunks(raw_text)

                # st.write(text_chunks)   #test



                #vector store

                vector_store=get_vector_store(text_chunks)

                # st.write(vector_store)    #test



                #conversation chain

                st.session_state.conversation = get_conversation(vector_store)

                #here the above code will be object which stores History of chats

                #session_state prevents it from losing history from chat in the when we reload and can be used outside the scope

if __name__ =="__main__":

    main()
