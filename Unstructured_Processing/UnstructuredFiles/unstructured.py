import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import PyPDF2
import docx



os.environ["OPENAI_API_KEY"] = "sk-..."     #openAI key

def enter_unstructured_docs():                #function to enter unstructured files
    file_list = []
    while True:
        file_name = input("Enter a file name (or 'q' to quit): ")   
        if file_name.lower() == 'q':
            break
        file_list.append(file_name)
    
    return file_list

def pdf_to_string(pdf_path):                #pdf to string
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            pdf_text += page.extract_text()
    except FileNotFoundError:
        return f"File not found: {pdf_path}"
    except Exception as e:
        return f"Error while processing PDF: {e}"
    return pdf_text

    
def docx_to_text(docx_path):            #docs to string
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except FileNotFoundError:
        return f"File not found: {docx_path}"
    except Exception as e:
        return f"Error while processing DOCX: {e}"
    return text

def text_file_to_string(txt_path):      #txt to string
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()
        return text
    except FileNotFoundError:
        return f"File not found: {txt_path}"
    except Exception as e:
        return f"Error while processing TXT: {e}"

def convert_files_to_text(file_list):           #list of file to single text inside all the files
    text = ""  # Initialize an empty string to store the concatenated text
    for file_name in file_list:
        file_extension = os.path.splitext(file_name)[-1].lower()
        if file_extension == ".pdf":
            text += pdf_to_string(file_name)
        elif file_extension == ".docx":
            text += docx_to_text(file_name)
        elif file_extension == ".txt":
            text += text_file_to_string(file_name)
        else:
            print(f"Unsupported file type: {file_extension}")    #rejects unsupported file types
    return text


def splitter(pdf_text):                 #text splitter
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
    )
    texts = text_splitter.split_text(pdf_text)
    # print(type(texts))
    # print(len(texts))
    return texts

def embeddings():               #helps in vectorizing
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    return embeddings

def  VectorDB(texts,embeddings):       #vectorDB
    # print(type(texts))
    # print(type(embeddings))
    document_search = FAISS.from_texts(texts, embeddings) #stores in FAISS db
    return document_search                      

def CreateChain():
    chain = load_qa_chain(OpenAI(), chain_type="stuff") #helps in  loads a chain that can be used to answer questions by retrieving documents and feeding them to an OpenAI LLM.
    return chain

def EnterQuery():#entering query
    query=input("Enter the input :")
    return query

def ChainRun(query,chain,db): #here we pas query,chain and VectorDB
    docs = db.similarity_search(query)#similarity Search 
    print(chain.run(input_documents=docs, question=query))#output


file_list=enter_unstructured_docs() #entering files-> FilesList
text=convert_files_to_text(file_list) #FileList->Text
# print(text)
split_text=splitter(text)   #Text-> Text Chunks
embed=embeddings()          #Embedding
db=VectorDB(split_text,embed)   #Storing in text Chunks Vector DB in embeddings format
chain=CreateChain()         #chain creation with chain_type="stuff"
print("Enter q to Quit")
while True:                 #while loop for Queries
    query=EnterQuery()
    if query=="q":
        break
    ChainRun(query,chain,db)#calling chain Run function
