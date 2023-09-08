import PyPDF2
# from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

import os
import chromadb


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

def enter_pdf():
    pdf_path =input("Enter the PDF File Name : ")
    pdf_text = pdf_to_string(pdf_path)
    # print(pdf_text)
    return pdf_text

def pdf_to_string(pdf_path):
    pdf_text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            pdf_text += page.extract_text()
    return pdf_text

def splitter(pdf_text):
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

def embeddings():
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    return embeddings

def VectorDB(texts,embeddings):
    print(type(texts))
    print(type(embeddings))
    document_search = Chroma.from_texts(texts, embeddings)
    return document_search

def CreateChain():
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain

def EnterQuery():
    query=input("Enter the input :")
    return query

def ChainRun(query,chain,document_search):
    docs = document_search.similarity_search(query)
    print(chain.run(input_documents=docs, question=query))



# pdf_path =input("Enter the File Name : ")
# pdf_text = pdf_to_string(pdf_path)
# print(pdf_text)

text=enter_pdf()
# print(text)
split_text=splitter(text)
embed=embeddings()
db=VectorDB(split_text,embed)
chain=CreateChain()
query=EnterQuery()
ChainRun(query,chain,db)
