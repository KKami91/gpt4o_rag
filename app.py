import streamlit as st
import pandas as pd
import os
from io import StringIO
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# OpenAI API 키 설정
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in os.environ:
    pass
else:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

# CSV 파일 업로드 함수
def upload_csv():
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    return None

# 데이터 전처리 및 RAG 모델 설정 함수
@st.cache_resource
def setup_rag_model(csv_data):
    # 텍스트 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(csv_data)

    # 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # 임시 디렉토리 생성
    persist_directory = os.path.join(os.getcwd(), 'chroma_db')
    os.makedirs(persist_directory, exist_ok=True)

    # Chroma 벡터 저장소 생성
    vectorstore = Chroma.from_texts(texts, embedding=embeddings, persist_directory=persist_directory)

    # RAG 모델 설정
    llm = OpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# Streamlit 앱 설정
st.title("RAG Chatbot with CSV Upload")

# CSV 파일 업로드
csv_data = upload_csv()

if csv_data:
    # CSV 데이터를 사용하여 RAG 모델 설정
    qa_chain = setup_rag_model(csv_data)

    # 사용자 입력
    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 챗봇 응답 생성
        response = qa_chain.run(user_input)
        st.write("챗봇:", response)
else:
    st.write("CSV 파일을 업로드해주세요.")