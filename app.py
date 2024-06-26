import streamlit as st
import pandas as pd
from io import StringIO
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# CSV 파일 업로드 함수
def upload_csv():
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    if uploaded_file is not None:
        # StringIO 객체를 사용하여 파일 내용을 문자열로 읽기
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # 파일 내용을 문자열로 반환
        return stringio.read()
    return None

# 데이터 전처리 및 RAG 모델 설정 함수
@st.cache_resource
def setup_rag_model(data):
    # 텍스트 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(data)

    # 임베딩 생성 및 벡터 저장소 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts, embeddings)

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