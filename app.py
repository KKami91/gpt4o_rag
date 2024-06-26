import streamlit as st
import pandas as pd
import os
from io import StringIO
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# OpenAI API 키 설정
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in os.environ:
    pass
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

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

    try:
        # FAISS 벡터 저장소 생성
        vectorstore = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error initializing FAISS: {str(e)}")
        st.stop()

    # RAG 모델 설정 (ChatOpenAI 사용)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
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
    try:
        # CSV 데이터를 사용하여 RAG 모델 설정
        qa_chain = setup_rag_model(csv_data)

        # 사용자 입력
        user_input = st.text_input("질문을 입력하세요:")

        if user_input:
            # 챗봇 응답 생성
            response = qa_chain.run(user_input)
            st.write("챗봇:", response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("CSV 파일을 업로드해주세요.")

# 디버그 정보 표시
st.write("### Debug Information")
st.write("Python version:", os.sys.version)

import pkg_resources
st.write("Installed Packages:")
installed_packages = pkg_resources.working_set
installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
for package in installed_packages_list:
    st.write(package)