import streamlit as st
import pandas as pd
import openai
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os

# OpenAI API 키 설정
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in os.environ:
    pass
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# Streamlit 설정
st.set_page_config(page_title="CSV 데이터 분석", layout="wide")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터프레임 요약 정보:")
    st.write(df.describe())

    # 에이전트 생성
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=1, model="gpt-4o"),
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # 위험한 코드 실행을 허용
    )

    # 사용자가 입력한 질문 처리
    query = st.text_input("질문을 입력하세요:")
    if st.button("질문하기"):
        if query:
            try:
                response = agent.invoke(query, handle_parsing_errors=True)
                st.write("응답:")
                st.write(response)
            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.error("질문을 입력하세요.")



# import streamlit as st
# import pandas as pd
# from langchain.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.chat_models import ChatOpenAI
# from langchain.agents.agent_types import AgentType
# from langchain.llms import OpenAI
# import os
# import matplotlib.pyplot as plt

# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from io import StringIO
# import numpy as np

# # OpenAI API 키 설정
# if 'OPENAI_API_KEY' in st.secrets:
#     os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# elif 'OPENAI_API_KEY' in os.environ:
#     pass
# else:
#     st.error("OpenAI API 키가 설정되지 않았습니다.")
#     st.stop()

# # Streamlit 앱 설정
# st.title("Langchain Pandas DataFrame Agent")

# # CSV 파일 업로드
# uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

# if uploaded_file is not None:
#     # CSV 파일 읽기
#     df = pd.read_csv(uploaded_file)
#     st.write("데이터 미리보기:")
#     st.write(df.head())

#     # Pandas DataFrame Agent 생성
#     agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model='gpt-4o'), df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

#     # 사용자 질문 입력
#     user_question = st.text_input("데이터에 대해 질문하세요:")

#     if user_question:
#         try:
#             # Agent를 사용하여 질문에 답변
#             answer = agent.run(user_question)
            
#             # 결과 표시
#             st.write("답변:")
#             st.write(answer)

#             # 만약 결과가 matplotlib 그래프라면 표시
#             if plt.get_fignums():
#                 st.pyplot(plt.gcf())

#         except Exception as e:
#             st.error(f"오류가 발생했습니다: {str(e)}")

#     # 데이터 기본 정보 표시
#     st.subheader("데이터 기본 정보")
#     st.write(df.info())

#     # 데이터 통계 요약
#     st.subheader("데이터 통계 요약")
#     st.write(df.describe())

# else:
#     st.write("CSV 파일을 업로드해주세요.")

# # 디버그 정보 표시
# st.write("### Debug Information")
# st.write("Python version:", os.sys.version)

# import pkg_resources
# st.write("Installed Packages:")
# installed_packages = pkg_resources.working_set
# installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
# for package in installed_packages_list:
#     st.write(package)

# # import streamlit as st
# # import pandas as pd
# # import os
# # from io import StringIO
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.chat_models import ChatOpenAI
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory
# # from io import StringIO
# # import numpy as np

# # # OpenAI API 키 설정
# # if 'OPENAI_API_KEY' in st.secrets:
# #     os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# # elif 'OPENAI_API_KEY' in os.environ:
# #     pass
# # else:
# #     st.error("OpenAI API 키가 설정되지 않았습니다.")
# #     st.stop()



# # def analyze_csv(csv_data):
# #     # CSV 데이터를 DataFrame으로 변환
# #     df = pd.read_csv(StringIO(csv_data))
    
# #     # 기본 정보 추출
# #     info = {
# #         "columns": df.columns.tolist(),
# #         "shape": df.shape,
# #         "dtypes": df.dtypes.to_dict(),
# #         "summary": df.describe().to_dict(),
# #         "missing_values": df.isnull().sum().to_dict()
# #     }
    
# #     # 데이터 타입별 분석
# #     for column in df.columns:
# #         if df[column].dtype == 'object':
# #             info[f"{column}_unique_values"] = df[column].nunique()
# #             info[f"{column}_top_values"] = df[column].value_counts().head().to_dict()
# #         elif np.issubdtype(df[column].dtype, np.number):
# #             info[f"{column}_mean"] = df[column].mean()
# #             info[f"{column}_median"] = df[column].median()
    
# #     return info

# # # CSV 파일 업로드 함수
# # def upload_csv():
# #     uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
# #     if uploaded_file is not None:
# #         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
# #         return stringio.read()
# #     return None

# # # 데이터 전처리 및 RAG 모델 설정 함수
# # @st.cache_resource
# # # RAG 모델 설정 함수 수정
# # def setup_rag_model(csv_data):
# #     # CSV 분석 정보 추출
# #     csv_analysis = analyze_csv(csv_data)
    
# #     # 기존의 텍스트 처리 로직
# #     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# #     texts = text_splitter.split_text(csv_data)
    
# #     # CSV 분석 정보를 텍스트에 추가
# #     analysis_text = f"CSV Analysis:\n{str(csv_analysis)}\n\n"
# #     texts = [analysis_text] + texts

# #     # 임베딩 생성
# #     embeddings = OpenAIEmbeddings()

# #     try:
# #         # FAISS 벡터 저장소 생성
# #         vectorstore = FAISS.from_texts(texts, embeddings)
# #     except Exception as e:
# #         st.error(f"Error initializing FAISS: {str(e)}")
# #         st.stop()

# #     # RAG 모델 설정 (ChatOpenAI 사용)
# #     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# #     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# #     qa_chain = ConversationalRetrievalChain.from_llm(
# #         llm=llm,
# #         retriever=vectorstore.as_retriever(),
# #         memory=memory
# #     )
# #     return qa_chain

# # # Streamlit 앱 설정
# # st.title("RAG Chatbot with CSV Upload")

# # # 세션 상태 초기화
# # if 'qa_chain' not in st.session_state:
# #     st.session_state.qa_chain = None

# # if 'chat_history' not in st.session_state:
# #     st.session_state.chat_history = []

# # # CSV 파일 업로드
# # csv_data = upload_csv()

# # if csv_data and st.session_state.qa_chain is None:
# #     try:
# #         # CSV 데이터를 사용하여 RAG 모델 설정
# #         st.session_state.qa_chain = setup_rag_model(csv_data)
# #         st.success("RAG 모델이 성공적으로 설정되었습니다.")
# #     except Exception as e:
# #         st.error(f"An error occurred: {str(e)}")

# # if st.session_state.qa_chain:
# #     # 사용자 입력
# #     user_input = st.text_input("질문을 입력하세요:")

# #     if user_input:
# #         try:
# #             # 사용자 입력에 CSV 분석 요청 감지
# #             if "analyze" in user_input.lower() and "csv" in user_input.lower():
# #                 csv_analysis = analyze_csv(csv_data)
# #                 st.write("CSV 파일 분석 결과:", csv_analysis)
# #             else:
# #                 # 기존의 챗봇 응답 생성 로직
# #                 response = st.session_state.qa_chain({"question": user_input})
# #                 st.session_state.chat_history.append(("사용자", user_input))
# #                 st.session_state.chat_history.append(("챗봇", response['answer']))
                
# #                 # 대화 히스토리 표시
# #                 for role, text in st.session_state.chat_history:
# #                     if role == "사용자":
# #                         st.write(f"👤 사용자: {text}")
# #                     else:
# #                         st.write(f"🤖 챗봇: {text}")
# #         except Exception as e:
# #             st.error(f"An error occurred: {str(e)}")
# # else:
# #     st.write("CSV 파일을 업로드해주세요.")

# # # 디버그 정보 표시
# # st.write("### Debug Information")
# # st.write("Python version:", os.sys.version)

# # import pkg_resources
# # st.write("Installed Packages:")
# # installed_packages = pkg_resources.working_set
# # installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
# # for package in installed_packages_list:
# #     st.write(package)