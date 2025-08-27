import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embed = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFDirectoryLoader("research_papers")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vector_db = FAISS.from_documents(documents,embed)
retriever = vector_db.as_retriever()

retriever_tool = Tool(
    name="PDF-Search",
    description="Useful for searching information about LLM and Attention Research Papers",
    func=lambda q: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(q)])
)


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=400)
wiki_wrapper  = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=400)

arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)
search = DuckDuckGoSearchRun(name = "Search")

st.title("Chat Bot Agent With Search")

#Side bar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key:",type='password')

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role":"assistant","content":"Hi, I am a chatbot, How can i help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools = [retriever_tool,arxiv,wiki,search]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
