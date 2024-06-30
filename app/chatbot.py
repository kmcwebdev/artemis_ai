from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_chain = self.setup_rag_chain()

    def setup_rag_chain(self):
        loader = PyPDFLoader("./data/Technology Services FAQs.pdf")
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def get_response(self, user_input):
        try:
            response = self.rag_chain.invoke(user_input)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))