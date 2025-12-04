import os

import langchain
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")


class Psycho(BaseModel):
    title: str = Field(..., description="Описание проблемы")
    solution: str = Field(..., description="Решение проблемы")
    link: str = Field(..., description="Ссылка на статью")


parser = JsonOutputParser(pydantic_object=Psycho)
model_name = "google/embeddinggemma-300m"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


def prepare_data():
    df = pd.read_json("data/final_dataset.json", orient='rows')
    loader = DataFrameLoader(df, page_content_column='question')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    sample_vec = embeddings.embed_query("Hello, world!")
    db = FAISS.from_documents(texts, embeddings)
    db.as_retriever()
    db.save_local('faiss')


prepare_data()

def send_to_model(msg):
    db = FAISS.load_local("./faiss", embeddings, allow_dangerous_deserialization=True)  # открываем сохранённую бд

    prompt = PromptTemplate(
        template=(
            """
            Ты - русскоязычный бот психологической помощи. 
            Тебе необходимо предоставить пользователю решение его проблемы и ссылку на статью с решением в формате JSON {format_instructions}.
            Решение проблемы:\n{query}
            """
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = ChatMistralAI(
        model="open-mixtral-8x22b",
        api_key=api_key,
        temperature=0
    )

    chain = prompt | llm | parser

    similar_answer = db.similarity_search(msg)
    doc = similar_answer[0].dict()['metadata']

    result = chain.run(doc)
    # id = data['response'][0]['id']
    return result
