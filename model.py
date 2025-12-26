import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)


class PsychoResponse(BaseModel):
    title: str = Field(..., description="Описание проблемы")
    solution: str = Field(..., description="Подробное решение проблемы")
    link: str = Field(..., description="Ссылка на источник")


class PsychologistRAG:
    def __init__(self, faiss_path: str = "./faiss_index"):
        self.faiss_path = faiss_path
        self.db = None
        self.chain = None
        self._initialize_system()

    def _initialize_system(self):
        """Инициализация FAISS и цепочки"""
        if os.path.exists(self.faiss_path):
            try:
                self.db = FAISS.load_local(
                    self.faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print("Ошибка загрузки индекса:", e)
                self.db = None
        else:
            print("FAISS индекс не найден, требуется векторизация.")
            self.vectorize_dataset()
            print("FAISS векторизован")
            self.db = FAISS.load_local(
                self.faiss_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

        self._initialize_chain()

    def _initialize_chain(self):
        """Создаёт цепочку RAG → LLM → JSON"""

        parser = JsonOutputParser(pydantic_object=PsychoResponse)

        template = """
        Ты - профессиональный русскоязычный психолог-консультант.
        Твоя задача: дать эмпатичный, структурированный ответ, опираясь ТОЛЬКО на предоставленный контекст. Всегда начинай сообщения со слов поддержки.
        Если вопрос не касается психологии или ментального здоровья, вежливо откажись отвечать, вместо ссылки напиши 'Не найдено'.
        Контекст: {context}
        Метаданные: {metadata}
        Вопрос пользователя: {question}
        
        Ответ строго в формате JSON: {format_instructions}
        
        Правила:
        1. Не придумывай данные, если их нет в контексте.
        2. Если данных недостаточно — так и скажи.
        3. Не давай вредных советов, не ставь диагнозы.
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "metadata", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=MISTRAL_API_KEY,
            temperature=0.2
        )

        # Финальная цепочка (RunnableMap → Prompt → LLM → JSON)
        self.chain = (
                RunnableMap({
                    "context": lambda x: x["context"],
                    "metadata": lambda x: x["metadata"],
                    "question": RunnablePassthrough()
                })
                | prompt
                | llm
                | parser
        )

    def vectorize_dataset(self, json_path="data/final_dataset.json"):
        print("Загрузка датасета...")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data["rows"])

        loader = DataFrameLoader(df, page_content_column="text")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128
        )
        chunks = splitter.split_documents(docs)

        print("Создание FAISS...")
        self.db = FAISS.from_documents(chunks, embeddings)

        print("Сохранение...")
        self.db.save_local(self.faiss_path)

        print("Готово.")
        return True

    def ask(self, question: str, k: int = 3):
        if self.db is None:
            return {
                "title": "Ошибка",
                "solution": "Индекс не загружен. Выполните векторизацию.",
                "link": ""
            }

        try:
            docs = self.db.similarity_search(question, k=k)

            if not docs:
                return {
                    "title": "Нет данных",
                    "solution": "Не удалось найти информацию. Пожалуйста, переформулируйте вопрос.",
                    "link": ""
                }

            context = "\n\n".join(d.page_content for d in docs)

            metadata = "\n".join(
                f"{d.metadata.get('name', 'Источник')}: {d.metadata.get('link', '')}"
                for d in docs
            )

            result = self.chain.invoke({
                "context": context,
                "metadata": metadata,
                "question": question
            })

            # Подстановка ссылки если LLM её не указал
            if not result.get("link"):
                result["link"] = docs[0].metadata.get("link", "")

            return result

        except Exception as e:
            return {
                "title": "Ошибка",
                "solution": f"Не удалось обработать запрос: {e}",
                "link": ""
            }
