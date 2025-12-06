import os
import json
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
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
        """Инициализирует FAISS индекс и цепочку"""
        if os.path.exists(self.faiss_path):
            print(f"Загрузка FAISS индекса из {self.faiss_path}")
            try:
                self.db = FAISS.load_local(
                    self.faiss_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("Индекс успешно загружен")
            except Exception as e:
                print(f"Ошибка загрузки индекса: {e}")
                print("Пожалуйста, сначала выполните векторизацию датасета")
                self.db = None
        else:
            print(f"FAISS индекс не найден в {self.faiss_path}")
            print("Пожалуйста, сначала выполните векторизацию датасета")
            self.db = None
        
        self._initialize_chain()
    
    def _initialize_chain(self):
        parser = JsonOutputParser(pydantic_object=PsychoResponse)
        
        prompt_template = """
        Ты - русскоязычный бот психологической помощи. Отвечай профессионально, но с сочувствием.
        
        Контекст из статей:
        {context}
        
        Метаданные статей:
        {metadata}
        
        Вопрос пользователя: {question}
        
        Основываясь на предоставленной информации, ответь в следующем формате:
        {format_instructions}
        
        Важные правила:
        1. Если в контексте нет информации для ответа, честно скажи об этом
        2. Не придумывай информацию, которой нет в контексте
        3. Будь максимально полезным и конкретным, ты работаешь с людьми у которых могут быть психологические проблемы. Главное парвило - не навреди и не советую ничего лишнего.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "metadata", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        llm = ChatMistralAI(
            model="open-mixtral-8x22b",
            api_key=api_key,
            temperature=0.3
        )
        
        self.chain = (
            {
                "context": lambda x: x["context"],
                "metadata": lambda x: x["metadata"],
                "question": RunnablePassthrough()
            } | prompt | llm | parser
        )
    
    def vectorize_dataset(self, json_path: str = "data/final_dataset.json"):
        print(f"Чтение датасета из {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['rows'])
        print(f"Загружено {len(df)} статей")
        
        loader = DataFrameLoader(df, page_content_column='text')
        documents = loader.load()
        
        # Разделение текста на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        print(f"Текст разделен на {len(texts)} чанков")
        
        print("Создание FAISS индекса...")
        self.db = FAISS.from_documents(texts, embeddings)
        
        print(f"Сохранение индекса в {self.faiss_path}...")
        self.db.save_local(self.faiss_path)
        
        print("Векторизация завершена успешно!")
        return True
    
    def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        if self.db is None:
            return {
                "title": "Система не инициализирована",
                "solution": "Пожалуйста, сначала выполните векторизацию датасета",
                "link": ""
            }
        
        try:
            docs = self.db.similarity_search(question, k=k)
            
            if not docs:
                return {
                    "title": "Информация не найдена",
                    "solution": "К сожалению, в базе знаний нет информации по вашему вопросу. Попробуйте переформулировать вопрос или обратитесь к специалисту.",
                    "link": ""
                }
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            metadata_list = []
            for doc in docs:
                meta = doc.metadata
                meta_str = f"Источник: {meta.get('name', 'Неизвестно')}"
                if 'category' in meta:
                    meta_str += f" | Категория: {meta['category']}"
                if 'date' in meta:
                    meta_str += f" | Дата: {meta['date']}"
                if 'link' in meta:
                    meta_str += f" | Ссылка: {meta['link']}"
                metadata_list.append(meta_str)
            
            metadata = "\n".join(metadata_list)
            
            result = self.chain.invoke({
                "context": context,
                "metadata": metadata,
                "question": question
            })
            
            # ссылка только из наиболее релевантного
            if 'link' not in result or not result['link']:
                result['link'] = docs[0].metadata.get('link', '')
            
            return result
            
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
            return {
                "title": "Ошибка обработки",
                "solution": f"Произошла ошибка при обработке вашего запроса: {str(e)}",
                "link": ""
            }



