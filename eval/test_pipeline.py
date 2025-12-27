from model import *

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


def main():
    rag_system = PsychologistRAG(faiss_path="../faiss_index")

    if rag_system.db is None:
        print("векторизация датасете")
        try:
            rag_system.vectorize_dataset("data/final_dataset.json")
        except Exception as e:
            print(f"ошибка при векторизации: {e}")
            return

    print("Введите ваш вопрос. Для выхода напишите «stop».\n")

    while True:
        question = input("Вопрос: ").strip()

        if question.lower() in {"stop", "exit"}:
            break

        print("генерация ответа...")
        answer = rag_system.ask(question)

        print("Ответ:")
        print(f"Заголовок: {answer.get('title', 'Нет заголовка')}")
        print(f"\nРешение:\n{answer.get('solution', 'Нет решения')}")
        link = answer.get('link', '').strip()
        print(f"Ссылка: {link if link else '—'}")


if __name__ == "__main__":
    main()
