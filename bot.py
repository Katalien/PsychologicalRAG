import os

from aiogram import Bot, Dispatcher, types, Router
from aiogram import F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from dotenv import load_dotenv

from model import PsychologistRAG

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

router = Router()
dp.include_router(router)

psychologist = PsychologistRAG("./faiss_index")


@router.message(CommandStart())
async def start(message: types.Message):
    welcome_text = (
        "Привет! 🤍\n\n"
        "Я — бот психологической поддержки.\n"
        "Вы можете задать любой вопрос, связанный с ментальным здоровьем, "
        "отношениями, тревогой, стрессом или эмоциональным выгоранием.\n\n"
        "Я не ставлю диагнозы и не заменяю психолога, "
        "но стараюсь поддержать и помочь разобраться.\n\n"
        "Нажмите кнопку ниже, чтобы начать диалог."
    )

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Начать диалог",
                    callback_data="start_dialog"
                )
            ]
        ]
    )

    await message.answer(welcome_text, reply_markup=keyboard)


@router.callback_query(F.data == "start_dialog")
async def start_dialog(callback: CallbackQuery):
    await callback.message.answer(
        "Я готов вас слушать.\n"
        "Напишите, что вас беспокоит"
    )
    await callback.answer()


@router.message()
async def handle_msg(message: types.Message):
    user_q = message.text

    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING
    )

    result = psychologist.ask(user_q)

    answer = (
        f"*{result['title']}*\n\n"
        f"{result['solution']}\n\n"
        f"Источник: {result['link']}"
    )

    await message.answer(answer, parse_mode="Markdown")


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
