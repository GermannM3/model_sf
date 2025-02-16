import os
try:
    from dotenv import load_dotenv
except ImportError:
    # Если модуль не установлен, определим пустую функцию-заглушку.
    def load_dotenv():
        pass

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Загружаем переменные окружения из файла .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

def start(update: Update, context: CallbackContext):
    """Обработчик команды /start"""
    update.message.reply_text("Привет! Я бот, готов к общению.")

def help_command(update: Update, context: CallbackContext):
    """Обработчик команды /help"""
    update.message.reply_text("Напиши /start для начала общения.")

def main():
    load_dotenv()  # Попытка загрузить переменные окружения (если модуль установлен)
    print("Telegram Bot is running...")
    # Здесь реализуйте логику бота или просто зациклите процесс
    import time
    while True:
        time.sleep(10)

if __name__ == '__main__':
    main() 