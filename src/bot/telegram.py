import os
from typing import Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.neural.processors.nlp import recognize_intent, analyze_sentiment

class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN не задан")
            
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text('Привет! Я ваш AI ассистент.')
        
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if text := update.message.text:
            intent = recognize_intent(text)
            sentiment = analyze_sentiment(text)
            
            response = {
                "Погода": "Сегодня солнечно и тепло!",
                "Новости": "В мире ничего нового.",
            }.get(intent, f"Тональность: {sentiment}. Спасибо за общение!")
            
            await update.message.reply_text(response)
            
    async def run(self):
        app = Application.builder().token(self.token).build()
        
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        await app.initialize()
        await app.start()
        await app.run_polling() 