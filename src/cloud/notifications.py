from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from dataclasses import dataclass

@dataclass
class NotificationConfig:
    email_enabled: bool = False
    slack_enabled: bool = False
    telegram_enabled: bool = False
    discord_enabled: bool = False
    email_settings: Optional[dict] = None
    slack_webhook: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = None

class NotificationManager:
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_notification(self, title: str, message: str, importance: str = "info"):
        """Отправляет уведомление через все настроенные каналы"""
        if self.config.email_enabled:
            await self.send_email(title, message)
            
        if self.config.slack_enabled:
            await self.send_slack(title, message, importance)
            
        if self.config.telegram_enabled:
            await self.send_telegram(title, message)
            
        if self.config.discord_enabled:
            await self.send_discord(title, message, importance)
            
    async def send_email(self, subject: str, body: str):
        """Отправляет email-уведомление"""
        if not self.config.email_settings:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.config.email_settings['from']
        msg['To'] = self.config.email_settings['to']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(self.config.email_settings['smtp_server'])
        server.starttls()
        server.login(
            self.config.email_settings['username'],
            self.config.email_settings['password']
        )
        server.send_message(msg)
        server.quit()
        
    async def send_slack(self, title: str, message: str, importance: str):
        """Отправляет уведомление в Slack"""
        if not self.config.slack_webhook:
            return
            
        color = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000"
        }.get(importance, "#36a64f")
        
        payload = {
            "attachments": [{
                "fallback": title,
                "color": color,
                "title": title,
                "text": message,
                "footer": "Kenga AI Notification System"
            }]
        }
        
        requests.post(self.config.slack_webhook, json=payload)
        
    async def send_telegram(self, title: str, message: str):
        """Отправляет уведомление в Telegram"""
        if not (self.config.telegram_token and self.config.telegram_chat_id):
            return
            
        text = f"*{title}*\n\n{message}"
        url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        requests.post(url, json=payload)
        
    async def send_discord(self, title: str, message: str, importance: str):
        """Отправляет уведомление в Discord"""
        if not self.config.discord_webhook:
            return
            
        color = {
            "info": 3066993,
            "warning": 16776960,
            "error": 15158332
        }.get(importance, 3066993)
        
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "footer": {
                    "text": "Kenga AI Notification System"
                }
            }]
        }
        
        requests.post(self.config.discord_webhook, json=payload) 