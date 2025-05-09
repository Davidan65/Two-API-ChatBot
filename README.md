# Telegram Bot with DeepSeek API Integration

This is a Telegram bot that uses the DeepSeek API to generate responses to user messages.

## Setup

1. Make sure you have Python 3.7 or higher installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Bot

1. Make sure you have the following environment variables set:
   - `TOGETHER_API_KEY`: Your Together API key
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token

2. Run the bot:
   ```bash
   python bot.py
   ```

## Features

- Responds to /start and /help commands
- Uses DeepSeek API to generate responses to user messages
- Handles errors gracefully

## Usage

1. Start a chat with your bot on Telegram
2. Send /start to begin
3. Type any message and the bot will respond using the DeepSeek API 