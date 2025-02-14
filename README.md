# KengaPy

Python-based AI assistant with web and telegram interfaces.

## Features

- 🤖 AI-powered chat interface
- 🌐 Web API
- 📱 Telegram bot integration
- 🧠 Autonomous learning capabilities
- 🔒 Secure P2P communication

## Installation

```bash
# Using poetry
poetry install

# Using pip
pip install -r requirements.txt
```

## Usage

1. Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN=your_token_here
```

2. Run the application:
```bash
python -m src.main
```

## Development

```bash
# Run tests
pytest

# Run linter
flake8

# Run type checker
mypy .
```

## Docker

```bash
docker-compose up -d
```

## License

MIT License - see LICENSE file for details. 