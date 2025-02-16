from setuptools import setup, find_packages

setup(
    name="kengapy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "python-telegram-bot>=20.6",
        "torch>=2.1.1",
        "aiohttp>=3.9.1",
        "asyncio>=3.4.3",
        "websockets>=12.0",
        "beautifulsoup4>=4.12.2",
        "python-multipart>=0.0.6",
        "uvicorn>=0.24.0",
    ],
    author="Kenga Team",
    author_email="team@kengarust.dev",
    description="Python based AI assistant with web and telegram interfaces",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kengarust/kengapy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
) 