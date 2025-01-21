from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
import logging
import asyncio
import inspect
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from openai import AzureOpenAI
import numpy as np
import nest_asyncio

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

WORKING_DIR = "./datatest"
logging.basicConfig(level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content

async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)

embedding_dimension = 3072

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

#with open("./data/Anbud1.txt", "r", encoding="utf-8") as book1, open(
#    "./data/Anbud2.txt", "r", encoding="utf-8"
#) as book2:
#    rag.insert([book1.read(), book2.read()])

nest_asyncio.apply()

app = FastAPI(title="LightRAG", description="LightRAG API with Chat UI")

html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightRAG Chat</title>
    <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        #input {
            display: flex;
            padding: 10px;
            background-color: #1E1E1E;
        }
        #input input {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            margin-right: 10px;
        }
        #input button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #6200EE;
            color: white;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div id="chat"></div>
    <div id="input">
        <input id="message" type="text" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const messageInput = document.getElementById('message');

        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onmessage = (event) => {
            const message = document.createElement('div');
            message.textContent = `Assistant: ${event.data}`;
            chat.appendChild(message);
            chat.scrollTop = chat.scrollHeight;
        };

        function sendMessage() {
            const message = messageInput.value;
            if (message) {
                const userMessage = document.createElement('div');
                userMessage.textContent = `You: ${message}`;
                chat.appendChild(userMessage);
                chat.scrollTop = chat.scrollHeight;
                ws.send(message);
                messageInput.value = '';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return html

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            response = rag.query(message, param=QueryParam(mode="hybrid"))
            if isinstance(response, str):
                await websocket.send_text(response)
            else:
                async for chunk in response:
                    await websocket.send_text(chunk)
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8020)
