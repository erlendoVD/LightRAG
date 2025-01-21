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

PROSJEKT = "Nye Rikshospitalet"
WORKING_DIR = f"./data RAG/{PROSJEKT}"

logging.basicConfig(level=logging.INFO)

#if not os.path.exists(WORKING_DIR):
#    os.mkdir(WORKING_DIR)

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
            background-color: #212121;
            color: #ececec;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #menu-bar {
            display: flex;
            justify-content: center;
            padding: 10px;
            background-color: #1E1E1E;
        }
        #left-menu {
            width: 200px;
            background-color: #1E1E1E;
            display: flex;
            flex-direction: column;
            padding: 10px;
        }
        #left-menu button {
            padding: 10px 20px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
            background-color: #380016;
            color: #ffa392;
            cursor: pointer;
            text-align: left;
        }
        #main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        #chat-container {
            flex: 1;
            display: flex;
        }
        #chat {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 20px;
            color: #ececec;
            scrollbar-width: thin;
            scrollbar-color: #616161 #212121;
        }
        #chat > div {
            max-width: 768px;
            margin: 0 auto;
            word-wrap: break-word;
        }
        #input {
            display: flex;
            justify-content: center;
            padding: 10px;
            background-color: #1F1F1F;
        }
        #input input {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            margin-right: 10px;
            max-width: 600px;
            width: 100%;
            color: white;
            background-color: #2E2E2E;
        }
        #input button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #380016;
            color: #ffa392;
            cursor: pointer;
        }
         .user-message {
            background-color: #2E2E2E;
            color: #FFFFFF;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .dropdown {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #2E2E2E;
            color: white;
            border: 1px solid #1E1E1E;
            border-radius: 5px;
            padding: 5px;
        }
        .dropdown select {
            background-color: #2E2E2E;
            color: white;
            border: none;
            outline: none;
            padding: 5px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div id="main-content">
        <div id="menu-bar">
            <div>Anbudsassistent</div>
        </div>
        <div class="dropdown">
            <select id="project-selector" onchange="refreshChatAndSetProject()">
            </select>
        </div>
        <div class="dropdown" style="top: 50px;">
            <select id="mode-selector" onchange="setMode()">
                <option value="naive">naive</option>
                <option value="local">local</option>
                <option value="global">global</option>
                <option value="hybrid" selected>hybrid</option>
                <option value="mix">mix</option>
            </select>
        </div>
        <div id="chat"></div>
        <div id="input">
            <input id="message" type="text" placeholder="Type your message..." onkeydown="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const messageInput = document.getElementById('message');
        const projectSelector = document.getElementById('project-selector');
        const modeSelector = document.getElementById('mode-selector');
        let selectedMode = modeSelector.value;

        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onmessage = (event) => {
            const message = document.createElement('div');
            message.innerHTML = `${marked.parse(event.data)}`;
            chat.appendChild(message);
            chat.scrollTop = chat.scrollHeight;
        };

        function sendMessage() {
            const message = messageInput.value;
            if (message) {
                const userMessage = document.createElement('div');
                userMessage.textContent = `(${selectedMode}) ${message}`;
                userMessage.className = 'user-message';
                chat.appendChild(userMessage);
                chat.scrollTop = chat.scrollHeight;
                ws.send(JSON.stringify({ message, mode: selectedMode }));
                messageInput.value = '';
            }
        }

        function refreshChatAndSetProject() {
            chat.innerHTML = ''; // Clear the chat window
            const selectedProject = document.getElementById('project-selector').value;
            fetch(`/api/set_project?project=${selectedProject}`);
            console.log("Chat refreshed for project: ", selectedProject);
        }

        function setMode() {
            selectedMode = modeSelector.value;
            fetch(`/api/set_mode?mode=${selectedMode}`);
            console.log("Mode set to: ", selectedMode);
        }

        async function loadFolders() {
            const response = await fetch('/api/folders');
            const folders = await response.json();
            projectSelector.innerHTML = '';
            folders.forEach(folder => {
                const option = document.createElement('option');
                option.value = folder;
                option.textContent = folder;
                projectSelector.appendChild(option);
            });
        }

        loadFolders();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return html

@app.get("/api/folders")
async def get_folders():
    base_path = "./data anbud"
    folders = [
        name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))
    ]
    return folders

@app.get("/api/set_project")
async def set_project(project: str):
    global PROSJEKT, WORKING_DIR, rag
    PROSJEKT = project
    WORKING_DIR = f"./data RAG/{PROSJEKT}"
    
    # Re-initialize the rag instance with the new working directory
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    return {"message": f"Project set to {PROSJEKT}"}

@app.get("/api/set_mode")
async def set_mode(mode: str):
    global rag
    # Re-initialize the rag instance with the new mode
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    return {"message": f"Mode set to {mode}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            message = data_json["message"]
            mode = data_json["mode"]
            response = rag.query(message, param=QueryParam(mode=mode))
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
