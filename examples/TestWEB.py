from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat</title>
</head>
<body>
    <h1>WebSocket Chat</h1>
    <form action="" onsubmit="sendMessage(event)">
        <input type="text" id="messageText" autocomplete="off" placeholder="Type your message here..." />
        <button>Send</button>
    </form>
    <ul id="messages">
    </ul>
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onmessage = (event) => {
            const messages = document.getElementById('messages');
            const message = document.createElement('li');
            message.textContent = event.data;
            messages.appendChild(message);
        };

        function sendMessage(event) {
            const input = document.getElementById("messageText");
            ws.send(input.value);
            input.value = '';
            event.preventDefault();
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(content=html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"You said: {data}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
