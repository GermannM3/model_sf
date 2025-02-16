# KengaPy API Documentation

## Endpoints

### Status
```
GET /status
```
Returns the current status of the system.

**Response:**
```json
{
    "status": "ok",
    "version": "0.1.0"
}
```

### Ask
```
POST /api/ask
```
Send a query to the AI assistant.

**Request Body:**
```json
{
    "query": "What is the weather today?"
}
```

**Response:**
```json
{
    "response": "Today will be sunny with a high of 25°C"
}
```

### WebSocket
```
WS /ws
```
Establish a WebSocket connection for real-time communication.

**Example:**
```python
import websockets
async with websockets.connect("ws://localhost:3000/ws") as websocket:
    await websocket.send("Hello")
    response = await websocket.recv()
``` 