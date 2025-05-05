import socketio
import eventlet

# Create a Socket.IO server instance
sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
def client_connected(data):
    print(f"Client {data['clientId']} connected")
    sio.emit('clientConnected', data['clientId'])

@sio.event
def round_complete(data):
    print(f"Round {data['round']} completed with accuracy: {data['accuracy']}%")
    sio.emit('roundComplete', data)

if __name__ == '__main__':
    print("Socket.IO server running on http://localhost:9091")
    eventlet.wsgi.server(eventlet.listen(('localhost', 9091)), app)

