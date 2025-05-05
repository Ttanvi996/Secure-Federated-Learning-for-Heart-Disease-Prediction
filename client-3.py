import socket
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import struct
import sys
import traceback
import time

from crypto_utils import generate_ecc_keypair, derive_shared_key, aes_encrypt, aes_decrypt, serialize_key, deserialize_key

class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size=30, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def train_model(model, X, y, epochs=500, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

class FederatedClient:
    def __init__(self, server_host='localhost', server_port=9090):
        self.server_host = server_host
        self.server_port = server_port
        self.file_path = 'datasets/client_3_data.csv'
        self.model = HeartDiseaseModel()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.private_key, self.public_key = generate_ecc_keypair()
        self.shared_key = None

    def connect_to_server(self):
        try:
            self.socket.connect((self.server_host, self.server_port))
            print("✅ Client connected to server.")

            # Step 1: Send client ECC public key
            pub_key_bytes = serialize_key(self.public_key)
            self.socket.send(struct.pack('I', len(pub_key_bytes)))
            self.socket.sendall(pub_key_bytes)

            # Step 2: Receive coordinator ECC public key
            key_len = struct.unpack('I', self.socket.recv(4))[0]
            coordinator_pub_key_bytes = b''
            while len(coordinator_pub_key_bytes) < key_len:
                packet = self.socket.recv(4096)
                if not packet:
                    print("❌ Disconnected while receiving coordinator's public key.")
                    return
                coordinator_pub_key_bytes += packet

            coordinator_pub_key = deserialize_key(coordinator_pub_key_bytes)
            self.shared_key = derive_shared_key(self.private_key, coordinator_pub_key)

            print("🔐 ECC key exchange complete. AES shared key derived.")
        except ConnectionRefusedError:
            print("❌ Error: Could not connect to server.")
            sys.exit(1)

    def train_and_send_model(self):
        X, y = load_data(self.file_path)

        # Local training time
        train_start = time.perf_counter()
        train_model(self.model, X, y)
        train_end = time.perf_counter()
        training_time = train_end - train_start
        print(f"🧠 Local training time: {training_time:.4f} seconds")

        model_data = pickle.dumps(self.model)

        # Encryption time
        encrypt_start = time.perf_counter()
        encrypted_data = aes_encrypt(self.shared_key, model_data)
        encrypt_end = time.perf_counter()
        encryption_time = encrypt_end - encrypt_start
        print(f"🔒 Model encryption time: {encryption_time:.4f} seconds")

        # Send to server
        data_size = struct.pack('I', len(encrypted_data))
        self.socket.send(data_size)
        self.socket.sendall(encrypted_data)
        print("📤 Encrypted model sent to server.")

    def receive_global_model(self):
        try:
            data_size = struct.unpack('I', self.socket.recv(4))[0]
            encrypted_data = b''
            while len(encrypted_data) < data_size:
                packet = self.socket.recv(4096)
                if not packet:
                    print("❌ Disconnected while receiving global model.")
                    return
                encrypted_data += packet

            # Decryption time
            decrypt_start = time.perf_counter()
            decrypted_model_data = aes_decrypt(self.shared_key, encrypted_data)
            decrypt_end = time.perf_counter()
            decryption_time = decrypt_end - decrypt_start
            print(f"🔓 Global model decryption time: {decryption_time:.4f} seconds")

            global_model = pickle.loads(decrypted_model_data)
            self.model.load_state_dict(global_model.state_dict())
            print("✅ Client received and loaded global model.")
        except Exception as e:
            print(f"❌ Error receiving global model: {e}")

if __name__ == '__main__':
    try:
        client = FederatedClient()
        client.connect_to_server()
        client.train_and_send_model()
        client.receive_global_model()
    except Exception:
        traceback.print_exc()
        sys.exit(1)