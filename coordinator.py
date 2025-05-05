import socket
import threading
import torch
import torch.nn as nn
import pickle
import struct
import numpy as np
import pandas as pd
import time

from crypto_utils import generate_ecc_keypair, derive_shared_key, aes_decrypt, aes_encrypt, serialize_key, deserialize_key

LOG_FILE = "training_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

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

def federated_averaging(models, data_sizes):
    global_model = models[0]
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)
    total_data = sum(data_sizes)
    for i, model in enumerate(models):
        for param, global_param in zip(model.parameters(), global_model.parameters()):
            global_param.data += param.data * (data_sizes[i] / total_data)
    return global_model

def evaluate_model(model, file_path):
    data = pd.read_csv(file_path)
    X_test = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(data['target'].values, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs = model(X_test)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
    log(f"Test Accuracy: {accuracy * 100:.2f}%")

class FederatedCoordinator:
    def __init__(self, host='localhost', port=9090, input_size=30, rounds=3):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clients = []
        self.models = []
        self.data_sizes = []
        self.global_model = HeartDiseaseModel(input_size)
        self.rounds = rounds
        self.ecc_private_key, self.ecc_public_key = generate_ecc_keypair()
        self.shared_keys = {}

    def start(self):
        try:
            open(LOG_FILE, "w").close()
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            log(f"Coordinator started on {self.host}:{self.port}. Waiting for clients...")

            for round_num in range(1, self.rounds + 1):
                log(f"\n=== Starting Round {round_num} ===")
                self.clients = []
                self.models = []
                self.data_sizes = []
                self.shared_keys = {}

                # Accept 3 clients
                while len(self.clients) < 3:
                    client_socket, addr = self.server_socket.accept()
                    log(f"Client connected from {addr}")
                    self.clients.append(client_socket)

                # Handle clients in threads
                threads = []
                for client_socket in self.clients:
                    thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to finish
                for t in threads:
                    t.join()

                log("Performing Federated Averaging...")
                agg_start = time.perf_counter()
                self.global_model = federated_averaging(self.models, self.data_sizes)
                agg_end = time.perf_counter()
                aggregation_time = agg_end - agg_start
                log(f"Aggregation time: {aggregation_time:.4f} seconds")

                log("Evaluating global model using test data...")
                evaluate_model(self.global_model, 'datasets/test_data.csv')

                self.broadcast_model()

            log("All rounds completed. Final model evaluation done.")
        except Exception as e:
            log(f"Error: {e}")
        finally:
            self.server_socket.close()

    def handle_client(self, client_socket):
        try:
            # Step 1: Receive client ECC public key
            key_size = struct.unpack('I', client_socket.recv(4))[0]
            client_pub_key_bytes = b''
            while len(client_pub_key_bytes) < key_size:
                packet = client_socket.recv(4096)
                if not packet:
                    log("Client disconnected during key exchange.")
                    return
                client_pub_key_bytes += packet
            client_pub_key = deserialize_key(client_pub_key_bytes)

            # Step 2: Send coordinator ECC public key back to client
            coord_pub_key_bytes = serialize_key(self.ecc_public_key)
            client_socket.send(struct.pack('I', len(coord_pub_key_bytes)))
            client_socket.sendall(coord_pub_key_bytes)

            # Step 3: Derive shared AES key
            shared_key = derive_shared_key(self.ecc_private_key, client_pub_key)
            self.shared_keys[client_socket] = shared_key

            # Step 4: Receive encrypted model
            data_size = struct.unpack('I', client_socket.recv(4))[0]
            encrypted_data = b''
            while len(encrypted_data) < data_size:
                packet = client_socket.recv(4096)
                if not packet:
                    log("Client disconnected unexpectedly during model transmission.")
                    return
                encrypted_data += packet

            # Step 5: Decrypt and load model
            decrypt_start = time.perf_counter()
            model_data = aes_decrypt(shared_key, encrypted_data)
            decrypt_end = time.perf_counter()
            decryption_time = decrypt_end - decrypt_start
            log(f"Decryption time for client: {decryption_time:.4f} seconds")

            model = pickle.loads(model_data)
            self.models.append(model)
            self.data_sizes.append(data_size)
            log(f"Model received and decrypted. Total models received: {len(self.models)}")

        except Exception as e:
            log(f"Error receiving client model: {e}")

    def broadcast_model(self):
        log("Broadcasting global model to clients...")
        for client_socket in self.clients:
            try:
                model_data = pickle.dumps(self.global_model)

                encrypt_start = time.perf_counter()
                encrypted_data = aes_encrypt(self.shared_keys[client_socket], model_data)
                encrypt_end = time.perf_counter()
                encryption_time = encrypt_end - encrypt_start

                data_size = struct.pack('I', len(encrypted_data))
                client_socket.send(data_size)
                client_socket.sendall(encrypted_data)
                log(f"Global model encrypted and sent to client. Encryption time: {encryption_time:.4f} seconds")
            except Exception as e:
                log(f"Error sending model to client: {e}")

if __name__ == '__main__':
    coordinator = FederatedCoordinator()
    coordinator.start()
