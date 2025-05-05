from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.concatkdf import ConcatKDFHash
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def generate_ecc_keypair():
    private_key = ec.generate_private_key(ec.SECP384R1())
    public_key = private_key.public_key()
    return private_key, public_key

def serialize_key(key):
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

def deserialize_key(key_bytes):
    return serialization.load_pem_public_key(key_bytes)

def derive_shared_key(private_key, peer_public_key):
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
    concat_kdf = ConcatKDFHash(
        algorithm=hashes.SHA256(),
        length=32,
        otherinfo=None,
    )
    return concat_kdf.derive(shared_secret)

def aes_encrypt(key, data):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct

def aes_decrypt(key, encrypted_data):
    nonce = encrypted_data[:12]
    ct = encrypted_data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None)
