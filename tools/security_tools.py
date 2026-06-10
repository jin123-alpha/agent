import hashlib
import secrets
import socket
import time


def generate_runtime_secret(length: int = 32) -> str:
    """
    generate_runtime_secret(length)：生成一个运行时随机密钥。
    """
    token = secrets.token_hex(length)
    payload = f"{token}-{time.time_ns()}-{socket.gethostname()}"

    return hashlib.sha256(payload.encode()).hexdigest()
