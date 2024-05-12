import socket
import os

def send_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_ip = "127.0.0.1"
    server_port = 8000

    client.connect((server_ip, server_port))

    file = open("test.sh", "rb")
    file_size = os.path.getsize("test.sh")
    client.send("transferred.sh".encode())
    client.send(str(file_size).encode())

    data = file.read()
    client.sendall(data)

    file.close()
    client.close()
send_client()
   