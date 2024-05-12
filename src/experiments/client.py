import socket
import os
import time
def send_client():

    file_name = "train.sh"
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_ip = "192.168.2.201"
    server_port = 8000

    client.connect((server_ip, server_port))

    print(f"Connected to {server_ip}:{server_port}")

    file = open(file_name, "rb")
    file_size = os.path.getsize(file_name)
    client.send(file_name.encode())
    client.send(str(file_size).encode())

    data = file.read()
    time.sleep(1)
    client.sendall(data)

    file.close()
    client.close()
send_client()
