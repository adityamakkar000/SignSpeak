import socket
import os
import subprocess

def send_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_ip = "192.168.2.201"
    port = 8000

    #bind
    server.bind((server_ip, port))

    server.listen()
    print(f"Listening on {server_ip}:{port}")

    while True:
        client, addr = server.accept()

        file_name = client.recv(1024).decode()
        file_size = client.recv(1024).decode()
        data = client.recv(1024).decode()

        fname = "experiment.sh"
        file = open(fname, "w")
        file.write(data)

        file.close()

        subprocess.call(['sh', fname])

    client.close()

send_server()
