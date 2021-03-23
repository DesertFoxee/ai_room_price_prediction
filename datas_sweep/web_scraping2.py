import socket

HOST = 'https://batdongsan.com.vn/cho-thue-nha-tro-phong-tro-ha-noi/p2'  # Server hostname or IP address
PORT = 80        # Port

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
client_socket.connect(server_address)

request_header = b'GET / HTTP/1.0\r\nHost: https://batdongsan.com.vn\r\n\r\n'
client_socket.sendall(request_header)

response = ''
while True:
    recv = client_socket.recv(1024)
    if not recv:
        break
    response += str(recv)

print(response)
client_socket.close()