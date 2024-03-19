import socket
import threading

# Define the address and port for the proxy server
LOCAL_HOST = '127.0.0.1'
LOCAL_PORT = 8888

# Define the address and port for the target server
TARGET_HOST = 'target_machine_ip_address'
TARGET_PORT = 9999

def handle_client(client_socket):
    while True:
        # Receive data from the client
        client_data = client_socket.recv(4096)
        if not client_data:
            break

        # Forward the received data to the target server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as target_socket:
            target_socket.connect((TARGET_HOST, TARGET_PORT))
            target_socket.sendall(client_data)

            # Receive data from the target server
            target_response = target_socket.recv(4096)
            if not target_response:
                break

            # Send the target server's response back to the client
            client_socket.sendall(target_response)

    # Close the client socket
    client_socket.close()

def main():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the server socket to the local address and port
    server_socket.bind((LOCAL_HOST, LOCAL_PORT))

    # Start listening for incoming connections
    server_socket.listen(5)
    print(f'[*] Listening on {LOCAL_HOST}:{LOCAL_PORT}')

    while True:
        # Accept incoming connections
        client_socket, client_address = server_socket.accept()
        print(f'[*] Accepted connection from {client_address[0]}:{client_address[1]}')

        # Create a new thread to handle the client
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == '__main__':
    # This script is being run directly, not being imported as a module
    main()
