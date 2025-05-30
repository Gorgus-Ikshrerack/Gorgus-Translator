import socket
import pickle


class Network:
    def __init__(self, server_ip: str):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server = server_ip
        self.port = 5555
        self.addr = (self.server, self.port)
        self.player = self.connect()

    def get_player(self):
        return self.player

    def connect(self):
        """When we connect to something we want to send back a piece of information to the thing that connected to us.
        """

        try:
            self.client.connect(self.addr)
            return pickle.loads(self.client.recv(2048))
        except (socket.error, EOFError) as e:
            return str(e)

    def send(self, data):
        try:
            self.client.send(pickle.dumps(data))
            return pickle.loads(self.client.recv(2048))
        except (socket.error, EOFError) as e:
            return str(e)