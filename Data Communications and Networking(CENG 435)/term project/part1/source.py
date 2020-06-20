from socket import *
import sys
import random
import time,os,stat
from threading import Thread
from datetime import datetime

# create TCP socket
#sock = socket.socket(AF_INET, SOCK_STREAM)

def client(ipv4, port):
    # send n random request
    # the connection is kept alive until client closes it.
    _client = socket(AF_INET, SOCK_STREAM)
    _client.connect((ipv4, port))

    _client.send(time.ctime(time.time()).encode())
    reply = _client.recv(1024)
    print(_client.getsockname(), reply)

    _client.close()
        
def server(port):
    _server = socket(AF_INET, SOCK_STREAM)
    _server.bind(('',port))
    _server.listen(3)    # 1 is queue size for "not yet accept()'ed connections"
    try:
        #while True:
        for i in range(3):    # just limit # of accepts for Thread to exit
            ns, peer = _server.accept()
            print(peer, "connected")

            req = ns.recv(1000)
            while req and req != '':
                ns.send(req.decode().encode())
                req = ns.recv(1000)
            # now main thread ready to accept next connection
    finally:
        _server.close()
        
    

server = Thread(target=server, args=(20445,))
server.start()
# create 5 clients
client = Thread(target = client, args=('10.10.1.2', 20446)) # R1  [Thread(target = client, args=(26011)) for i in range(5)]
# start clients
client.start() #for cl in clients: cl.start()