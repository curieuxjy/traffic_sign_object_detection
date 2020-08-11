import socket

# socket 만들기
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.99.1", 9999)) 
#ip 주소, port 번호(사용중인 번호만 아니면 모든 int 가능)

# message 작성
test_msg = "abcd" # 판단 1
sock.send(test_msg.encode())

# message 받기
data_size = 1024
data = sock.recv(data_size)
# print(data.decode())

# 연결 종료
sock.close()