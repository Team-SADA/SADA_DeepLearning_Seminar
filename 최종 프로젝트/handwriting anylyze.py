# import cupy as cp
import numpy as cp
import imageio
import matplotlib.pyplot as plt
import time


class NeuralNetwork:
    # 신경망 초기화 하기
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.i_nodes = inputNodes
        self.h_nodes = hiddenNodes
        self.o_nodes = outputNodes
        # 학습률
        self.lr = learningRate
        # 가중치 설정
        self.wih = cp.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = cp.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        # 활성화 함수로 sigmoid 함수 사용
        self.activation_function = lambda x: 1 / (1 + cp.exp(-x))

    def train(self, inputs_list, target_list):
        # input 리스트와 output 리스트 벡터화
        train_inputs = cp.array(inputs_list, ndmin=2).T
        train_targets = cp.array(target_list, ndmin=2).T
        # 은닉계층으로 들어오는 신호 계산
        hidden_inputs = cp.dot(self.wih, train_inputs)
        # 은닉계층에서 나가는 신호 계산(활성화 함수 적용)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력계층으로 들어오는 신호 계산
        final_inputs = cp.dot(self.who, hidden_outputs)
        # 최종 출력계층에서 나가는 신호 계산(활성화 함수 적용)
        final_outputs = self.activation_function(final_inputs)
        # 오차 계산
        outputs_errors = train_targets - final_outputs
        hidden_errors = cp.dot(self.who.T, outputs_errors)
        # 가중치 업데이트
        self.who += self.lr * cp.dot((outputs_errors * final_outputs * (1.0 - final_outputs)),
                                     cp.transpose(hidden_outputs))
        self.wih += self.lr * cp.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     cp.transpose(train_inputs))

        pass

    def query(self, inputs_list):
        inputs_query = cp.array(inputs_list, ndmin=2).T

        hidden_inputs = cp.dot(self.wih, inputs_query)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = cp.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
print("training...\n")
for e in range(epochs):
    start = time.perf_counter()
    if e == 0:
        print("First Epoch...")
    elif e == 1:
        print("Second Epoch...")
    elif e == 2:
        print("Third Epoch...")
    elif e == 3:
        print("Fourth Epoch...")
    else:
        print("Last Epoch...")
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (cp.asarray(all_values[1:], dtype=cp.float32) / 255.0 * 0.99) + 0.01
        targets = cp.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    print(f"spent time: {time.perf_counter() - start}sec")
    pass
print("Finished Training!\n")
done = False
while not done:
    a = input("Please type file name: ")
    img_file = imageio.v2.imread("handwriting/" + a, as_gray=True)
    img_data = 255.0 - img_file.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    img_array = cp.asarray(img_data, dtype=cp.float32).reshape(28, 28)
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()
    outputs = n.query(img_data)
    print(outputs)
    label = cp.argmax(outputs)
    print(label)
    q = input("Do you want to quit this program?(y/n)\n")
    if q == 'y' or q == 'Y':
        done = True
    else:
        continue
