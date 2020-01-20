# coding: utf-8
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
#print(type(test_data),len(test_data),type(test_data[0]),'\n',)
#test_data1 = list(test_data[0][0])


'''
with open('test.txt','a+',encoding='utf-8') as f:
    n = 1
    for i in test_data1:
        if n%28 == 0:
            f.write(str(i) + '\n')
        else:
            if len(str(i)) == 4:
                f.write(str(i))
            else:
                f.write('[' + str(i)[2:4] + ']')
        n += 1
'''      

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
