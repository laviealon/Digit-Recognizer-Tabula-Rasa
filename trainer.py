"""The following file trains a neural network using the training data in the
folder <training_data> and tests its accuracy. Note that <training_data> must
be a folder with the structure specified in this library's README."""
from os import fsencode, listdir
from network import NeuralNetwork, int_to_results, results_to_int
from data_manager import DataPair
from image_transformation import convert


def network_train(ntwrk: NeuralNetwork, foldername: str) -> None:
    training_data = []
    for folder in listdir(foldername):
        if folder == '.DS_Store':
            continue
        for img in listdir(f'{foldername}/{folder}'):
            img_data = convert(f'{foldername}/{folder}/{img}')
            results_data = int_to_results(int(folder))
            dp = DataPair(img_data, results_data)
            training_data.append(dp)
    ntwrk.train_network(training_data)


def network_test(ntwrk: NeuralNetwork, foldername: str) -> float:
    test_data = []
    for folder in listdir(foldername):
        if folder == '.DS_Store':
            continue
        for img in listdir(f'{foldername}/{folder}'):
            img_data = convert(f'{foldername}/{folder}/{img}')
            results_data = int_to_results(int(folder))
            dp = DataPair(img_data, results_data)
            test_data.append(dp)
    return ntwrk.network_test(test_data)


if __name__ == '__main__':
    n = NeuralNetwork(784, 2, 16, 10)
    network_train(n, 'training_data/trainingSample/trainingSample')
    print(network_test(n, 'training_data/testSample'))
