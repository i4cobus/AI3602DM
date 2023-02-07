import Parameter
import itertools
from collections import defaultdict
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


def LoadRatingDataset(datafile):
    """
    :param name: the name of dataset like ml-1m, ml-100k
    :return: nested dictionary in forms of {user_id:{movie_id:score}}
    """
    ratings = {}
    with open(datafile) as f:
        for line in itertools.islice(f, 0, None):
            user, movie, rate = line.strip('\r\n').split(Parameter.seperator)[:3]
            ratings.setdefault(user, {})
            ratings[user][movie] = int(rate)
    return ratings


def LoadRatingDataset_mat(datafile):
    """
    :param name: the name of dataset like ml-1m, ml-100k
    :return: nested dictionary in forms of {user_id:{movie_id:score}}
    """
    ratings = {}
    with open(datafile) as f:
        for line in itertools.islice(f, 0, None):
            user, movie, rate = line.strip('\r\n').split(Parameter.seperator)[:3]
            ratings.setdefault(user, {})
            ratings[user][movie] = int(rate)

    like_matrix = [[] for _ in range(6040)]
    for user in ratings.keys():
        for film, score in ratings[user].items():
            like_matrix[int(user) - 1].append(int(film) - 1)

    tmp = []
    for user in ratings.keys():
        tmp.append(len(like_matrix[int(user) - 1]))

    return tmp


def LoadMovieDataset(name='ml-1m'):
    ratings = {}
    with open(Parameter.movies_path, encoding='gbk', errors='ignore') as f:
        for line in itertools.islice(f, 0, None):
            movie_id, movie_name = line.strip('\r\n').split(Parameter.seperator)[:2]
            ratings[movie_id] = movie_name
    return ratings


class DivideData(object):

    def __init__(self):
        self.trainset = []
        self.testset = []

    def generate_data_set(self, filename, pivot=0.8):
        a, b = 0, 0
        for line in self.loadfile(filename):
            if random.random() < pivot:
                self.trainset.append(line)
                a += 1
            else:
                self.testset.append(line)
                b += 1

        self.export_file('dataset/train_set.dat', self.trainset)
        self.export_file('dataset/test_set.dat', self.testset)

    @staticmethod
    def loadfile(filename):
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
        fp.close()

    @staticmethod
    def export_file(filename, data):
        with open(filename, "w") as f:
            for line in data:
                f.write(line)
                f.write("\n")


if __name__ == '__main__':
    # res = LoadRatingDataset()
    # print(res['6040'])
    # test = LoadMovieDataset()
    # print(test['5'])

    test = LoadRatingDataset_mat(Parameter.rating_path)
    plt.plot(test)
    plt.show()
    plt.violinplot(test, showmedians=True)
    plt.show()
    # dd = DivideData()
    # dd.generate_data_set(Parameter.rating_path)
