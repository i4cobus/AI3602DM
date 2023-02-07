import pickle
import numpy as np
import copy
from scipy.stats import multivariate_normal
from tqdm import tqdm
import Parameter
import Dataset
import Evaluate
import random
import collections
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA



class UserCluster:
    def __init__(self, train_rating):
        self.usernum = 6040
        self.movienum = 3952
        self.k = 11
        self.user_film_mat = np.zeros((self.usernum,self.movienum))
        self.reduced_dim = 100
        for user in train_rating.keys():
            for film, score in train_rating[user].items():
                self.user_film_mat[int(user)-1][int(film)-1] = score

        print("start PCA ",self.user_film_mat.shape)
        self.dim_reduction()
        print("FInish ",self.user_film_mat.shape)

        self.init_param(self.usernum, self.reduced_dim, self.k)

    def dim_reduction(self):
        """
        Reduce the dimension from input_dim to out_dim
        [samples, input_dim] ---> [samples, output_dim]
        :return: reduced matrix
        """
        xTx = np.dot(self.user_film_mat.T, self.user_film_mat) #[in_dim, in_dim]

        eig_value, eig_vector = np.linalg.eig(xTx)
        self.eig_value = eig_value

        # sort the eigen value from high to low
        sorted_eigvalue_idx = sorted(range(len(eig_value)), key=lambda k: eig_value[k],reverse=True)

        # pick the top k columns in eigenvector corresponding to sorted eigenvalue
        feature_vector = np.array([eig_vector[:,i] for i in sorted_eigvalue_idx[:self.reduced_dim]])
        # [out_dim, n_sample]

        reduced_data = np.dot(self.user_film_mat, feature_vector.T) #[n_sample, out_dim]

        for i in range(reduced_data.shape[1]):
            max_ = reduced_data[:, i].max()
            min_ = reduced_data[:, i].min()
            reduced_data[:, i] = (reduced_data[:, i] - min_) / (max_ - min_)

        with open('InterResult/reduced_mat.pkl', 'wb') as f:
            pickle.dump(reduced_data, f, pickle.HIGHEST_PROTOCOL)

        self.user_film_mat = reduced_data

    @staticmethod
    def phi(Y, mu_k, cov_k):
        norm = multivariate_normal(mean=mu_k, cov=cov_k)
        return norm.pdf(Y)

    def init_param(self, sample, dimension, k):
        self.mu = np.random.rand(k, dimension)
        self.sigma = np.array([np.eye(dimension)] * k)
        self.alpha = np.array([1.0 / k] * k)
        self.gamma = np.mat(np.zeros((sample, k)))

    def Estep(self):

        prob = np.zeros((self.usernum, self.k))

        for i in range(self.k):
            norm = multivariate_normal(mean = self.mu[i], cov = self.sigma[i])
            prob[:, i] = norm.pdf(self.user_film_mat)

        prob = np.mat(prob)

        for i in range(self.k):
            self.gamma[:,i] = self.alpha[i] * prob[:, i]
        for i in range(self.usernum):
            self.gamma[i,:] /= np.sum(self.gamma[i,:])

        return self.gamma


    def Mstep(self):
        for i in range(self.k):
            N_k = np.sum(self.gamma[:,i])
            self.mu[i, :] = np.sum(np.multiply(self.user_film_mat, self.gamma[:, i]), axis=0) / N_k
            # 更新 cov
            self.sigma[i] = (self.user_film_mat - self.mu[i]).T * np.multiply((self.user_film_mat - self.mu[i]), self.gamma[:, i]) / N_k
            # 更新 alpha
            self.alpha[i] = N_k / self.usernum

    def train(self):
        epoch = 10

        for i in tqdm(range(epoch)):
            print(f"[epoch {i}] start E step")
            self.Estep()
            print(f"[epoch {i}] start M step")
            self.Mstep()

        with open('InterResult/EM_mu_sigma_alpha.pkl', 'wb') as f:
            pickle.dump([self.mu,self.sigma,self.alpha], f, pickle.HIGHEST_PROTOCOL)



    def Recommand(self, user):
        pass

class BagUserCluster:
    def __init__(self, train_rating):
        self.usernum = 6040
        self.movienum = 3952
        self.k = 5
        self.user_film_mat = np.zeros((self.usernum,self.movienum))
        self.reduced_dim = 1000
        for user in train_rating.keys():
            for film, score in train_rating[user].items():
                self.user_film_mat[int(user)-1][int(film)-1] = score

        self.pca = PCA(n_components=self.reduced_dim)
        self.pca.fit(self.user_film_mat)
        self.user_film_mat = self.pca.transform(self.user_film_mat)


    def fit(self):
        self.gm = GaussianMixture(n_components=self.k)
        cluster_labels = self.gm.fit_predict(self.user_film_mat)

        tmp = []
        for i in cluster_labels:
            address_index = [x + 1 for x in range(len(cluster_labels)) if cluster_labels[x] == i]
            tmp.append([i, address_index])
        self.dict_address = dict(tmp)

        cluster_size = []
        for i in range(self.k):
            cluster_size.append(len(self.dict_address[i]))
        print("cluster size:\n",cluster_size)



    def recommand(self, test_rating):

        user_film = np.zeros((self.usernum, self.movienum))
        for user in test_rating.keys():
            for film, score in test_rating[user].items():
                user_film[int(user)-1][int(film)-1] = score

        reduced_mat = self.pca.transform(user_film)
        cluster_labels = self.gm.predict(reduced_mat)
        print("test ",cluster_labels[:100])

        for idx, category in enumerate(cluster_labels):
            cluster = self.dict_address[category]
            watched_movies = test_rating[str(idx+1)]
            candidate_movies = dict()
            for similar_user in cluster:
                if str(similar_user) in test_rating:
                    for movies, score in test_rating[str(similar_user)].items():
                        if movies in watched_movies:
                            continue
                        candidate_movies.setdefault(movies, 0)
                        candidate_movies[movies] += score

            candidate_movies_order = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
            print("recom movie",candidate_movies_order[:10])
        return 0



if __name__ == '__main__':
    train_rating = Dataset.LoadRatingDataset(Parameter.train_path)
    test_rating = Dataset.LoadRatingDataset(Parameter.test_path)

    # EMalg = UserCluster(train_rating)
    #
    # EMalg.train()

    bag_em = BagUserCluster(train_rating)
    bag_em.fit()
    bag_em.recommand(test_rating)
    # for user in test_rating:
    #     bag_em.recommand(test_rating[user])
