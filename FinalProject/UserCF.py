import Parameter
import Dataset
from tqdm import tqdm
import pickle
import Evaluate
import random


class UserBasedCF:

    def __init__(self, train_set):
        self.trainset = train_set

        self.n_sim_user = 20
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

    def CalUserSim(self, pretrain):
        if pretrain == False:
            for userA in tqdm(self.trainset.keys()):
                for userB in self.trainset.keys():
                    if userA == userB: continue
                    self.user_sim_mat.setdefault(userA,{})
                    self.user_sim_mat[userA].setdefault(userB,0)
                    self.user_sim_mat[userA][userB] = self.CalCosSim(self.trainset[userA],self.trainset[userB])
            with open('InterResult/UserSimMat_0.8.pkl', 'wb') as f:
                pickle.dump(self.user_sim_mat, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('InterResult/UserSimMat_0.8.pkl', 'rb') as f:
                self.user_sim_mat = pickle.load(f)
            print("Finish UserCF Initialization")

    @staticmethod
    def CalCosSim(userA, userB):
        A_avg_score = 0
        B_avg_score = 0
        for i in userA.values():
            A_avg_score += i
        for i in userB.values():
            B_avg_score += i
        A_avg_score /= len(userA)
        B_avg_score /= len(userB)

        A_fenmu = 0
        B_fenmu = 0
        for i in userA.values():
            A_fenmu += ((i - A_avg_score) ** 2)
        for i in userB.values():
            B_fenmu += ((i - B_avg_score) ** 2)
        A_fenmu = A_fenmu ** 0.5
        B_fenmu = B_fenmu ** 0.5
        fenmu = A_fenmu * B_fenmu

        fenzi = 0
        for i in userA.keys():
            if i in userB.keys():
                fenzi += (userA[i] - A_avg_score) * (userB[i] - B_avg_score)
        if fenmu == 0:
            return 1
        return fenzi / fenmu


    def MovieRecommend(self, user):
        """
        :param user: The id of user in string form
        :return: [(movie,weight),...,(movie,weight)]. movie in string form from 1 to 3952, weight in float form
        """
        K = self.n_sim_user
        N = self.n_rec_movie
        candidate_movies = dict()
        watched_movies = self.trainset[user]
        similar_users = sorted(self.user_sim_mat[user].items(), key=lambda x: x[1], reverse=True)
        similar_users = similar_users[:self.n_sim_user]

        for similar_user, weight in similar_users:
            for movies, score in self.trainset[similar_user].items():
                if movies in watched_movies:
                    continue
                candidate_movies.setdefault(movies, 0)
                candidate_movies[movies] += score * weight

        candidate_movies_order = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
        # print("recommend res",candidate_movies_order[:self.n_rec_movie])
        return candidate_movies_order[:self.n_rec_movie]


if __name__ == '__main__':
    train_rating = Dataset.LoadRatingDataset(Parameter.train_path)
    test_rating = Dataset.LoadRatingDataset(Parameter.test_path)
    ucf = UserBasedCF(train_rating)
    ucf.CalUserSim(True)


    accurate = 0
    recall = 0
    for user in test_rating.keys():
        rec_movies = ucf.MovieRecommend(user)
        truth_movies = test_rating[user]
        hit, total = Evaluate.CountEval(rec_movies, truth_movies)

        if total != 0:
            accurate += (hit/total)
        recall += (hit/10)

    # random test
    # for user in test_rating.keys():
    #
    #     rec_movies = []
    #     for i in range(10):
    #         rec_movies.append((str(random.randint(1,3952)),0))
    #
    #     truth_movies = test_rating[user]
    #     hit, total = Evaluate.CountEval(rec_movies, truth_movies)
    #
    #     if total != 0:
    #         accurate += (hit/total)
    #     recall += (hit/10)

    print(len(test_rating.keys()))
    print("acc cnt", accurate)
    print("accurate",accurate/len(test_rating.keys()))
    print("recall cnt", recall)
    print("recall",recall/len(test_rating.keys()))




    # res = UserCFRecommand(test_user,rating)
    # id2name = Dataset.LoadMovieDataset()
    # for i in res:
    #     print(id2name[i[0]])






