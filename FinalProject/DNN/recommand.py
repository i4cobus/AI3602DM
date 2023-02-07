import torch
from tqdm import tqdm
import numpy as np
import Dataset
import Parameter


class NCFrecommand:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.movienum = 3952
        self.usernum = 6040

    def recommand(self, user:int):
        """
        :param user: integer, starting from 1 to 6040
        :return: top-10 recommand movie [int,int,...,int] from 1 to 3592
        """
        user_movie = np.zeros((6040,2), dtype=np.long)
        for i in range(self.movienum):
            user_movie[i][0] = user-1
            user_movie[i][1] = i
        user_movie = torch.from_numpy(user_movie)

        test = self.model(user_movie)
        all_movies = test.detach().numpy()
        # for i in [25, 27, 30, 32, 49, 52, 81, 82, 89, 95, 96, 103, 105, 110, 120, 128, 130, 135, 147, 182, 191, 198, 210]:
        #     print("test ",all_movies[i])
        # print("\n return by net",all_movies[:30])

        recommand = sorted(range(1, self.movienum+1), key = lambda k:all_movies[k-1], reverse=True)
        # recommand = sorted(range(self.movienum), key = lambda k:all_movies[k], reverse=True)
        # print("return by command ",recommand[:30])
        return recommand




if __name__ == "__main__":
    train_rating = Dataset.LoadRatingDataset("../dataset/train_set.dat")
    test_rating = Dataset.LoadRatingDataset("../dataset/test_set.dat")


    modelpath = "../InterResult/NCFmodel_50.pkl"
    ncf = NCFrecommand(modelpath)

    accurate = 0
    recall = 0
    for user in tqdm(test_rating.keys()):
        watched_movies = train_rating[user]

        list_movie = ncf.recommand(int(user))
        cnt = 0
        rec_movie = []
        while len(rec_movie)<100:
            if str(list_movie[cnt]) in watched_movies:
                cnt += 1
                continue
            else:
                rec_movie.append(list_movie[cnt])
                cnt += 1

        test_movies = test_rating[user]
        hit = 0
        for i in rec_movie:
            if str(i) in test_movies:
                hit += 1
        total = len(test_movies)
        if total != 0:
            accurate += (hit/total)
        recall += (hit/100)

    print("acc cnt", accurate)
    print("accurate",accurate/len(test_rating.keys()))
    print("recall cnt", recall)
    print("recall",recall/len(test_rating.keys()))




