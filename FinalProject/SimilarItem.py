import Parameter
import Dataset
from tqdm import tqdm
import pickle
import Evaluate
import numpy as np
import random
import collections


class SimItem:
    def __init__(self, train_rating, pretrain):
        self.hashnum = 30
        self.maxsingle = 6039
        self.movienum =3952
        self.like_matrix = [[] for _ in range(self.movienum)]
        for user in train_rating.keys():
            for film, score in train_rating[user].items():
                if score > 2:
                    self.like_matrix[int(film)-1].append(int(user)-1)

        if pretrain == True:
            with open('InterResult/MovieSimMat-0.8.pkl', 'rb') as f:
                self.SimMatrix = pickle.load(f)
            print("Initialization Finish")
        else:
            print("Run FindSignature and CompareSignature first to build matrix")

    def pickRandomCoeffs(self):
        # Create a list of 'k' random values.
        randList = []
        k = self.hashnum
        maxShingleID = 2 ** 32 - 1

        while k > 0:
            # Get a random shingle ID.
            randIndex = random.randint(0, maxShingleID)
            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = random.randint(0, maxShingleID)
                # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1
        return randList

    def FindSignature(self):
        coeffA = self.pickRandomCoeffs()
        coeffB = self.pickRandomCoeffs()
        signatures = []

        for movie in tqdm(range(self.movienum)):
            # Get the shingle set for this document.
            shingleIDSet = self.like_matrix[movie]
            # The resulting minhash signature for this document.
            signature = []
            # For each of the random hash functions...
            for i in range(0, self.hashnum):
                # For each of the shingles actually in the document, calculate its hash code
                # using hash function 'i'.

                # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
                # the maximum possible value output by the hash.
                minHashCode = self.maxsingle + 1

                # For each shingle in the document...
                for shingleID in shingleIDSet:
                    # Evaluate the hash function.
                    hashCode = (coeffA[i] * shingleID + coeffB[i]) % (self.maxsingle+1)

                    # Track the lowest hash code seen.
                    if hashCode < minHashCode:
                        minHashCode = hashCode

                # Add the smallest hash code value as component number 'i' of the signature.
                signature.append(minHashCode)

            # Store the MinHash signature for this document.
            signatures.append(signature)

        with open('InterResult/MovieSignature-0.8.pkl', 'wb') as f:
            pickle.dump(signatures, f, pickle.HIGHEST_PROTOCOL)
        self.signatures = signatures

    def CompareSignature(self):

        self.SimMatrix = np.zeros((self.movienum,self.movienum))

        for i in tqdm(range(0, self.movienum)):
            # Get the MinHash signature for document i.
            signature1 = self.signatures[i]

            # For each of the other test documents...
            for j in range(i + 1, self.movienum):

                # Get the MinHash signature for document j.
                signature2 = self.signatures[j]
                count = 0
                # Count the number of positions in the minhash signature which are equal.
                for k in range(0, self.hashnum):
                    count = count + (signature1[k] == signature2[k])

                # Record the percentage of positions which matched.
                self.SimMatrix[i][j] = count / self.hashnum

        with open('InterResult/MovieSimMat-0.8.pkl', 'wb') as f:
            pickle.dump(self.SimMatrix, f, pickle.HIGHEST_PROTOCOL)

    def MovieRecommand(self, movie):
        """
        :param movie: The id of movie in string form
        :return: A list with ten most similar movies in string form
        """

        recommand_number = 10

        # with open('InterResult/MovieSimMat-0.8.pkl', 'rb') as f:
        #     self.SimMatrix = pickle.load(f)


        Recommand = sorted(range(len(self.SimMatrix[int(movie)-1])), key=lambda k: self.SimMatrix[int(movie)-1][k], reverse=True)

        Result = []
        for i in range(recommand_number):
            Result.append(str(Recommand[i]+1))

        return Result




if __name__ == '__main__':
    train_rating = Dataset.LoadRatingDataset(Parameter.train_path)
    test_rating = Dataset.LoadRatingDataset(Parameter.test_path)

    sit = SimItem(train_rating,True)

    # sit.FindSignature()
    # sit.CompareSignature()

    accurate = 0
    recall = 0

    for user in tqdm(test_rating.keys()):
        relate_movies = []
        for movies, _ in train_rating[user].items():
            relate_movies.extend(sit.MovieRecommand(movies))

        rec_movies = collections.Counter(relate_movies)
        tmp_movie = []
        rec_movies = sorted(rec_movies.items(), key=lambda x: x[1], reverse=True)[:10]

        truth_movies = test_rating[user]
        hit, total = Evaluate.CountEval(rec_movies, truth_movies)

        if total != 0:
            accurate += (hit / total)
        recall += (hit / 10)

    print(len(test_rating.keys()))
    print("acc cnt",accurate)
    print("accurate", accurate / len(test_rating.keys()))
    print("recall cnt",recall)
    print("recall", recall / len(test_rating.keys()))



