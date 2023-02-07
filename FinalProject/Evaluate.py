import numpy as np



def CountEval(recommandation, truth):
    hit = 0
    total = len(truth)
    for movie,_ in recommandation:
        if movie in truth:
            hit += 1

    return hit,total






