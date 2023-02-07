import pandas
import numpy
import random


gt = numpy.array(pandas.read_csv("./data/ground_truth.csv"))
res = numpy.array(pandas.read_csv("./res.csv"))

for i, j in res:
    assert res[i, 0] == i

m_perf = 0.0

for attempt in range(5000):
    d5 = list(range(5))
    random.shuffle(d5)
    perf = 0
    for i, j in gt:
        if res[i, 1] == d5[j]:
            perf += 1
    m_perf = max(m_perf, perf / len(gt))
print(m_perf)
