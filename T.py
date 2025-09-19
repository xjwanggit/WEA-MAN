from scipy import stats
import numpy as np

# 成对样本t检验
sample1 = [1.623, 3.186, 15.474, 4.23, 7.72, 13.38, 23.55, 5.86, 5.47, 12.64, 17.87, 12.26]
sample2 = [39.22, 33.80, 27.31, 35.46, 32.78, 24.41, 23.19, 39.93, 39.51, 30.55, 29.31, 34.23]
sample1 = np.asarray(sample1)
sample2 = np.asarray(sample2)
r = stats.ttest_rel(sample1, sample2)
print("statistic:", r.__getattribute__("statistic"))
print("pvalue:", r.__getattribute__("pvalue"))