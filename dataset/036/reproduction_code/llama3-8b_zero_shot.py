import numpy as np

gamma = 0.99
lambda_ = 0.95
r1, r2 = 10, 20

w1 = (1 - lambda_) * pow(lambda_, 0)
w2 = (1 - lambda_) * pow(lambda_, 1)

v1 = w1 * r1 + w2 * r2

v2 = gamma ** 2 * r2
v3 = gamma ** 2 * r2

print(v1, v2, v3)