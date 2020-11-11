import numpy as np

def uniform_sampling_ball(centroid, radius):
    """
    Uniformly sample a ball around the centroid.
    Reference: https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
    :param centroid:
    :param radius:
    :return:
    """
    dim = len(centroid)
    Y = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    S = Y / np.linalg.norm(Y)
    r = radius * np.random.uniform(0, 1, dim) ** (1/dim)
    X = r * S

    return X + centroid

centroid = np.array([1, 1])
r = 1
res = []
for _ in range(50):
    X = uniform_sampling_ball(centroid, r)
    res.append(X)
    print(X)

