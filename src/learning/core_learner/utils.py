import torch, pdb, math, numpy
import matplotlib.pyplot as plt


def precision_at_k(scores, labels, k=5, dim=1):
    b = scores.size(0)
    topk = torch.topk(scores, k=k, dim=dim, largest=True)[1].float()
    matches = labels.view(-1, 1).expand(b, k) == topk
    return matches.float().sum().item() / b


def pca(X, ndim=2):
    X_mean = torch.mean(X, 1)
    X = X - X_mean.unsqueeze(1).expand_as(X)
    U, S, V = torch.svd(X)
    pcs = U[:, :ndim]
    return pcs


def eigenmap(X, n=1000, ndim=2):
    W = torch.mm(X, X.t())
    W = W[:n, :n]
    e,v=torch.symeig((W+W.t())/2, eigenvectors=True)
    '''
    W = W + W.min()
    W = W / W.max()
    D = torch.sum(W, 1).squeeze()
    Dinv = torch.diag(1/torch.sqrt(D))
    D = torch.diag(D)
    L = Dinv*(D - W)*Dinv
    '''
    return v[:, :2]


def scatter(data, k, size=(80, 80, 3)):

    x = data[:, 0]
    y = data[:, 1]
    # convert to pixel coordinates
    x = x - x.min()
    x = x / x.max()
    x = x * (size[0]-1)
    y = y - y.min()
    y = y / y.max()
    y = y * (size[1]-1)
    n = data.shape[0]
    img = numpy.zeros(size)
    for i in range(n):
        img[int(round(x[i]))][int(round(y[i]))][0] = 0.5
        img[int(round(x[i]))][int(round(y[i]))][1] = 0.5
        img[int(round(x[i]))][int(round(y[i]))][2] = 0.5

    img[int(round(x[k]))][int(round(y[k]))][0] = 1
    img[int(round(x[k]))][int(round(y[k]))][1] = 0
    img[int(round(x[k]))][int(round(y[k]))][2] = 0
    return img
