def calc_max_mat_norm(Y, A_inv):
    n = Y.shape[0]
    res = np.zeros(n)
    for i in range(n):
        y = Y[i]
        res[i] = y.T @ A_inv @ y
    ind = np.argmax(res)
    return res[ind], ind
        
def f(X, Y, lam):
    d = X.shape[1]
    A = X.T @ np.diag(lam) @ X + 0.0001 * np.eye(d)
    A_inv = np.linalg.inv(A)
    res, ind = XYStatic.calc_max_mat_norm(Y, A_inv)
    return res, ind, A_inv

def grad_f(X, Y, lam):
    # TODO: better variable name for X
    _, ind, A_inv = f(X, Y, lam)
    return -np.power(Y @ A_inv @ Y[ind], 2)
        
def frank_wolfe(X, Y, grad_f, N=500):
    n, d = X.shape
    inds = np.random.choice(n, 2*d, p=1/n * np.ones(n)).tolist()
    lam = np.bincount(inds, minlength=n) / len(inds)
    lam[np.argmax(lam)] += (1 - sum(lam))  # Yes, this is super hacky, I know. Fix later.
    for i in range(2*d+1, N+1):
        eta = 2/(i+1)
        ind = np.argmin(grad_f(X, Y, lam))
        lam = (1-eta)*lam + eta * np.eye(1, n, ind).flatten()
        lam[np.argmax(lam)] += (1 - sum(lam))
        inds.append(ind)
    return lam