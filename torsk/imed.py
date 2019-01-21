import numpy as np

def metric_matrix(img_shape,sigma=1):
    (M,N) = img_shape;
    
    X = np.arange(M); # M,N
    Y = np.arange(N)

    P = (X[None,:,None,None]-X[None,None,None,:])**2 \
      + (Y[:,None,None,None]-Y[None,None,:,None])**2

    G = 1/(2*np.pi*sigma**2)*np.exp(-(0.5/(2*sigma**2))*P)    
   
    return G.reshape((M*N,M*N));


def metric_matrix_slow(img_shape):
    size = img_shape[0] * img_shape[1]
    G = np.empty([size, size])

    coords = [[x, y] for x in range(img_shape[1]) for y in range(img_shape[0])]
    coords = np.asarray(coords)

    for ii in range(size):
        for jj in range(size):
            pipj = np.sum((coords[ii] - coords[jj])**2.)
            G[ii, jj] = np.exp(- pipj * 0.5) / (2 * np.pi)
    return G




def imed_metric(a_imgs, b_imgs):
    assert a_imgs.shape == b_imgs.shape
    a_seq = a_imgs.reshape([a_imgs.shape[0], -1])
    b_seq = b_imgs.reshape([b_imgs.shape[0], -1])
    G = metric_matrix(a_imgs.shape[1:])
    return np.array([(x - y).dot(G.dot(x - y)) for x, y in zip(a_seq, b_seq)])
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torsk.data.utils import gauss2d_sequence, mackey_sequence, normalize
    
    t = np.arange(0, 4*np.pi, 0.5)
    #x, y = np.sin(t), np.cos(0.3 * t)
    x, y = np.sin(t), np.cos(t)
    # x = normalize(mackey_sequence(N=t.shape[0])) * 2 - 1
    shape = [20, 20]
    
    center = np.array([y, x]).T
    images = gauss2d_sequence(center, sigma=0.5, size=shape)
    
    plt.imshow(images[0])
    plt.show()
    
    G = metric_matrix(shape)
    print(G)
    plt.imshow(G)
    plt.colorbar()
    plt.show()
    
    diff = []
    for ii in range(images.shape[0] - 1):
        x = images[ii].reshape(-1)
        y = images[0].reshape(-1)
        # plt.plot(x - y)
        # plt.plot(G.dot(x-y))
        # plt.show()
        d = (x - y).dot(G.dot(x - y))
        diff.append(d)
    
    eucl = (((images - images[0])**2.)**.5).sum(axis=-1).sum(axis=-1)
    
    plt.plot(diff)
    plt.plot(eucl)
    plt.show()
