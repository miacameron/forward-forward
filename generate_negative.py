import numpy as np

gaussian_kernel = (1/12) * np.array([[0, 2, 0], 
                                     [2, 4, 2], 
                                     [0, 2, 0]], dtype=np.float16)

def convolve(A, K):

    """
    Convolves a square kernel across a square input image A
    """

    B = np.zeros((A.shape)) # new image same size as output
    r = A.shape[0] # assume that the image is square
    n = K.shape[0] # assume that the kernel is square

    A_ = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
    A_[1:r+1, 1:r+1] = A

    for i in range(r):
        for j in range(r):
            B[i,j] = np.sum( np.multiply(A_[i:i+n, j:j+n], K) ) # sum of elementwise / Hadamard product
    
    return B


def generate_masks(num_masks, num_blurs):

    masks = np.random.randint(0, 2, (num_masks, 28, 28)).astype(np.float16)

    for i in range(num_masks):
        for _ in range(num_blurs):
            masks[i] = convolve(masks[i], gaussian_kernel)
        
    thresh = 0.5# masks[0].mean()
    masks = (masks > thresh)
        
    return masks


def mask_combine_images(A, B, M):
    A_ = np.multiply(A, M)
    B_ = np.multiply(B, (M == 0))

    return A_ + B_


def generate_negatives(X_pos):

    N = X_pos.shape[0]
    X_neg = np.zeros((X_pos.shape))
    masks = generate_masks(20, 10)

    for i in range(N):

        a_i = np.random.randint(0, N)
        b_i = np.random.randint(0, N)

        while (a_i == b_i):
            b_i = np.random.randint(0, N)

        A = X_pos[a_i] # choose two random positive images
        B = X_pos[b_i]
        M = masks[np.random.randint(0, masks.shape[0])] # choose a random mask

        X_neg[i] = mask_combine_images(A, B, M)

    return X_neg
