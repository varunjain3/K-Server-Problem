from tabnanny import verbose
import yaml
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(42)

def verify_sequence(requests):
    n,values = np.unique(requests, return_counts=True)
    plt.bar(n,values)
    plt.show()


def equal_weights(N=100,verbose=False):

    with open("dataset/config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        data = data['equalweights_sequence']
        yamlfile.close()

    M = data['M']  # Number of pages
    n = data['n'] # Number of requests
    p = data['p'] # Probability of a common requests
    q = 1-p # Probability of a rare requests
    COMMON_RATIO = data['COMMON_RATIO'] # Ratio of common requests
    RARE_RATIO = data['RARE_RATIO'] # Ratio of rare requests


    network_graph = np.ones((M,M))

    requests = []
    for j in range(N):
        COMMON_PAGES = np.random.choice(M, int(M*COMMON_RATIO), replace=False)
        RARE_PAGES = []
        request = []

        for i in range(M):
            if not i in COMMON_PAGES:
                RARE_PAGES.append(i)

        if verbose:
            print(f"Common pages: {COMMON_PAGES}")
            print(f"Rare pages: {RARE_PAGES}")

        for i in range(n):      
            if np.random.rand() > p:
                request.append(np.random.choice(RARE_PAGES))
            else:
                request.append(np.random.choice(COMMON_PAGES))    

        if verbose:
            verify_sequence(request)   
        requests.append(request)
    return requests

if __name__ == "__main__":
    requests = equal_weights(verbose=True)

