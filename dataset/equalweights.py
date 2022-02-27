import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(42)


M = 25  # Number of pages
N = 100 # Number of requests
p = 0.8 # Probability of a common requests
q = 1-p # Probability of a rare requests
COMMON_RATIO = 0.2 # Ratio of common requests
RARE_RATIO = 0.8 # Ratio of rare requests

network_graph = np.ones((M,M))

COMMON_PAGES = np.random.choice(M, int(M*COMMON_RATIO), replace=False)
RARE_PAGES = []

for i in range(M):
    if not i in COMMON_PAGES:
        RARE_PAGES.append(i)

print(f"Common pages: {COMMON_PAGES}")
print(f"Rare pages: {RARE_PAGES}")

requests = []

for i in range(N):
    if np.random.rand() > p:
        requests.append(np.random.choice(RARE_PAGES))
    else:
        requests.append(np.random.choice(COMMON_PAGES))    


def verify_sequence(requests):
    n,values = np.unique(requests, return_counts=True)
    plt.bar(n,values)
    plt.show()

verify_sequence(requests)   
print(requests)
