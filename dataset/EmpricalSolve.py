import yaml
import json
import numpy as np

from tqdm import tqdm
from equalweights import equal_weights


def solve_kserver(request, verbose=False):
    """
    Solve the k-server problem.
    """
    with open("dataset/config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        K = data['kservers']
        yamlfile.close()

    K_pages = [-1 for i in range(K)]
    optimal_k = []
    for i in range(len(request)):
        flag = 0
        if request[i] in K_pages:
            optimal_k.append(-1)
            continue
        for k in range(len(K_pages)):
            if K_pages[k] == -1:
                K_pages[k] = request[i]
                flag = 1
                break
        else:
            temp = []
            for j in range(i+1,len(request)):
                if request[j] in K_pages and request[j] not in temp:
                    temp.append(request[j])
                
                if len(temp) == K-1 or j == len(request)-1:
                    for k in range(K):
                        if not K_pages[k] in temp:
                            K_pages[k] = request[i]
                            flag = 1
                            break
        if flag:
            optimal_k.append(k)

    return optimal_k
    print(f"Optimal k-server: {optimal_k}")

        


print(0)

if __name__ == "__main__":
    requests = equal_weights(10000)
    dataset = []

    for i in tqdm(range(len(requests))):
        solved_request = solve_kserver(requests[i])
        dataset.append({
            'request': ",".join(map(str,requests[i])),
            'solved_request': ",".join(map(str,solved_request)),
            'id': i+1
        })
    
    with open('equalweights.json','w', encoding ='utf8') as f:
        json.dump(dataset,f)

