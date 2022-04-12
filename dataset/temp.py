import yaml


dataset =     { "equalweights_sequence":{

        'N': 100,
        'M': 25,
        'n': 100,
        'p': 0.8,
        'q': 0.2,
        'COMMON_RATIO': 0.2,
        'RARE_RATIO': 0.8,
}
    }

with open("./dataset/config.yaml", 'w') as yamlfile:
    data = yaml.dump(dataset, yamlfile)
    print("Write successful")


# N = 100 # number of sequences
# M = 25  # Number of pages
# n = 100 # Number of requests
# p = 0.8 # Probability of a common requests
# q = 1-p # Probability of a rare requests
# COMMON_RATIO = 0.2 # Ratio of common requests
# RARE_RATIO = 0.8 # Ratio of rare requests