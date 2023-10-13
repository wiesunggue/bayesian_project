import numpy as np

data = [17.88,28.92,33.00,41.52,42.12,45.60,
        48.48,51.84,51.96,54.12,55.56,67.80,
        68.64,68.64,68.88,84.12,93.12,98.64,
        105.12,105.84,127.92,128.04,173.40]
data = np.array(data)

def sample_mean(data):
        return sum(data)/len(data)

def sample_sd(data):
        sm = sample_mean(data)
        return np.sqrt(sum((data-sm)**2)/(len(data)-1))

def CV(data):
        return sample_sd(data)/sample_mean(data)

def skewness(data):
        sm = sample_mean(data)
        sd = sample_sd(data)
        return sum(((data-sm)/sd)**3)/len(data)

def kurtosis(data):
        sm = sample_mean(data)
        sd = sample_sd(data)
        return sum(((data-sm)/sd)**4)/len(data)