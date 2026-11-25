import torch  # tensor library
import torch.nn as tn  # importing neural network layers

x = torch.tensor([[[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]]) # input of 3 words or numbers
Wq = tn.Linear(2, 2)  # layer that creates queries
Wk = tn.Linear(2, 2)  # layer that creates keys
Wv = tn.Linear(2, 2)  # layer that creates values
Q = Wq(x)  # queries for each word
K = Wk(x)  # keys for each word
V = Wv(x)  # values for each word
# matching queries with keys divide by sqrt(2) so that numbers are small
scores = torch.matmul(Q, K.transpose(-2, -1)) / (2 ** 0.5)
# turn scores into percentages that have sum of 1
attentionWeights = torch.softmax(scores, dim=-1)
# mix the values using percentages
output = torch.matmul(attentionWeights, V)
print(output)  # to show the result
