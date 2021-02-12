import numpy as np
import numpy.linalg as la
from io import StringIO


def pagerank(L, d=0.5):
    n = L.shape[0]
    J = np.ones((n,n))
    M = d * L + ((1-d)/n) * J

    r = 100 * np.ones(n) / n
    r = M @ r
    prevR = r
    r = M @ r

    while la.norm(prevR - r) > 0.01:
        prevR = r
        r = M @ r

    return r

'''
L = np.array([
    [0,   1/2, 1/3, 0, 0,   0, 0 ],
    [1/3, 0,   0,   0, 1/2, 0, 0 ],
    [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
    [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
    [0,   0,   0,   0, 0,   0, 0 ],
    [0,   0,   1/3, 0, 0,   0, 0 ],
    [0,   0,   0,   0, 0,   1/3, 1 ]
])

s = StringIO()
np.savetxt(s, L, fmt="%.3f")
print(s.getvalue())


e_vals, e_vecs = la.eig(L)

vec = np.transpose(e_vecs)[0]
vec = vec * (100/sum(vec))
for num in vec:
    print("%.3f" % num.real)


r = 100 * np.ones(7) / 7
r = L @ r
prevR = r
r = L @ r

while la.norm(prevR - r) > 0.01:
    prevR = r
    r = L @ r

for num in r:
    print("%.3f" % num.real)

d = 0.5
n = 7
J = np.ones((7,7))
M = d * L + ((1-d)/n) * J

r = 100 * np.ones(7) / 7
r = M @ r
prevR = r
r = M @ r

while la.norm(prevR - r) > 0.01:
    prevR = r
    r = M @ r
'''
tokens = input().split()
dim = int(tokens[0])
#damping = float(tokens[1])

sites = [site for site in input().split()]


linkMat = np.zeros((dim, dim))
for i in range(dim):
    tokens = input().split()
    for j, token in enumerate(tokens):
        linkMat[i][j] = np.double(token)

searchTerm = input()
ranks = pagerank(linkMat)
if searchTerm in sites:
    ranks[sites.index(searchTerm)] = 101.0

topSites = sorted(zip(sites, ranks), key=lambda x:x[0], reverse=True)  # Deal with unstated requirement
topSites = sorted(topSites, key=lambda x:x[1], reverse=True)[:5]

for site, _ in topSites:
    print(site)

#for num in ranks:
    #print("%.3f" % num.real)
