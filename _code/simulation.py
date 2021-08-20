import numpy as np
import matplotlib.pyplot as plt

def f(x,lam):
    return lam*np.exp(-lam*x) if x>0 else 0

# This is the inverse of the cumulative distribution function of a random variable X we explicitly know
def F_inv(x,lam):
    return -1/lam * np.log(1-x)

def X(lam):
    U = np.random.uniform(0,1) # Generates a uniform random number between 0 and 1
    return F_inv(U,lam)
experiments = 10000
results = []
lam = 2.0
for _ in range(experiments):
    results.append(X(lam))

x = np.linspace(0.01,5,100)
plt.plot(x,[f(_,lam) for _ in x], color="red")
plt.hist(results, density=True, bins=20, color="blue")
plt.show()

def sample_circle():
    accept = False
    while not accept:
        [x,y] = np.random.uniform(-1,1,2)
        if (x**2 + y**2 < 1):
            accept = True
            return [x,y]

def rejection_sampling(experiments):
    accepted = []
    rejected = []
    for _ in range(experiments):
        [x,y] = np.random.uniform(-1,1,2)
        if (x*x + y*y < 1):
            accepted.append([x,y])
        else:
            rejected.append([x,y])
    plt.scatter([a[0] for a in accepted],[a[1] for a in accepted],color='green')
    plt.scatter([b[0] for b in rejected], [b[1] for b in rejected],color='red')
    plt.show()
    return [accepted,rejected]

#rejection_sampling(1000)

def beta_distribution(x,n,m):
    return x**n * (1-x)**m

def wiggly(x):
    return 2*np.exp(-x*x)*x*x / np.sqrt(np.pi)
def campana(x):
    return 0.7*np.exp(-x*x/4)

#x = np.linspace(-3,3,1000)
#plt.plot(x, [campana(_) for _ in x],color='blue')
#plt.plot(x,[wiggly(_) for _ in x],color='red')
#plt.show()

###

## NORMAL VARIABLE
#def n(x):
#    return np.exp(-x*x/2)/np.sqrt(2*np.pi)
#
#def f(x):
#    return 2*np.exp(-x**2/2)/np.sqrt(2*np.pi) if x>= 0 else 0
#
#def g(x):
#    return np.exp(-x) if x>=0 else 0
#
#def simulate_Y():
#    U = np.random.uniform(0,1)
#    return -np.log(U)
#
#def rejection_sampling():
#    while True:
#        U = np.random.uniform(0,1)
#        Y = simulate_Y()
#        if U <= np.exp(-(Y-1)**2 / 2):
#            S = 1 if np.random.uniform(0,1)>0.5 else -1
#            return S*Y
#
#samples = 100000
#res = []
#for _ in range(samples):
#    res.append(rejection_sampling())
#
#x = np.linspace(-5,5,1000)
#plt.plot(x,[n(_) for _ in x], color="red")
#plt.hist(res,density=True, bins = 30, color="blue")
#plt.show()