import numpy as np
from collections.abc import Callable

def memoize(f:Callable):
    cache = {}
    def wrapper(t:int):
        if t not in cache:
            cache[t] = f(t)
        return cache[t]
    return wrapper

@memoize
def factorial(n:int):
    if n < 0:
        raise ValueError('n need to be non-negative integer')
    if n == 0 or n==1:
        return 1
    else:
        return n * factorial(n-1)
    

def binomial_coefficient(n:int,k:int)->float:
    return factorial(n)/(factorial(k) * factorial(n-k))

def bernstein_polynomial(index:int,degree:int,t:float):
    return binomial_coefficient(degree,index) * t**index * (1-t)**(degree-index)

def bezier_curve(control_points:np.ndarray,degree:int,t:np.ndarray):
    return np.array([
        sum(
            control_points[i] * bernstein_polynomial(i,degree,t[j]) for i in range(degree+1)
        ) 
        for j in range(len(t))
    ])

def bezier_gradient(degree:int,t:np.ndarray):
    return np.array(
        [
            [
                bernstein_polynomial(i,degree,t[j])
                for i in range(degree+1)
            ]
            for j in range(len(t))
        ]
    ).T

def least_square_fit(data:np.ndarray,t:np.ndarray,degree:int):
    A = bezier_gradient(degree,t)
    control_points = np.linalg.pinv(A@A.T) @ (A @ data)
    return control_points