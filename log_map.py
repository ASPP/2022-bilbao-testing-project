

def logistic_map(x, r):
    x_next = r * x * (1 - x)

    return x_next

def iterate_f(x,r,it):
    it_vec = []
    for i in range(it):
        xi = logistic_map(x,r)
        it_vec.append(xi)
        x = xi
    print(it_vec)

    return it_vec
