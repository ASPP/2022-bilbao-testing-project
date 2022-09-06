def logistic_f(x,r):
    return r*x*(1-x)


def iterate_f(r,seed,num_it):
    #num_it = 100

    x = [seed]
    for i in range(num_it):
        out_f = logistic_f(x[-1],r)
        x.append(out_f)
    return x