import numpy as np
from logistic import logistic_f, iterate_f
import pytest
from math import isclose
from numpy.testing import assert_allclose


@pytest.mark.parametrize("input_pars, expected",[([.1, 2.2], .198),([.2, 3.4], .544),([.75,1.7],.31875)])
def test_logistic_f(input_pars,expected):
    f_output = logistic_f(input_pars[0],input_pars[1])
    assert isclose(f_output,expected)

#iterate_f(r,seed,num_it)
@pytest.mark.parametrize("input_pars, expected",[([.1, 2.2, 1], [.1, 0.198]),
                                                    ([.2, 3.4, 4], [.2, .544, 0.843418, 0.449019, 0.841163]),
                                                    ([.75,1.7,2],[0.75, 0.31875, 0.369152])])
def test_iterate_f(input_pars,expected):
    f_output = iterate_f(input_pars[1],input_pars[0], input_pars[2])
    assert_allclose(f_output,expected, rtol= 1e-3)


#@pytest.fixture
seed = np.random.randint(0,500)

@pytest.fixture
def random_state():
    print(seed)
    random_state = np.random.RandomState(seed)
    return random_state


def test_convergence(random_state):
     r = 1.5
     
     numIt = 1000
     #x0 = np.random.rand()
     x0 = random_state.rand()

     f_output = iterate_f(r,x0,numIt)
     assert isclose(f_output[-1],1.0/3)

def test_chaos(random_state):
    r = 3.8
    numIt = 100000
    x0 = random_state.rand()
    f_output = np.array(iterate_f(r,x0,numIt))

    # bounded, min - max between 0 and 1
    assert np.all(np.logical_and(f_output >= 0, f_output <= 1))
    num_comp  = 1000
    # last 1000 values different 
    diff_vals = len(np.unique(f_output[-num_comp:].round(decimals=10)))
    assert diff_vals == num_comp