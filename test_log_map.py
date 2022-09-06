import pytest
import numpy as np
from log_map import logistic_map
from log_map import iterate_f
from math import isclose

@pytest.mark.parametrize("x,r,result",
                         [(0.1, 2.2,0.198),
                          (0.2, 3.4,0.544),
                          (0.75, 1.7, 0.31875)])
def test_log_map_single(x,r,result):
    output = logistic_map(x, r)

    assert isclose(output, result)

SEED = np.random.randint(0,20)

@pytest.fixture
def random_state():
    print(f'using seed {SEED}')
    random_state = np.random.RandomState(SEED)
    return random_state

def test_convergence(random_state):
    r = 1.5
    x = random_state.rand()
    result = 1/3
    output = iterate_f(x,r,30)[-1]
    
    assert np.isclose(output, result,atol=0.000001)

def test_orbit(random_state):
    r = 3.8
    it = 100000
    x = random_state.rand()
    output = np.array(iterate_f(x,r,it))

    assert np.all(np.logical_and(output >= 0, output <= 1))