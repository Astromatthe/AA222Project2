#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

# strategy map and config map
best_strategy_map = {
        'simple1': "quadratic_penalty_l_bfgs",
        'simple2': "quadratic_penalty_l_bfgs",
        'simple3': "quadratic_penalty_l_bfgs",    # quadratic_penalty_l_bfgs passes
        'secret1': "quadratic_penalty_l_bfgs",
        'secret2': "quadratic_penalty_l_bfgs"
    }
strategy_config_map = {
    'simple1':{
        'quadratic_penalty_l_bfgs': {
                'penalty': {'rho_init': 0.01, 'rho_max': 1e6, 'inc': 5}, # 0.01, 1e6, 5
                'l_bfgs': {'m_max': 5},   # 5
                'line_search': {'n_searches': 20, 'step': 0.01, 'beta': 0.5, 'sigma': 1e-4} # 20, 0.01, 0.5, 1e-4
            },
        'quadratic_penalty_gradient_descent': {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            }
    },
    'simple2':{
        'quadratic_penalty_l_bfgs':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'l_bfgs': {'m_max': 10},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            },
            'quadratic_penalty_gradient_descent':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            }
    },
    'simple3':{
        'quadratic_penalty_l_bfgs':
            {
                'penalty': {'rho_init': 0.0001, 'rho_max': 1e6, 'inc': 3}, # 20, 1e6, 1.3
                'l_bfgs': {'m_max': 5},    # 10
                'line_search': {'n_searches': 20, 'step': 0.01, 'beta': 0.5, 'sigma': 1e-4} # 20, 0.01, 0.5, 1e-4
            },
            'quadratic_penalty_gradient_descent':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            }
    },
    'secret1':{
        'quadratic_penalty_l_bfgs':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'l_bfgs': {'m_max': 10},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            },
            'quadratic_penalty_gradient_descent':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            }
    },
    'secret2':{
        'quadratic_penalty_l_bfgs':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'l_bfgs': {'m_max': 10},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            },
            'quadratic_penalty_gradient_descent':
            {
                'penalty': {'rho_init': 0.001, 'rho_max': 1e6, 'inc': 1.5},
                'line_search': {'n_searches': 20, 'step': 0.5, 'beta': 0.5, 'sigma': 1e-4}
            }
    },
}

def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments and returns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """

    strategy = best_strategy_map[prob]
    config = strategy_config_map[prob][strategy]
    x_history = [x0.copy()]

    if strategy == "quadratic_penalty_l_bfgs":
        x_history, _, _, _, _, _, _ = quadratic_penalty_l_bfgs(f, g, c, x0, n, count, config)
        x_best = x_history[-1]
    
    if strategy == "quadratic_penalty_gradient_descent":
        x_history, _, _, _ = quadratic_penalty_gradient_descent(f, g, c, x0, n, count, config)
        x_best = x_history[-1]

    return x_best

# ---- helper functions for penalty methods ----
def quadratic_penalty(x, c):
    return np.sum(np.maximum(0, c(x))**2)

def quadratic_penalty_gradient(x, c):
    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e_i = np.zeros_like(x)
        e_i[i] = eps
        grad[i] = (quadratic_penalty(x + e_i, c) - quadratic_penalty(x - e_i, c)) / (2 * eps)
    return grad

def penalized_f(x, f, c, rho):
    pen = quadratic_penalty(x, c)
    f_val = f(x)
    f_pen = f_val + rho * pen
    return f_pen, f_val, pen

def penalized_g(x, g, c, rho):
    pen_grad = quadratic_penalty_gradient(x, c)
    g_x = g(x)
    g_pen = g_x + rho * pen_grad
    return g_pen, g_x, pen_grad

# ---- optimization algorithms ----
def quadratic_penalty_gradient_descent(f, g, c, x0, n, count, config):
    """
    Quadratic penalty method with gradient descent for the inner loop.
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments and returns current count
        config (dict): configuration dictionary for penalty and gradient descent parameters
    Returns:
        x_history (list of np.array): history of positions visited during optimization
    """
    x = x0.copy()
    x_history = [x.copy()]
    # penalty parameter
    rho = config['penalty']['rho_init']
    rho_max = config['penalty']['rho_max']
    inc = config['penalty']['inc']

    # initialize
    f_pen_x0, f_x0, pen_f_x0 = penalized_f(x, f, c, rho)
    f_pen_history = [f_pen_x0] # f + penalty
    f_history = [f_x0] # f only
    pen_f_history = [pen_f_x0] # penalty only

    # penalized_g costs: g(2) + finite-diff penalty gradient (2*dim calls to c)
    penalized_g_cost = 2 + 2 * len(x)

    while True:
        if count() + penalized_g_cost > n:
            break
        g_pen, _, _ = penalized_g(x, g, c, rho)
        grad_norm = np.linalg.norm(g_pen)
        if grad_norm < 1e-9:
            break

        # line search
        _, x_new, f_pen_new, f_new, pen_f_new = line_search(lambda xx: penalized_f(xx, f, c, rho), x, -g_pen, g_pen, count, n, cost_per_eval=2, config=config)

        x_history.append(x_new.copy())
        f_pen_history.append(f_pen_new)
        f_history.append(f_new)
        pen_f_history.append(pen_f_new)

        # update penalty parameter
        if count() + 1 <= n: # cost for c
            if np.max(c(x)) > 1e-3:
                rho = min(inc * rho, rho_max)

        x = x_new
        if count() >= n:
            break
    return x_history, f_pen_history, f_history, pen_f_history
            
def quadratic_penalty_l_bfgs(f, g, c, x0, n, count, config):
    """
    Quadratic penalty method with L-BFGS optimization for the inner loop.
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments and returns current count
        config (dict): configuration dictionary for penalty and L-BFGS parameters
    Returns:
        x_history (list of np.array): history of positions visited during optimization
    """
    # initialize  
    x = x0.copy()
    x_history = [x.copy()]  # x
    f_pen_x0, f_x0, pen_f_x0 = penalized_f(x, f, c, rho=config['penalty']['rho_init'])
    f_pen_history = [f_pen_x0] # f + penalty
    f_history = [f_x0] # f only
    pen_f_history = [pen_f_x0] # penalty only
    g_pen_x0, g_x0, pen_g_x0 = penalized_g(x, g, c, rho=config['penalty']['rho_init'])
    g_pen_history = [g_pen_x0] # g + g penalty
    g_history = [g_x0] # g only
    pen_g_history = [pen_g_x0] # g penalty only

    # penalty parameter
    rho = config['penalty']['rho_init']
    rho_max = config['penalty']['rho_max']
    inc = config['penalty']['inc']

    # L-BFGS parameters
    m_max = config['l_bfgs']['m_max']  # memory size
    deltas = []  # list to store delta_k = x_{k+1} - x_k
    gammas = []  # list to store gamma_k = grad_{k+1} - grad_k

    dim = len(x)
    # penalized_g costs: g(2) + 2*dim calls to c(1 each) via finite differences
    penalized_g_cost = 2 + 2 * dim
    # penalized_f costs: f(1) + c(1)
    penalized_f_cost = 2

    while True:
        if count() + penalized_g_cost > n:
            break
        g_pen, _, _ = penalized_g(x, g, c, rho)
        grad_norm = np.linalg.norm(g_pen)
        if grad_norm < 1e-6:
            break

        # compute L-BFGS direction using two-loop recursion
        m = len(deltas)
        if m > 0:
            q = g_pen.copy()
            alphas = []
            # backward loop
            for delta, gamma in zip(reversed(deltas), reversed(gammas)):
                alpha = (delta @ q) / (gamma @ delta + 1e-8)
                alphas.append(alpha)
                q -= alpha * gamma

            z = (gammas[-1] @ deltas[-1]) / (gammas[-1] @ gammas[-1] + 1e-8) * q

            # forward loop
            for delta, gamma, alpha in zip(deltas, gammas, reversed(alphas)):
                z = z + delta * (alpha - (gamma @ z) / (gamma @ delta + 1e-8))
            d_dir = -z
        else:
            d_dir = -g_pen

        # line search: penalized_f costs 2 per eval (f + c)
        alpha, x_new, f_pen_new, f_new, pen_f_new = line_search(
            lambda xx: penalized_f(xx, f, c, rho), 
            x, 
            d_dir, 
            g_pen, 
            count, 
            n, 
            cost_per_eval=penalized_f_cost, 
            config=config)

        # update history
        x_history.append(x_new.copy())
        f_pen_history.append(f_pen_new)
        f_history.append(f_new)
        pen_f_history.append(pen_f_new)

        # need budget for another penalized_g + c check before continuing
        if count() + penalized_g_cost + 1 > n:
            x = x_new
            break
        g_pen_new, g_new, pen_g_new = penalized_g(x_new, g, c, rho)
        g_pen_history.append(g_pen_new)
        g_history.append(g_new)
        pen_g_history.append(pen_g_new)

        # update memory
        gamma_new = g_pen_new - g_pen
        delta_new = x_new - x

        # skip update if curvature condition fails (keeps Hessian approx PD)
        if gamma_new @ delta_new > 1e-10:
            if len(deltas) == m_max:
                deltas.pop(0)
                gammas.pop(0)
            deltas.append(delta_new)
            gammas.append(gamma_new)
        x = x_new

        # update penalty parameter
        if count() + 1 <= n: # cost for c
            if np.max(c(x)) > 1e-3:
                rho = min(inc * rho, rho_max)
                deltas.clear()
                gammas.clear()

        if count() >= n:
            break
    return x_history, f_pen_history, f_history, pen_f_history, g_pen_history, g_history, pen_g_history

# ---- helpers ----
def line_search(f, x, d, grad, count, n, cost_per_eval=1, config=None):
    """
    Backtracking line search to find step size along direction `d`.
    Args:
        f (function): Function to be optimized
        x (np.array): current position
        d (np.array): search direction
        grad (np.array): gradient at current position
        count (function): function that returns current count of evaluations
        n (int): maximum number of evaluations allowed
        cost_per_eval (int): budget cost of one call to f
        config (dict): configuration dictionary for line search parameters
    Returns:
        alpha (float): step size found by line search
        f_val (float): function value at new point
        pen_val (float): penalty value at new point

    """
    n_searches = config['line_search']['n_searches']
    step = config['line_search']['step']
    beta = config['line_search']['beta']
    sigma = config['line_search']['sigma']

    if count() + cost_per_eval > n:
        return 0.0, np.zeros_like(x), 0.0, 0.0, 0.0

    f_res = f(x)
    # detect how many instances are in tuple output
    if isinstance(f_res, tuple):
        f_x_pen = f_res[0]
        f_x = f_res[1]
        pen_x = f_res[2]
    else:
        f_x_pen = f_res
        f_x = f_res
        pen_x = 0.0

    for _ in range(n_searches):
        x_new = x + step * d
        if count() + cost_per_eval > n:
            break

        f_res_new = f(x_new)
        if isinstance(f_res_new, tuple):
            f_new_pen = f_res_new[0]
            f_new = f_res_new[1]
            pen_new = f_res_new[2]
        else:
            f_new_pen = f_res_new
            f_new = f_res_new
            pen_new = 0.0

        # 
        if f_new_pen <= f_x_pen + sigma * step * grad @ d:
            x = x_new
            f_x_pen = f_new_pen
            f_x = f_new
            pen_x = pen_new
            break
        else:
            step *= beta

    return step, x, f_x_pen, f_x, pen_x