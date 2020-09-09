import time
import argparse
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

# internal
from utils.gauss_elim import gaussian_elimination
from utils.gauss_elim import gaussian_elimination_special_case
from utils.utils import tridiag, error

def u(x):
  """Differential equation to solve"""
  return 1 - (1-np.exp(-10))*x - np.exp(-10*x)

def f(x):
  """Approximation of u''(x)"""
  return 100 * np.exp(-10 * x)


def general_gauss_elim_experiment():
    print('1. Running general Gaussian Elimination experiments')
    ns = [10,100,1000]
    i = 1

    plt.figure(1,figsize=(9,6))
    plt.suptitle('Analytical and Numerical Solution for $u(x)$', y=1.002)
    for n in ns:
        h = 1/(n+1)
        x = np.linspace(0,1,n).reshape(-1,1)

        a = -1 * np.ones(n-1); b = 2 * np.ones(n); c = -1 * np.ones(n-1)
        A = tridiag(a, b, c)
        b = h**2 * f(x)

        print('n =',n)

        start = time.time()
        u_approx = gaussian_elimination(A,b)
        print('Took',time.time()-start, 'seconds.')

        u_closed_form = u(x)

        plt.subplot(2,2,i)
        plt.plot(x,u_closed_form, label='Closed-form')
        plt.plot(x,u_approx, label='Numerical')
        plt.title(f'n = {n}')
        plt.legend()
        plt.tight_layout()
        i+=1
    plt.savefig('results/fig-1b.png')

def special_gauss_elim_experiment():
    print('2. Running special case Gaussian Elimination experiments...')
    ns = [10,100,1000, int(10e6)]
    i = 1

    plt.figure(1,figsize=(9,6))
    plt.suptitle('Analytical and Numerical Solution for $u(x)$', y=1.002)
    for n in ns:
        # init
        h = 1/(n+1)
        x = np.linspace(0,1,n)

        b = h**2 * f(x)

        print('n =',n)
        start = time.time()
        u_approx = gaussian_elimination_special_case(b)
        print('Took',time.time()-start, 'seconds.')
        u_closed_form = u(x)

        plt.subplot(2,2,i)
        plt.plot(x,u_closed_form, label='Closed-form')
        plt.plot(x,u_approx, label='Numerical')
        plt.title(f'n = {n}')
        plt.legend()
        plt.tight_layout()
        i+=1
    plt.savefig('results/fig-1c.png')

def LU_experiment():
    print('3. Running special case Gaussian Elimination experiments...')
    ns = [10,100,1000]
    for n in ns:
        print('n =', n)
        h = 1/(n+1)
        x = np.linspace(0,1,n).reshape(-1,1)

        a = -1 * np.ones(n-1); b = 2 * np.ones(n); c = -1 * np.ones(n-1)
        A = tridiag(a, b, c)
        b = h**2 * f(x)
        start = time.time()
        fact = lu_factor(A)
        v_lu = lu_solve(fact, b)
        print('Took',time.time()-start, 'seconds.')


def error_curve_experiment():
    ns = 10**np.arange(1,6)
    max_epsilons = []
    for n in ns:
        print(f'n = {n}')
        h = 1/(n+1)
        x = np.linspace(0,1,n)
        b = h**2 * f(x)

        v = gaussian_elimination_special_case(b)

        epsilon = error(v[1:-1],u(x[1:-1]))

        max_error = np.max(epsilon)

        # print(f'max error = {max_error}')
        # plt.plot(x, u(x), label='u(x)')
        # plt.plot(x, v, label='v(x)')
        # plt.title(f'u and v for n = {n}')
        # plt.legend()
        # plt.grid()
        # plt.show()
        #
        # plt.plot(x[1:-1], epsilon, label='error')
        # plt.title(f'Error(u,v) for n = {n}')
        # plt.legend()
        # plt.grid()
        # plt.show()

        max_epsilons.append(max_error)

    plt.figure(1,figsize=(9,6))
    plt.plot(1/(ns+1), max_epsilons)
    plt.title('Relative error by step length `h`.')
    plt.xscale("log")
    plt.yscale('log')
    plt.ylabel('max(epsilon)')
    plt.xlabel('log(h)')
    plt.savefig('results/fig-1d.png')



def main():
    print('Started!')

    general_gauss_elim_experiment()

    special_gauss_elim_experiment()

    LU_experiment()

    error_curve_experiment()

    print('Fin.')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Some physics..')
    # parser.add_argument('-n', '--n_gridpoints', type=int,
    #                     help='an integer')
    # parser.add_argument('-o', '--output', type=str
    #                     help='the output file')
    #
    # args = parser.parse_args()
    #
    # main(args)
    main()
