#!/usr/bin/env python3

from numpy import (
    array, kron, pi, linspace, exp, dot, eye, ones, where, zeros, finfo,
    sin, cos
    )
from numpy.linalg import eig
from scipy.linalg import toeplitz, expm
from numpy.fft import fft2, fftshift

from datetime import datetime

from energetics import potential, get_psi1, get_psi2
from derivatives import get_DN1, get_DN2, get_laplacian


def get_tests_params():

    N = 50  # inverse space discretization. Keep this number high!

    # model constants
    beta = 1.0  # thermodynamic beta: 1/kT
    m1 = m2 = 1.0  # masses of Fo and F1

    # model-specific parameters
    gamma1 = gamma2 = 1000.0  # drag coefficients of Fo and F1

    E0 = 2.0 # energy scale of Fo
    Ecouple = 0.0 # energy scale of coupling between Fo and F1
    E1 = 2.0 # energy scale of F1
    mu_Hp = 0.0 #  mu_{H+}: energy INTO (positive) Fo by F1
    mu_atp = 0.0 # mu_{ATP}: energy INTO (positive) F1 by Fo

    n1 = 3.0  # number of minima in the potential of Fo
    n2 = 3.0  # number of minima in the potential of F1
    phase = 0.0  # how much sub-systems are offset from one another

    return (
        N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, mu_Hp, mu_atp
    )

def test_first_derivative(args=None):

    a = 0.0
    b = 2.0*pi
    N = 20
    M = 30

    L = (b-a)
    dx = L/N
    dy = L/M

    x = linspace(a, b-dx, N)
    y = linspace(a, b-dy, M)
    z = sin(2.0*x[:,None])*cos(y[None,:])

    pzpx = 2.0*cos(2.0*x[:,None])*cos(y[None,:])
    pzpy = -sin(2.0*x[:,None])*sin(y[None,:])

    # First derivative matrices
    DN1x = get_DN1(a, b, N); Ix = eye(N)
    DN1y = get_DN1(a, b, M); Iy = eye(M)
    Dx = kron(DN1x, Iy); Dy = kron(Ix, DN1y)

    # associated errors
    pzpx_num = dot(Dx, z.flatten())
    pzpy_num = dot(Dy, z.flatten())
    xerr = (pzpx_num.reshape((N,M))-pzpx).__abs__().max()
    xerr_loc = (pzpx_num.reshape((N,M))-pzpx).__abs__().argmax()
    yerr = (pzpy_num.reshape((N,M))-pzpy).__abs__().max()
    yerr_loc = (pzpy_num.reshape((N,M))-pzpy).__abs__().argmax()

    print(xerr, xerr_loc, yerr, yerr_loc)


def test_laplacian(args=None):

    a = 0.0
    b = 2.0*pi
    N = 20
    M = 30

    L = (b-a)
    dx = L/N
    dy = L/M

    x = linspace(a, b-dx, N)
    y = linspace(a, b-dy, M)
    z = sin(2.0*x[:,None])*cos(y[None,:])

    laplz = -5.0*cos(y[None,:])*sin(2.0*x[:,None])

    # First derivative matrices
    lapl_num = get_laplacian(ax=a, bx=b, ay=a, by=b, N=N, M=M)

    # associated errors
    lapl_error = (dot(lapl_num, z.flatten()).reshape((N,M))-laplz).__abs__().max()
    lapl_error_loc = (dot(lapl_num, z.flatten()).reshape((N,M))-laplz).__abs__().argmax()

    print(lapl_error, lapl_error_loc)


def test_operator():

    (
        N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, mu_Hp, mu_atp
        ) = get_tests_params()

    # calculate derived discretization parameters
    dx = (2*pi) / N  # space discretization: total distance / number of points

    print("Setting up...")

    positions = linspace(0, (2*pi)-dx, N)

    potential_at_pos = potential(
        positions, positions, n1, n2, phase, E0, Ecouple, E1
        )

    # define equilibrium probability
    p_equil = exp(-potential_at_pos)
    p_equil /= p_equil.sum(axis=None)

    # defininig the drift vector
    drift_at_pos = zeros((2, N, N))
    drift_at_pos[0,...] = fft2(get_psi1(
        positions, positions, n1, n2, phase, E0, Ecouple, mu_Hp, m1*gamma1
        ))
    drift_at_pos[1,...] = fft2(get_psi2(
        positions, positions, n1, n2, E1, Ecouple, mu_atp, m2*gamma2
        ))

    # define the diffusion tensor
    diffusion_at_pos = zeros((4, N, N))
    diffusion_at_pos[0,...] = 1.0/(beta*m1*gamma1)
    diffusion_at_pos[3,...] = 1.0/(beta*m2*gamma2)

    # define the fokker-planck operator
    # get necessary derivative matrices
    II = eye(N); DN1 = get_DN1(0.0, 2.0*pi, N); DN2 = get_DN2(0.0, 2.0*pi, N)
    Dx = kron(DN1, II); Dy = kron(II, DN1)
    lapl = get_laplacian(N=N, M=N)

    # working definition of the Fokker-Planck operator
    fpe_operator = (
        Dx*drift_at_pos[0,...].flatten()
        + Dy*drift_at_pos[1,...].T.flatten()
        + (1.0/(m1*gamma1))*lapl
    )

    print(f"Operator shape = {fpe_operator.shape}")

    # ensure that transition probability matrix preserves norm
    T = expm(fpe_operator)
    norm_flag_thrown = False
    for i in range(10):
        Z = (dot(T, p_equil.flatten())).sum(axis=None)
        if (abs(Z-1.0) > finfo("float32").eps):
            norm_flag_thrown = True
            print(f"Normalization not preserved: {Z}")
            break
        T = dot(T, T)

    if not norm_flag_thrown:
        print("Norm test passed!")
    else:
        exit(1)

    # check the error on the fpe_operator
    D, U = eig(fpe_operator)

    p_now = array(
        U[:,where(abs(D)<=finfo("float32").eps)[0][0]]
    ).__abs__()
    p_now /= p_now.sum()
    p_now.shape = (N,N)

    print(f"Steady-state inf-norm error = {(p_now-p_equil).__abs__().max():.10e}")


if __name__ == "__main__":
    test_operator()