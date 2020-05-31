#!/usr/bin/env python3

from numpy import (
    zeros, pi, finfo, linspace, exp, dot, eye, kron, ones, sort,
    array, where
    )
from numpy.linalg import eig, eigvals
from scipy.linalg import toeplitz, expm
from numpy.fft import fft2

from datetime import datetime

from energetics import potential, get_psi1, get_psi2
from derivatives import get_DN1, get_DN2, get_laplacian

from pltconfig import *


def get_params():

    """Specify parameters of simulation here."""

    # discretization parameters
    N = 50  # inverse space discretization. Keep this number high!

    # model constants
    beta = 1.0  # thermodynamic beta: 1/kT
    m1 = m2 = 1.0  # masses of Fo and F1

    # model-specific parameters
    gamma1 = gamma2 = 1000.0  # drag coefficients of Fo and F1

    E0 = 3.0 # energy scale of Fo
    Ecouple = 3.0 # energy scale of coupling between Fo and F1
    E1 = 3.0 # energy scale of F1
    mu_Hp = 0.0 #  mu_{H+}: energy INTO (positive) Fo by F1
    mu_atp = 0.0 # mu_{ATP}: energy INTO (positive) F1 by Fo

    n1 = 3.0  # number of minima in the potential of Fo
    n2 = 3.0  # number of minima in the potential of F1
    phase = 0.0  # how much sub-systems are offset from one another

    # specify full path to where simulation results are output
    data_dir = '/Users/jlucero/data_dir/2020-05-27/'

    return (
        N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, mu_Hp, mu_atp, data_dir
    )


def save_data_reference(
    n1, n2, phase,
    E0, Ecouple, E1, mu_Hp, mu_atp, p_now, p_equil,
    potential_at_pos, drift_at_pos, diffusion_at_pos,
    N, data_dir
    ):

    """Helper function that writes results of simulation to file."""

    data_filename = (
        f"/reference_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_"
        + f"psi1_{mu_Hp}_psi2_{mu_atp}_"
        + f"n1_{n1}_n2_{n2}_phase_{phase}_"
        + "outfile.dat"
    ) #TODO: Consult with Emma on filenames. psi -> mu.

    data_total_path = data_dir + data_filename

    with open(data_total_path, 'w') as dfile:
        for i in range(N):
            for j in range(N):
                dfile.write(
                    f"{p_now[i, j]:.15e}\t"
                    + f"{p_equil[i, j]:.15e}\t"
                    + f"{potential_at_pos[i, j]:.15e}\t"
                    + f"{drift_at_pos[0, i, j]:.15e}\t"
                    + f"{drift_at_pos[1, i, j]:.15e}\t"
                    + f"{diffusion_at_pos[0, i, j]:.15e}\t"
                    + f"{diffusion_at_pos[1, i, j]:.15e}\t"
                    + f"{diffusion_at_pos[2, i, j]:.15e}\t"
                    + f"{diffusion_at_pos[3, i, j]:.15e}\n"
                )


def main():

    # unload parameters
    [
        N, gamma1, gamma2, beta, m1, m2, n1, n2,
        phase, E0, E1, Ecouple, mu_Hp, mu_atp, data_dir
    ] = get_params()

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

    fpe_operator = (
        Dx*drift_at_pos[0,...].flatten()
        + Dy*drift_at_pos[1,...].T.flatten()
        + (1.0/(m1*gamma1))*lapl
    )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Finding the steady-state..."
    )

    D, U = eig(fpe_operator)

    p_now = array(
        U[:,where(abs(D)<=finfo("float32").eps)[0][0]]
    ).real
    p_now /= p_now.sum()
    p_now.shape = (N,N)

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Steady-state found!"
    )

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Processing data..."
    )

    # checks to make sure nothing went weird: bail at first sign of trouble
    # check the non-negativity of the distribution
    assert (p_now >= 0.0).all(), \
        "ABORT: Probability density has negative values!"
    # check the normalization
    assert (abs(p_now.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
        "ABORT: Probability density is not normalized!"

    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
        + "Processing finished!"
    )

    # write to file
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data..."
    )
    save_data_reference(
        n1, n2, phase,
        E0, Ecouple, E1, mu_Hp, mu_atp, p_now, p_equil,
        potential_at_pos, drift_at_pos, diffusion_at_pos, N,
        data_dir
    )
    print(
        f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!"
    )

    print("Exiting...")

if __name__ == "__main__":
    main()
