#!/usr/bin/env python3

from numpy import arange, linspace, cos, sin, ndarray


def potential(
    x: ndarray, y: ndarray, n1: float, n2: float, phase: float,
    E0: float, Ecouple: float, E1: float
    ) -> ndarray:
    """Define the bare potential landscape."""
    return 0.5*(
        E0*(1-cos(n1*(x[:,None]-phase)))
        + Ecouple*(1-cos(x[:,None]-y[None,:]))
        + E1*(1-cos((n2*y[None,:])))
        )


def get_psi1(
    x: ndarray, y: ndarray, n1: float, n2: float, phase: float,
    E0: float, Ecouple: float, mu_Hp: float, zeta1: float
    ) -> ndarray:
    """Define the first component of the drift vector."""
    return (-1.0/zeta1)*((0.5)*(
        Ecouple*sin(x[:,None]-y[None,:])
        + (n1*E0*sin(n1*(x[:,None]-phase)))
        ) - mu_Hp)


def get_psi2(
    x: ndarray, y: ndarray, n1: float, n2: float,
    E1: float, Ecouple: float, mu_atp: float, zeta2: float
    ) -> ndarray:
    """Define the second component of the drift vector."""
    return (-1.0/zeta2)*((0.5)*(
        (-1.0)*Ecouple*sin(x[:,None]-y[None,:])
        + (n2*E1*sin(n2*y[None,:]))
        ) - mu_atp)