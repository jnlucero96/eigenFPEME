#!/usr/bin/env python3

from numpy import (
    zeros, eye, arange, pi, sin, dot, kron, linspace, cos, tan
    )
from scipy.linalg import toeplitz


def get_DN1(a=0.0, b=(2.0*pi), N=20):

    """Define the first derivative matrix."""

    if b > a:
        L = float(b-a)
    else:
        print("b must be greater than a")
        exit(1)

    h = L/N

    col = zeros(N); row = zeros(N)
    col[0] = 0.0; col[1:] = 0.5*((-1.0)**arange(1,N))/tan(arange(1,N)*h/2.0)
    row[0] = col[0]; row[1:] = col[N-1:0:-1]

    return toeplitz(col,row)



def get_DN2(a=0.0, b=(2.0*pi), N=30):

    """Define the second derivative matrix"""

    if b > a:
        L = float(b-a)
    else:
        print("b must be greater than a")
        exit(1)

    h = L/N; h_sq = h*h

    jj = arange(1,N)
    column = zeros(N)
    column[0] = -((pi**2)/(3*h_sq)+(1./6))
    column[1:] = -0.5*((-1)**jj)/(sin(h*0.5*jj)**2)

    return ((2*pi/L)**2)*toeplitz(column)


def get_laplacian(ax=0.0, bx=(2.0*pi), ay=0.0, by=(2.0*pi), N=20, M=30):

    DN2x = get_DN2(ax, by, N); Ix = eye(N)
    DN2y = get_DN2(ay, by, M); Iy = eye(M)

    return kron(DN2x, Iy) + kron(Ix, DN2y)
