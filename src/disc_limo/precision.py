# precision.py
# Thomas Hilder

from numpy.typing import NDArray
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import spsolve

from .regularisation import Λ_operator
from .setup import Setup


def dense_precision(setup: Setup) -> NDArray:
    # Get Λ
    Λ, _ = Λ_operator(setup.ω, setup.s, setup.λ)
    # If we have Cinv it's pretty easy
    if setup.Cinv is not None:
        return (setup.A.H @ setup.Cinv @ setup.A + Λ).todense()
    # If not, we need to do some more work
    # Setting Γ = A.H @ C^-1 implies Γ @ C = A.H, adjoint both sides gives C.H @ Γ.H = A
    C = csr_array(setup.C.todense())
    A = csc_array(setup.A.todense())
    Γ = spsolve(C.T, A).T
    # Now we get the precision matrix
    return Γ @ A + Λ.todense()
