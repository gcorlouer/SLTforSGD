from fipy import Variable, FaceVariable, CellVariable, Grid1D, PeriodicGrid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer, ExponentialConvectionTerm
from fipy.tools import numerix
import sympy as sym
from sympy.vector import CoordSys3D
from sympy.utilities.lambdify import lambdastr, implemented_function, lambdify
import numpy as np

def get_coeffs(q, print_func=False):
    """
    Return the convection and diffusion coefficients (eq 12) as sympy objects

    q: function of w (learned variable)
    Can optionally be a function of:
    
    w0: fixed, free parameter
    v: variance of the x variable
    ve: variance of the noise term
    lr: learning rate
    b: batch size
    """
    w, w0, v, ve, l, b = sym.symbols('w w0 v ve lr b')
    N = CoordSys3D('N')
    grad_q = sym.diff(q, w)
    h = -l*v*q*grad_q
    g = (l/sym.sqrt(b))*(-v*q*grad_q*N.i + sym.sqrt(v*ve/2)*grad_q*N.j)
    d1 = h + g.dot(sym.diff(g, w))
    d2 = g.dot(g)
    convection = -d1 + sym.diff(d2, w)
    if print_func:
        display("convection term:", sym.factor(convection))
        display("diffusion term:", d2)
    return lambdify((l, v, ve, w0, b, w), convection), lambdify((l, v, ve, w0, b, w), d2)

def run(q, nw, lr, v, ve, w0, b, L, dt, steps, w_left, w_right, zero_prob_left=False, zero_prob_right=False, pbc=False):
    """
    Numerically solve the 1d Fokker-Planck equation using the fipy package

    nw: number of numerical cells
    lr: learning rate
    v: variance of the x variable
    ve: variance of the noise term
    w0: fixed, free parameter
    b: batch size
    dt: time step size
    steps: number of time steps
    w_left: left end of the interval over which w isn't initially zero
    w_right: right end of the inverval over which w isn't initially zero
    zero_prob_left: if True, zero probability density on the left boundary, if False, zero derivative
    zero_prob_right: same as above, on the right boundary
    pbc: whether to use periodic boundary conditions

    NOTE: I removed the L parameter
    """
    
    convection, d2 = get_coeffs(q)

    if pbc:
        mesh = PeriodicGrid1D(nx=nw, dx=1./nw)
    else:
        mesh = Grid1D(nx=nw, Lx=L)
    
    P = CellVariable(mesh=mesh, name=r"$P$")
    P.value = 0.
    w = mesh.cellCenters[0]

    def creneau(w, left, right):
        """
        Return 0 if w < left or w > right, return a constant value otherwise
        """
        k = 1./(2.*(right-left))
        return np.where(w < left, -k, k) + np.where(w > right, -k, k)
    
    P.setValue(creneau(w, w_left, w_right))

    if w_left < 0 and w_right > 0 and pbc:
        w_left = w_left % 1.
        P.setValue( .5*(creneau(w, 0., w_right) + creneau(w, w_left, 1.)) )

    if zero_prob_left and not(pbc):
        # set zero probability on the left end of the simulation interval
        P.constrain(0., where=mesh.facesLeft)
    if zero_prob_right and not(pbc):
        # set zero probability on the left end of the simulation interval
        P.constrain(0., where=mesh.facesRight)

    w = mesh.cellCenters[0]
    D2 = d2(lr, v, ve, w0, b, w)
    
    w = mesh.faceCenters[0]
    minusD1plusgradD2 = convection(lr, v, ve, w0, b, w)
    # unit vector
    u = FaceVariable(mesh=mesh, value = 1., rank=1)
    
    # implicit scheme
    eq = TransientTerm() == DiffusionTerm(CellVariable(mesh=mesh, value = D2)) + ExponentialConvectionTerm(u * FaceVariable(mesh=mesh, value = minusD1plusgradD2))
    # explicit scheme: numerically unstable unless dt is small enough compared to spatial discretization
    # eq = TransientTerm() == ExplicitDiffusionTerm(CellVariable(mesh=mesh, value = D2)) + ExponentialConvectionTerm(u * FaceVariable(mesh=mesh, value = minusD1plusgradD2))

    Ps = [P.copy()]
    for step in range(steps):
        eq.solve(var=P, dt=dt)
        Ps.append(P.copy())

    return mesh.x, Ps