import sys
from dolfin import *
from dolfin_adjoint import *
from pyadjoint.placeholder import Placeholder
import numpy as np
import pandas as pd
import time
import os
from shutil import copyfile
from mpi4py import MPI
import matplotlib.pyplot as plt

# Set FEniCS parameters for optimization and numerical stability
parameters['linear_algebra_backend'] = 'PETSc'  # Use PETSc for efficient linear algebra
parameters['form_compiler']['representation'] = 'uflacs'  # Optimized representation for form compilation
parameters['form_compiler']['optimize'] = True  # Enable optimizations
parameters['form_compiler']['cpp_optimize'] = True  # Use C++ optimizations
parameters['form_compiler']['quadrature_degree'] = 3  # Quadrature degree for integration

# FFC options for further optimization
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

def main(B_m_max):
    """
    Main function to set up simulation parameters and initiate the solver.
    :param B_m_max: Maximum magnetic body force per unit volume.
    """
    cell = 'sphere'  # Define cell type for mesh loading

    # Simulation parameters
    params = {
        'chunks': 30,  # Number of time steps/chunks
        'newton_absolute_tolerance': 1e-10,  # Absolute tolerance for Newton solver
        'newton_relative_tolerance': 1e-9,  # Relative tolerance for Newton solver
        'mesh': "../geometry_data/" + cell + "/process.xdmf",  # Path to mesh file
        'domains': "../geometry_data/" + cell + "/process_domains.xdmf",  # Path to domain file
        'boundaries': "../geometry_data/" + cell + "/process_boundaries.xdmf",  # Path to boundaries file
        'mu_g': 7.5,  # Shear modulus for gel
        'mu_b': 1,  # Shear modulus for magnetic bead
        'nu_g': 0.49,  # Poisson's ratio for gel
        'nu_b': 0.499,  # Poisson's ratio for for magnetic bead
        'output_folder': "../output/"  # Output folder for results
    }

    # Call the solver with the parameters and maximum magnetic body force
    solver_call(params, B_m_max)

def solver_call(params, B_m_max):
    """
    Solver function for computing displacement under magnetic body force.
    :param params: Simulation parameters dictionary.
    :param B_m_max: Maximum magnetic body force per unit volume.
    """

    # MPI setup for parallel processing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create output directory if not present
    output_folder = params["output_folder"]
    if rank == 0 and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load mesh and domain/boundary data
    mesh = Mesh()
    with XDMFFile(params['mesh']) as infile:
        infile.read(mesh)

    domains = MeshFunction('size_t', mesh, mesh.topology().dim())
    with XDMFFile(params['domains']) as infile:
        infile.read(domains)

    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(params["boundaries"]) as infile:
        infile.read(mvc, "boundaries")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    dx = Measure("dx", domain=mesh, subdomain_data=domains)  # Volume integration
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)  # Surface integration

    '''
    Domain/volume labels:
    100: Gel
    200: Sphere

    Surface labels:
    101: Gel surface
    1: Sphere surface
    '''

    # Define finite element spaces
    V = VectorFunctionSpace(mesh, 'Lagrange', 1)  # Vector function space for displacement
    V0 = FunctionSpace(mesh, 'Lagrange', 1)  # Scalar function space
    V2 = TensorFunctionSpace(mesh, "Lagrange", degree=1, shape=(3, 3))  # Tensor function space

    # Initialize displacement-related functions
    u = Function(V)  # Displacement field
    u_ = TestFunction(V)  # Test function
    du = TrialFunction(V)  # Trial function

    # Kinematics
    B = Constant((0, 0, 0))  # External body force (default to zero)
    B_m = Constant((0, 0, 0))  # Magnetic body force (to be updated in chunks)
    T = Constant((0, 0, 0))  # Traction force on the boundary
    d = u.geometric_dimension()
    I = Identity(d)  # Identity tensor
    F = I + grad(u)  # Deformation gradient
    Ju = det(F)  # Jacobian determinant
    C = F.T * F  # Right Cauchy-Green tensor
    Ic = tr(C)  # First invariant of C
    C_bar = C / Ju ** (2 / d)  # Isochoric part of the tensor
    IC_bar = tr(C_bar)  # Invariant of isochoric part

    # Material parameters and strain energy densities
    mu_g, mu_b = params['mu_g'], params['mu_b']
    nu_g, nu_b = params['nu_g'], params['nu_b']
    lmbda_g = 2 * mu_g * nu_g / (1 - 2 * nu_g)
    lmbda_b = 2 * mu_b * nu_b / (1 - 2 * nu_b)
    c1_g, c2_g = mu_g / 2 * 1e-6, lmbda_g / 2 * 1e-6  # Gel coefficients
    c1_b, c2_b = mu_b / 2 * 1e-6, lmbda_b / 2 * 1e-6  # Magnetic bead coefficients

    psi_g = c1_g * (Ic - 3) - 2 * c1_g * ln(Ju) + c2_g * (ln(Ju)) ** 2  # Gel strain energy
    psi_b = c1_b * (Ic - 3) - 2 * c1_b * ln(Ju) + c2_b * (ln(Ju)) ** 2  # Magnetic bead strain energy

    # Define total potential energy
    Pi = (psi_g * dx(100) + psi_b * dx(200) - inner(B, u) * dx
          - inner(B_m, u) * dx(200) - inner(T, u) * ds + inner(grad(u), grad(u)) * dx(200)) # last term ensures body does not deform

    # Derivatives of the potential energy
    dPi = derivative(Pi, u, u_)  # First variation (residual)
    ddPi = derivative(dPi, u, du)  # Jacobian (tangent stiffness matrix)

    # Define Dirichlet boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    u_D = Expression(('0.', '0.', '0.'), degree=2)
    bc = DirichletBC(V, u_D, boundary)

    # Convert max magnetic body force to appropriate units
    B_m_max = B_m_max * 1e-9 * 1e6 / 47.71

    # Create output files
    simulation_output_file = XDMFFile(os.path.join(output_folder, "simulation_output.xdmf"))
    simulation_output_file.parameters["flush_output"] = True
    simulation_output_file.parameters["functions_share_mesh"] = True
    material_file = XDMFFile(os.path.join(output_folder, "domains_output.xdmf"))
    material_file.write(domains)

    # Solve the problem in chunks
    chunks = params['chunks']
    for i in range(chunks):
        if rank == 0:
            print(f'Chunk number = {i}')
            sys.stdout.flush()

        # Update magnetic body force incrementally
        B_m.assign(Constant(((i + 1) / chunks) * B_m_max, 0, 0))

        # Solve the variational problem
        solve(dPi == 0, u, bc, J=ddPi,
              solver_parameters={"newton_solver": {
                  "absolute_tolerance": params['newton_absolute_tolerance'],
                  "relative_tolerance": params['newton_relative_tolerance'],
                  "linear_solver": "gmres",
                  "preconditioner": "hypre_amg",
                  "maximum_iterations": 10000}},
              form_compiler_parameters=ffc_options)

        # Rename displacement field for output
        u.rename('u', 'displacement')

        # Write displacement to output file
        simulation_output_file.write(u, i)

# Runs script
if __name__ == '__main__':

    B_m_max = 10 # sets maximum magnetic bead force
    main(B_m_max) 
