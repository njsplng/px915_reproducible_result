from tqdm import tqdm
import numpy as np
from src.solvers import *
from src.utils import *


def run_experiment(
    timesteps,
    element_count,
    lengthscale_parameter,
    max_control_displacement=0.06,
    loading_directions=[0, 1],
    domain=[0, 100],
):
    ## ----------------------------------------------------------------------------------------------- ##
    ## Global variables set up
    total_iterations = 0
    timesteps = timesteps
    max_raphson_steps = 20
    newton_raphson_tolerance = 1e-6
    diverged = False

    ## ----------------------------------------------------------------------------------------------- ##
    ## Solver mode set up
    solutions_modes = ["load control", "displacement control"]
    solver_mode = solutions_modes[1]

    ## Loading mode set up
    loading_modes = ["concentrated", "body"]
    loading_mode = loading_modes[1]

    ## ----------------------------------------------------------------------------------------------- ##
    ## Geometry set up
    # element_size = 1
    element_count = element_count
    domain = domain
    ids = np.arange(element_count)
    nodes = np.linspace(domain[0], domain[1], element_count + 1)
    connectivities = np.array([np.array([i, i + 1]) for i in range(element_count)])
    degrees_of_freedom = connectivities

    s = StaggeredStructure(
        ids,
        nodes,
        degrees_of_freedom,
        connectivities,
        poisson_ratio=0.18,
        youngs_modulus=25.85,
        body_force_nodal=1 / element_count,
        l_0=lengthscale_parameter,
        H=0,
        k=0,
        thickness=1,
        G_c=8.9e-5,
    )

    ## ----------------------------------------------------------------------------------------------- ##
    ## Tracker set up
    applied_total_force = []
    current_applied_total_force = 0
    forced_node_positions = []
    pinned_node_degradations = []
    fracture_energies = []

    ## ----------------------------------------------------------------------------------------------- ##
    ## Solution-specific variable set up
    if solver_mode == "load control":
        load_factor = np.linspace(0, 1, timesteps + 1)
        load_factor_reverse = load_factor[::-1]
        load_factor_combined = np.concatenate(
            (
                load_factor,
                load_factor_reverse[1:],
                -load_factor[1:],
                -load_factor_reverse[1:],
                load_factor[1:],
            )
        )
        load_factor = load_factor_combined
        timesteps = len(load_factor) - 1

    if solver_mode == "displacement control":
        total_control_displacement = max_control_displacement
        incremental_displacements = np.linspace(
            0, total_control_displacement, timesteps + 1
        )

        overall_loading = np.sum(np.abs(np.diff(loading_directions)))
        fractional_timestep = timesteps // overall_loading
        current_reference_maximum_position = 0
        current_timestep = 0
        for i in range(len(loading_directions) - 1):
            current_loading_step = loading_directions[i + 1] - loading_directions[i]
            next_increment = np.linspace(
                current_reference_maximum_position,
                current_reference_maximum_position + current_loading_step,
                int(abs(current_loading_step) * fractional_timestep),
            )
            incremental_displacements[
                current_timestep : current_timestep
                + int(abs(current_loading_step) * fractional_timestep)
            ] = next_increment
            current_reference_maximum_position += current_loading_step
            current_timestep += int(np.abs(current_loading_step) * fractional_timestep)
        incremental_displacements *= total_control_displacement
        incremental_displacements[-1] = total_control_displacement
        target_dof = -1

    ## ----------------------------------------------------------------------------------------------- ##
    ## Set the force vector
    if loading_mode == "concentrated":
        s.F = np.zeros(len(s.nodal_coordinates))
        s.F[-1] = 1

    elif loading_mode == "body":
        s.F = np.zeros(len(s.nodal_coordinates))
        for element in s.elements:
            s.F[element.degrees_of_freedom] += element.F

    ## ----------------------------------------------------------------------------------------------- ##
    ## Set up boundary conditions
    boundary_conditions = [0]

    ## Apply the constraints
    s.free_dofs = np.setdiff1d(
        np.arange(s.nodal_coordinates.shape[0]), boundary_conditions
    )

    ## ----------------------------------------------------------------------------------------------- ##
    ## Default values initialisation
    for element in s.elements:
        element.displacement_incremental = np.zeros(2)
        for point in element.integration_points:
            point.strain_tp1 = 0
            point.stress_tp1 = 0
            point.g = 1

    s.g = 1
    s.c_at_integration_points = 1
    s.displacement = np.zeros(len(s.nodal_coordinates))

    ## ----------------------------------------------------------------------------------------------- ##
    ## Time-stepping loop
    for t in tqdm(range(timesteps)):
        ## ----------------------------------------------------------------------------------------------- ##
        ## Update stress and strains for current time step
        for element in s.elements:
            for point in element.integration_points:
                point.strain_t = point.strain_tp1
                point.stress_t = point.stress_tp1

        ## Reset the displacement vector
        s.displacement_iterative = np.zeros(s.nodal_coordinates.shape[0])

        ## ----------------------------------------------------------------------------------------------- ##
        ## Update the solver mode-specific variables
        if solver_mode == "load control":
            ## Get the current incremental force according to the load factor
            s.F_incremental = s.F * (load_factor[t + 1] - load_factor[t])

            ## Initialise/reset start-of-loop variables
            s.internal_forces_incremental = np.zeros(len(s.nodal_coordinates))
            s.residual_iterative = s.internal_forces_incremental - s.F_incremental
            s.displacement_incremental = np.zeros(len(s.nodal_coordinates))

        if solver_mode == "displacement control":
            ## Initialise/reset start-of-loop variables
            s.displacement_incremental = np.zeros(len(s.nodal_coordinates))
            s.displacement_iterative_unfactored = np.zeros(len(s.nodal_coordinates))
            s.displacement_target_incremental = (
                incremental_displacements[t + 1] - incremental_displacements[t]
            )
            s.load_factor_incremental = 0

        ## ----------------------------------------------------------------------------------------------- ##
        ## Begin the Newton-Raphson loop
        j = 0
        while True:
            ## Increment the iteration counters
            total_iterations += 1
            j += 1
            if j > max_raphson_steps:
                print("NEWTON-RAPHSON DID NOT CONVERGE")
                diverged = True
                break

            ## Set the structure-level internal forces back to 0
            s.internal_forces_incremental = np.zeros(len(s.nodal_coordinates))

            ## ----------------------------------------------------------------------------------------------- ##
            ## Set up the displacement field stiffness matrix
            for element in s.elements:
                element.calculate_matrices_displacement()  # update the elemental-level K values

            ## Calculate the global stiffness matrix
            s.K = np.zeros((s.nodal_coordinates.shape[0], s.nodal_coordinates.shape[0]))
            for element in s.elements:
                s.K[
                    np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
                ] += element.K

            ## ----------------------------------------------------------------------------------------------- ##
            ## Solve for the displacements
            if solver_mode == "load control":
                s.displacement_iterative[s.free_dofs] = np.linalg.solve(
                    s.K[np.ix_(s.free_dofs, s.free_dofs)],
                    -s.residual_iterative[s.free_dofs],
                )

            if solver_mode == "displacement control":
                ## Get the iterative displacement for the full applied force
                s.displacement_iterative_unfactored[s.free_dofs] = np.linalg.solve(
                    s.K[np.ix_(s.free_dofs, s.free_dofs)], s.F[s.free_dofs]
                )

                if j == 1:
                    ## Get the load factor for the current iteration
                    s.load_factor_iterative = (
                        s.displacement_target_incremental
                        / s.displacement_iterative_unfactored[target_dof]
                    )

                    ## Increment the load factor
                    s.load_factor_incremental += s.load_factor_iterative

                    ## Factor the displacement and applied force
                    s.displacement_iterative[s.free_dofs] = (
                        s.displacement_iterative_unfactored[s.free_dofs]
                        * s.load_factor_incremental
                    )
                    s.F_incremental = s.F * s.load_factor_incremental
                else:
                    ## Find the displacement due to the residual forces
                    s.displacement_iterative[s.free_dofs] = np.linalg.solve(
                        s.K[np.ix_(s.free_dofs, s.free_dofs)],
                        -s.residual_iterative[s.free_dofs],
                    )

                    ## Get the load factor for the current iteration
                    s.load_factor_iterative = (
                        -s.displacement_iterative[target_dof]
                        / s.displacement_iterative_unfactored[target_dof]
                    )

                    ## Increment the load factor
                    s.load_factor_incremental += s.load_factor_iterative

                    ## Scale the incremental and iterative forces according to the load factor
                    s.F_incremental = s.F * s.load_factor_incremental
                    s.F_iterative = s.F * s.load_factor_iterative

                    ## Solve for the displacements with the adjusted forcing
                    s.displacement_iterative[s.free_dofs] = np.linalg.solve(
                        s.K[np.ix_(s.free_dofs, s.free_dofs)],
                        -s.residual_iterative[s.free_dofs] + s.F_iterative[s.free_dofs],
                    )

            s.displacement_incremental += s.displacement_iterative

            ## ----------------------------------------------------------------------------------------------- ##
            ## Calculate the strains, stresses and internal forces
            for element in s.elements:
                element.displacement_incremental = s.displacement_incremental[
                    element.degrees_of_freedom
                ]
                element.internal_forces_incremental = np.zeros(2)

                for point in element.integration_points:
                    ## Find the new strain
                    point.strain_incremental = (
                        point.B @ element.displacement_incremental
                    )
                    point.strain_tp1 = point.strain_t + point.strain_incremental

                    ## Split the strain
                    point.strain_plus = (
                        point.strain_tp1 + np.abs(point.strain_tp1)
                    ) / 2
                    point.strain_minus = (
                        point.strain_tp1 - np.abs(point.strain_tp1)
                    ) / 2

                    ## Calculate the stress
                    point.stress_plus_tp1 = (
                        element.lmbda * point.strain_plus
                        + 2 * element.mu * point.strain_plus
                    )
                    point.stress_minus_tp1 = (
                        element.lmbda * point.strain_minus
                        + 2 * element.mu * point.strain_minus
                    )
                    point.stress_tp1 = (
                        point.g * point.stress_plus_tp1 + point.stress_minus_tp1
                    )

                    ## Calculate the strain energy
                    point.strain_energy_plus = (
                        0.5 * element.lmbda * point.strain_plus**2
                        + element.mu * point.strain_plus**2
                    )
                    point.strain_energy_minus = (
                        0.5 * element.lmbda * point.strain_minus**2
                        + element.mu * point.strain_minus**2
                    )
                    point.strain_energy = (
                        point.g * point.strain_energy_plus + point.strain_energy_minus
                    )

                    ## Calculate the internal forces
                    point.internal_forces_incremental = (
                        point.B * (point.stress_tp1 - point.stress_t) * point.volume
                    )
                    element.internal_forces_incremental += (
                        point.internal_forces_incremental
                    )

                ## Add the internal forces to the structure
                s.internal_forces_incremental[
                    element.connectivity
                ] += element.internal_forces_incremental

            ## ----------------------------------------------------------------------------------------------- ##
            ## Residual calculation
            s.residual_iterative = s.internal_forces_incremental - s.F_incremental

            ## ----------------------------------------------------------------------------------------------- ##
            ## HYPLAS-style residual norm

            ## Get the reaction forces
            s.reactive_forces = np.zeros(len(s.nodal_coordinates))
            s.reactive_forces[s.free_dofs] = s.F_incremental[s.free_dofs]
            s.reactive_forces[boundary_conditions] = s.internal_forces_incremental[
                boundary_conditions
            ]

            ## Get the ratios
            residual_ratio = np.sum(s.residual_iterative[s.free_dofs] ** 2)
            reactive_forces_ratio = np.sum(s.reactive_forces[s.free_dofs] ** 2)
            maximum_residual = np.max(np.abs(s.residual_iterative[s.free_dofs]))
            residual_norm = np.sqrt(residual_ratio)
            reactive_forces_norm = np.sqrt(reactive_forces_ratio)

            ## Calculate the ratio
            if reactive_forces_norm == 0:
                s.ratio = 0
            else:
                s.ratio = 100 * residual_norm / reactive_forces_norm

            ## Check for convergence
            if s.ratio < newton_raphson_tolerance or np.abs(maximum_residual) < (
                newton_raphson_tolerance * 1e-3
            ):
                break

        ## ----------------------------------------------------------------------------------------------- ##
        ## In case of NR not converging, stop
        if diverged:
            break

        ## Increment the displacement
        s.displacement += s.displacement_incremental

        ## Tracking
        forced_node_positions.append(s.displacement[-1])
        current_applied_total_force += np.sum(s.F_incremental)
        applied_total_force.append(current_applied_total_force)

        ## ----------------------------------------------------------------------------------------------- ##
        ## Update the phase-field values
        for element in s.elements:
            for point in element.integration_points:
                ## Update the history field
                try:
                    max_positive_strain = float(point.strain_energy_plus.squeeze())
                except:
                    max_positive_strain = point.strain_energy_plus
                point.H = max(point.H, max_positive_strain)
            element.calculate_matrices_phasefield()

        ## Update stiffness matrix
        s.K_pf = np.zeros((s.nodal_coordinates.shape[0], s.nodal_coordinates.shape[0]))
        for element in s.elements:
            s.K_pf[
                np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
            ] += element.K_pf

        ## Update the phase-field force vector
        s.F_pf = np.zeros(len(s.nodal_coordinates))
        for element in s.elements:
            s.F_pf[element.degrees_of_freedom] += element.F_pf

        ## Calculate the phase-field values
        phase_field_boundary_conditions = []
        s.c = np.zeros(s.K_pf.shape[0])
        s.free_dofs_pf = np.setdiff1d(
            np.arange(s.K_pf.shape[0]), phase_field_boundary_conditions
        )
        s.c[s.free_dofs_pf] = np.linalg.solve(
            s.K_pf[np.ix_(s.free_dofs_pf, s.free_dofs_pf)], s.F_pf[s.free_dofs_pf]
        )

        ## Reset the current value arrays
        s.c_at_integration_points = []
        s.g = []

        ## Interpolate the phase field values at integration points
        for element in s.elements:
            element.internal_forces_incremental = np.zeros(2)
            for point in element.integration_points:
                point.c = np.dot(point.N, s.c[element.degrees_of_freedom])
                point.g = (1 - s.k) * point.c**2 - s.k
                s.c_at_integration_points.append(point.c)
                s.g.append(point.g)

        ## Parse into numpy arrays
        s.c_at_integration_points = np.array(s.c_at_integration_points)
        s.g = np.array(s.g)

        ## Tracking
        pinned_node_degradations.append(s.c_at_integration_points[0])

        fracture_energy = 0
        for element in s.elements:
            elemental_phasefield = s.c[element.degrees_of_freedom]
            for point in element.integration_points:
                c_at_gp = np.dot(point.N, elemental_phasefield)
                dc_at_gp = point.dN @ elemental_phasefield
                fracture_energy += (
                    element.G_c
                    * (
                        ((c_at_gp - 1) ** 2 / (4 * element.l_0))
                        + element.l_0 * dc_at_gp**2
                    )
                    * point.volume
                )
        fracture_energies.append(fracture_energy)

    return (
        np.array(applied_total_force),
        np.array(forced_node_positions),
        np.array(pinned_node_degradations),
        np.array(total_control_displacement),
        np.array(s.c),
        np.array(fracture_energies),
    )
