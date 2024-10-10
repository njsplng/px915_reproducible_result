import numpy as np
from src.utils import *
from src.tangential_constitutive_matrix import (
    calculate_plane_strain,
    calculate_plane_stress,
)


class Structure:
    def __init__(
        self, ids, nodal_coordinates, degrees_of_freedom, connectivities
    ) -> None:
        self.elements = []

        ## Initialize elements with the inputs provided
        for element_id, element_dofs, element_conns in zip(
            ids, degrees_of_freedom, connectivities
        ):
            self.elements.append(
                Element(
                    element_id,
                    nodal_coordinates[element_conns],
                    element_dofs,
                    element_conns,
                )
            )

        ## Also save the global values for later use
        self.ids = ids
        self.nodal_coordinates = nodal_coordinates
        self.degrees_of_freedom = degrees_of_freedom
        self.connectivities = connectivities

    def calculate_K(self):
        """
        Calculates the global stiffness matrix for the structure
        """
        ## Initialize the global stiffness matrix
        self.K = np.zeros(
            (self.nodal_coordinates.shape[0] * 2, self.nodal_coordinates.shape[0] * 2)
        )

        ## Sum up over all the element stiffness matrices
        for element in self.elements:
            self.K[
                np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
            ] += element.K

    def solve_displacements(self, force_vector, boundary_conditions):
        """
        Solves for the displacements in the structure
        """
        ## Initialize the displacement vector
        self.displacement = np.zeros(self.K.shape[0])

        ## Apply the constraints
        self.free_dofs = np.setdiff1d(np.arange(self.K.shape[0]), boundary_conditions)

        ## Solve for the displacements
        free_K = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        free_displacement = np.linalg.solve(free_K, force_vector[self.free_dofs])
        self.displacement[self.free_dofs] = free_displacement


class Element:
    """
    Generic Quadrilateral Q4 Element implementation
    """

    def __init__(
        self,
        id_number,
        coordinates,
        degrees_of_freedom,
        connectivity,
        thickness=0.1,
        youngs_modulus=120e9,
        poisson_ratio=0.27,
        dimensionality=1,
    ):
        ## Set the provided values
        self.id_number = id_number
        self.coordinates = coordinates
        self.degrees_of_freedom = degrees_of_freedom
        self.connectivity = connectivity
        self.thickness = thickness
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

        ## Set the integration points
        if dimensionality == 1:
            self.integration_points = [
                IntegrationPoint1D(coordinates)
                for coordinates in INTEGRATION_POINT_COORDINATES_1D
            ]
        elif dimensionality == 2:
            self.integration_points = [
                IntegrationPoint2D(coordinates)
                for coordinates in INTEGRATION_POINT_COORDINATES_2D
            ]

        ## Calculate the element elasticity matrix
        # self.C = None
        # self.calculate_C()

    def calculate_B(self, derivatives: np.ndarray):
        """
        Returns the strain-displacement matrix at provided physical derivatives
        """
        ## Initialize B matrix
        B = np.zeros((3, 8))

        ## Fill B matrix
        B[0, 0::2] = derivatives[0]
        B[1, 1::2] = derivatives[1]
        B[2, 0::2] = derivatives[1]
        B[2, 1::2] = derivatives[0]

        return B

    def calculate_C(self):
        """
        Calculates the element elasticity matrix for plane stress configuration
        """
        self.C = np.array(
            [
                [
                    self.youngs_modulus / (1 - self.poisson_ratio**2),
                    self.poisson_ratio
                    * self.youngs_modulus
                    / (1 - self.poisson_ratio**2),
                    0,
                ],
                [
                    self.poisson_ratio
                    * self.youngs_modulus
                    / (1 - self.poisson_ratio**2),
                    self.youngs_modulus / (1 - self.poisson_ratio**2),
                    0,
                ],
                [0, 0, self.youngs_modulus / (2 * (1 + self.poisson_ratio))],
            ]
        )

    def calculate_K(self):
        """
        Calculates the stiffness matrix for the element
        """
        self.K = np.zeros((8, 8))
        for point in self.integration_points:
            ## Map the natural derivative to the physical derivative
            jacobian = point.dN @ self.coordinates

            # physical_derivatives = point.dN / jacobian
            ## if 2D:
            physical_derivatives = np.linalg.inv(jacobian) @ point.dN / jacobian

            ## Calculate the B matrix
            point.B = self.calculate_B(physical_derivatives)

            ## Calculate the volume at each gauss point
            ## 2D:
            point.volume = np.linalg.det(jacobian) * point.weight * self.thickness
            # point.volume = jacobian * point.weight * self.thickness

            ## Calculate the point contribution for the stiffness matrix
            point.K = point.B.T @ self.C @ point.B * point.volume

            ## Append to the matrix
            self.K += point.K


class IntegrationPoint2D:
    """
    Generic integration point set up for use with Legendre-Gauss quadrature of order 2
    """

    def __init__(self, coordinates):
        ## Set the default values
        self.coordinates = coordinates  # Assigned on input
        self.weight = 1

        ## To be calculated
        # self.strain = None
        # self.stress = None
        # self.internal_force = None

        ## Shape function-related values
        # self.N = None
        # self.dN = None
        ## Set the shape function values and derivatives at the integration point
        self.calculate_N()
        self.calculate_dN()

    def calculate_N(self):
        """
        Calculates the shape function values at the integration point
        """
        xi, eta = self.coordinates
        self.N = 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
            ]
        )

    def calculate_dN(self):
        """
        Calculates the shape function derivatives at the integration point
        """
        xi, eta = self.coordinates
        self.dN = 0.25 * np.array(
            [
                [-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)],
                [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi],
            ]
        )


class IntegrationPoint1D:
    """
    Generic integration point set up for use with Legendre-Gauss quadrature of order 2
    """

    def __init__(self, coordinates):
        ## Set the default values
        self.coordinates = coordinates  # Assigned on input
        self.weight = 1

        ## Set the shape function values and derivatives at the integration point
        self.calculate_N()
        self.calculate_dN()

    def calculate_N(self):
        """
        Calculates the shape function values at the integration point
        """
        xi = self.coordinates
        self.N = 0.5 * np.array(
            [
                1 - xi,
                1 + xi,
            ]
        )

    def calculate_dN(self):
        """
        Calculates the shape function derivatives at the integration point
        """
        self.dN = 0.5 * np.array([-1, 1])


## Store all the (4) values for the integration points in an array to feed into the element
INTEGRATION_POINT_COORDINATES_2D = np.array(
    [[-1, -1], [1, -1], [1, 1], [-1, 1]]
) / np.sqrt(3)
INTEGRATION_POINT_COORDINATES_1D = np.array([-1, 1]) / np.sqrt(3)


class StaggeredElement(Element):
    def __init__(
        self,
        id_number,
        coordinates,
        degrees_of_freedom,
        connectivity,
        thickness=1,
        youngs_modulus=210,
        poisson_ratio=0.3,
        G_c=None,
        l_0=None,
        H=None,
        k=None,
        body_force=None,
        mass_density=0,
    ) -> None:
        super().__init__(
            id_number,
            coordinates,
            degrees_of_freedom,
            connectivity,
            thickness,
            youngs_modulus,
            poisson_ratio,
            dimensionality=1,
        )

        ## Write some defaults
        if G_c is not None:
            self.G_c = G_c
        else:
            self.G_c = 0.0027

        if l_0 is not None:
            self.l_0 = l_0
        else:
            self.l_0 = 0.1

        if H is not None:
            for point in self.integration_points:
                point.H = H
        else:
            for point in self.integration_points:
                point.H = 0

        if k is not None:
            self.k = k
        else:
            self.k = 0

        ## Write some defaults
        self.lmbda = (
            poisson_ratio
            * youngs_modulus
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        self.mu = youngs_modulus / (2 * (1 + poisson_ratio))

        if body_force is not None:
            self.body_force = body_force
        else:
            self.body_force = 0

        self.mass_density = mass_density

        self.calculate_matrices_phasefield()
        self.calculate_matrices_displacement()

    def calculate_matrices_phasefield(self):
        self.K_pf = np.zeros((2, 2))
        self.F_pf = np.zeros(2)
        for point in self.integration_points:
            ## Map the natural derivative to the physical derivative
            jacobian = point.dN @ self.coordinates
            physical_derivatives = point.dN / jacobian

            ## Get the point volume
            point.volume = jacobian * point.weight * self.thickness
            ## Get terms
            term1 = (
                4
                * self.l_0**2
                * np.outer(physical_derivatives, physical_derivatives)
                * point.volume
            )
            term2 = (
                ((4 * self.l_0 * (1 - self.k) * point.H / self.G_c) + 1)
                * np.outer(point.N, point.N)
                * point.volume
            )

            ## Calculate the stiffness matrix at the point
            point.K_pf = term1 + term2

            ## Append the point stiffness matrix to the element stiffness matrix
            self.K_pf += point.K_pf

            ## Calculate the force
            self.F_pf += point.N * point.volume

    def calculate_matrices_displacement(self):
        self.K = np.zeros((2, 2))
        self.F = np.zeros(2)
        for point in self.integration_points:
            ## Calculate the constitutive matrix
            point.D = self.lmbda + 2 * self.mu
            ## If the strain is positive, multiply by the degradation function
            if hasattr(point, "strain_tp1") and point.strain_tp1 > 0:
                point.D *= point.g

            ## Map the natural derivative to the physical derivative
            jacobian = point.dN @ self.coordinates
            physical_derivatives = point.dN / jacobian

            ## Get the point volume
            point.volume = jacobian * point.weight * self.thickness

            ## Calculate the stiffness matrix at the point
            point.B = physical_derivatives
            point.K = (
                point.D
                * np.outer(physical_derivatives, physical_derivatives)
                * point.volume
            )

            ## Append the point stiffness matrix to the element stiffness matrix
            self.K += point.K

            ## Calculate the force
            self.F += point.N * self.body_force * point.volume

    def calculate_mass_matrix(self, matrix_type: str = "consistent") -> None:
        """
        Calculates the specified type mass matrix
        """
        ## Switch
        match matrix_type:
            case "consistent":
                ## Create a zero matrix to store the mass matrix
                self.M = np.zeros(
                    (len(self.degrees_of_freedom), len(self.degrees_of_freedom))
                )

                ## Add the mass for each point
                for i, point in enumerate(self.integration_points):
                    point.mass = self.mass_density * point.volume
                    self.M += point.mass * np.outer(point.N, point.N.T)

            case "lumped":
                ## Create a zero vector to store the mass matrix trace
                self.M = np.zeros((len(self.degrees_of_freedom), 1))

                ## Add the mass for each point
                for i, point in enumerate(self.integration_points):

                    point.mass = self.mass_density * point.volume

                    self.M += point.mass * point.N.reshape(-1, 1)

            ## Default case (error!)
            case _:
                raise ValueError("Matrix type not recognized")


class StaggeredStructure(Structure):
    def __init__(
        self,
        ids,
        nodal_coordinates,
        degrees_of_freedom,
        connectivities,
        body_force_nodal=None,
        G_c=0.0027,
        l_0=None,
        H=None,
        k=None,
        poisson_ratio=0.3,
        youngs_modulus=210,
        thickness=1,
        mass_density=0,
    ) -> None:
        self.elements = []

        ## Initialize elements with the inputs provided
        for element_id, element_dofs, element_conns in zip(
            ids, degrees_of_freedom, connectivities
        ):
            self.elements.append(
                StaggeredElement(
                    element_id,
                    nodal_coordinates[element_conns],
                    element_dofs,
                    element_conns,
                    body_force=body_force_nodal,
                    G_c=G_c,
                    l_0=l_0,
                    H=H,
                    k=k,
                    poisson_ratio=poisson_ratio,
                    youngs_modulus=youngs_modulus,
                    thickness=thickness,
                    mass_density=mass_density,
                )
            )

        ## Also save the global values for later use
        self.ids = ids
        self.nodal_coordinates = nodal_coordinates
        self.degrees_of_freedom = degrees_of_freedom
        self.connectivities = connectivities
        self.k = k

    def calculate_mass_matrix(self, matrix_type: str = "consistent") -> None:
        """
        Calculates the specified type mass matrix
        """
        for element in self.elements:
            element.calculate_mass_matrix(matrix_type)

        ## Switch
        match matrix_type:
            case "consistent":
                ## Create a zero matrix to store the mass matrix
                self.M = np.zeros(
                    (len(self.nodal_coordinates), len(self.nodal_coordinates))
                )

                ## Iterate through each element and add the mass to the matrix
                for element in self.elements:
                    self.M[
                        np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
                    ] += element.M

            case "lumped":
                ## Store the matrix in a trace to preserve memory
                self.M = np.zeros((len(self.nodal_coordinates), 1))

                ## Iterate through each element and add the mass to the trace
                for element in self.elements:
                    self.M[element.degrees_of_freedom] += element.M

                ## Expand into a normal matrix instead of a trace
                self.M = np.diag(np.reshape(self.M, len(self.nodal_coordinates)))

            ## Default case (error!)
            case _:
                raise ValueError("Matrix type not recognized")

    def calculate_damping_matrix(self, alpha: float = 0.9, beta: float = 0.02) -> None:
        """
        Calculates the damping matrix for the system, assuming Rayleigh damping
        """
        self.C = alpha * self.K + beta * self.M

    def calculate_matrices_phasefield(self):
        """
        Calculates the global stiffness matrix for the structure
        """
        ## Initialize the global stiffness matrix
        self.K_pf = np.zeros(
            (self.nodal_coordinates.shape[0], self.nodal_coordinates.shape[0])
        )
        ## Sum up over all the element stiffness matrices
        for element in self.elements:
            self.K_pf[
                np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
            ] += element.K_pf

        ## Get the force vector
        self.F_pf = np.zeros(self.nodal_coordinates.shape[0])
        ## Sum up over all the element force vectors
        for element in self.elements:
            self.F_pf[element.degrees_of_freedom] += element.F_pf

    def solve_phasefield(self, boundary_conditions=None):
        ## Initialize the displacement vector
        self.c = np.zeros(self.K_pf.shape[0])

        ## Apply the constraints
        self.free_dofs = np.setdiff1d(
            np.arange(self.K_pf.shape[0]), boundary_conditions
        )

        ## Solve for the displacements
        free_K = self.K_pf[np.ix_(self.free_dofs, self.free_dofs)]
        free_c = np.linalg.solve(free_K, self.F_pf[self.free_dofs])
        self.c[self.free_dofs] = free_c

    def post_process_phasefield(self):
        self.c_at_integration_points = []
        self.g = []
        for element in self.elements:
            for point in element.integration_points:
                point.c = np.dot(point.N, self.c[element.degrees_of_freedom])
                point.g = (1 - self.k) * point.c**2 - self.k
                self.c_at_integration_points.append(point.c)
                self.g.append(point.g)
        self.c_at_integration_points = np.array(self.c_at_integration_points)
        self.g = np.array(self.g)

    def calculate_stiffness_matrix_displacement(self):
        ## Initialize the global stiffness matrix
        self.K = np.zeros(
            (self.nodal_coordinates.shape[0], self.nodal_coordinates.shape[0])
        )
        ## Sum up over all the element stiffness matrices
        for element in self.elements:
            self.K[
                np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
            ] += element.K

    def calculate_force_vector_displacement(self):
        ## Get the force vector
        self.F = np.zeros(self.nodal_coordinates.shape[0])
        ## Sum up over all the element force vectors
        for element in self.elements:
            self.F[element.degrees_of_freedom] += element.F

    def calculate_matrices_displacement_raphson(self):
        ## Initialize the global stiffness matrix
        self.K = np.zeros(
            (self.nodal_coordinates.shape[0], self.nodal_coordinates.shape[0])
        )
        ## Sum up over all the element stiffness matrices
        for element in self.elements:
            self.K[
                np.ix_(element.degrees_of_freedom, element.degrees_of_freedom)
            ] += element.K

        ## Get the force vector
        self.F_raphson = self.F - self.internal_forces
        ## Sum up over all the element force vectors

    def solve_displacement_raphson(self, boundary_conditions=None):
        ## Initialize the displacement vector
        self.displacement = np.zeros(self.K.shape[0])

        ## Apply the constraints
        self.free_dofs = np.setdiff1d(np.arange(self.K.shape[0]), boundary_conditions)

        ## Solve for the displacements
        free_K = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        free_displacement = np.linalg.solve(free_K, self.F_raphson[self.free_dofs])
        self.displacement[self.free_dofs] = free_displacement

    def recalculate_element_matrices(self):
        for element in self.elements:
            element.calculate_matrices_phasefield()
            element.calculate_matrices_displacement()

    def calculate_residual(self):
        self.residual = np.linalg.norm(
            self.internal_forces[self.free_dofs] - self.F[self.free_dofs]
        )

    def initialise_internal_forces(self):
        # self.internal_forces = np.zeros(len(self.nodal_coordinates))
        self.internal_forces_incremental = np.zeros(len(self.nodal_coordinates))

    def solve_displacement_iterative(self, boundary_conditions):
        ## Initialize the displacement vector
        self.displacement_iterative = np.zeros(self.K.shape[0])

        ## Apply the constraints
        self.free_dofs = np.setdiff1d(np.arange(self.K.shape[0]), boundary_conditions)

        ## Solve for the displacements
        free_K = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        free_displacement_iterative = np.linalg.solve(
            free_K, -self.residual_incr[self.free_dofs]
        )
        self.displacement_iterative[self.free_dofs] = free_displacement_iterative


class StaggeredElement2D(Element):
    def __init__(
        self,
        id_number,
        coordinates,
        degrees_of_freedom,
        connectivity,
        thickness=100,
        youngs_modulus=210,
        poisson_ratio=0.3,
        G_c=0.0027,
        l_0=None,
        H=None,
        k=None,
        body_force_x=None,
        body_force_y=None,
        constitutive_matrix_mode="plane stress",
    ) -> None:
        super().__init__(
            id_number,
            coordinates,
            degrees_of_freedom,
            connectivity,
            thickness,
            youngs_modulus,
            poisson_ratio,
            dimensionality=2,
        )

        ## Write some defaults
        if G_c is not None:
            self.G_c = G_c
        else:
            self.G_c = 0.0027

        if l_0 is not None:
            self.l_0 = l_0
        else:
            self.l_0 = 0.1

        if H is not None:
            for point in self.integration_points:
                point.H = H
        else:
            for point in self.integration_points:
                point.H = 0

        if k is not None:
            self.k = k
        else:
            self.k = 0

        self.youngs_modulus = youngs_modulus

        self.poisson_ratio = poisson_ratio

        self.lmbda = (
            poisson_ratio
            * youngs_modulus
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        self.mu = youngs_modulus / (2 * (1 + poisson_ratio))

        if body_force_x is not None:
            self.body_force_x = body_force_x
        else:
            self.body_force_x = 0

        if body_force_y is not None:
            self.body_force_y = body_force_y
        else:
            self.body_force_y = 0

        self.constitutive_matrix_mode = constitutive_matrix_mode

        self.calculate_C()
        for point in self.integration_points:
            point.D = self.C

        self.calculate_matrices_phasefield()
        self.calculate_matrices_displacement()

    def calculate_B(self, derivatives):
        ## Extract the number of rows and columns and use that to form the B matrix
        rows = np.shape(derivatives)[0] + 1
        cols = np.shape(derivatives)[1] * 2
        B = np.zeros((rows, cols))

        ## Fill the B matrix (done for 3x8, might need something more generic in the future)
        B[0, 0::2] = derivatives[0]
        B[1, 1::2] = derivatives[1]
        B[2, 0::2] = derivatives[1]
        B[2, 1::2] = derivatives[0]

        return B

    def calculate_C(self):
        if self.constitutive_matrix_mode == "plane stress":
            self.C = calculate_plane_stress(self.youngs_modulus, self.poisson_ratio)
        elif self.constitutive_matrix_mode == "plane strain":
            self.C = calculate_plane_strain(self.youngs_modulus, self.poisson_ratio)
        else:
            raise ValueError("invalid constitutive matrix mode provided")

    def calculate_matrices_phasefield(self):
        self.K_pf = np.zeros((4, 4))
        self.F_pf = np.zeros(4)
        for point in self.integration_points:
            ## Map the natural derivative to the physical derivative
            jacobian = point.dN @ self.coordinates
            physical_derivatives = np.linalg.solve(jacobian, point.dN)

            ## Get the point volume
            point.volume = np.linalg.det(jacobian) * point.weight * self.thickness
            ## Get terms
            term1 = (
                4
                * self.l_0**2
                * physical_derivatives.T
                @ physical_derivatives
                * point.volume
            )
            term2 = (
                ((4 * self.l_0 * (1 - self.k) * point.H / self.G_c) + 1)
                * np.outer(point.N, point.N)
                * point.volume
            )

            ## Calculate the stiffness matrix at the point
            point.K_pf = term1 + term2

            ## Append the point stiffness matrix to the element stiffness matrix
            self.K_pf += point.K_pf

            ## Calculate the force
            self.F_pf += point.N * point.volume

    def calculate_matrices_displacement(self):
        self.K = np.zeros((8, 8))
        self.F = np.zeros(8)
        for point in self.integration_points:
            ## Map the natural derivative to the physical derivative
            point.jacobian = point.dN @ self.coordinates

            point.physical_derivatives = np.linalg.solve(point.jacobian, point.dN)

            point.B = self.calculate_B(point.physical_derivatives)

            ## Get the point volume
            point.jacobian_determinant = np.linalg.det(point.jacobian)
            point.volume = point.jacobian_determinant * point.weight * self.thickness

            ## Calculate the stiffness matrix at the point
            point.K = point.B.T @ point.D @ point.B * point.volume

            ## Append the point stiffness matrix to the element stiffness matrix
            self.K += point.K

            ## Calculate the force
            force_component_x = point.N * self.body_force_x * point.volume
            force_component_y = point.N * self.body_force_y * point.volume
            self.F += np.vstack((force_component_x, force_component_y)).flatten(
                order="F"
            )


class StaggeredStructure2D(Structure):
    def __init__(
        self,
        ids,
        nodal_coordinates,
        degrees_of_freedom,
        connectivities,
        body_force=None,
        G_c=8.9e-5,
        l_0=None,
        H=None,
        k=0,
        thickness=None,
        poisson_ratio=0.18,
        youngs_modulus=25.85,
        constitutive_matrix_mode="plane stress",
    ) -> None:
        self.elements = []

        ## Initialize elements with the inputs provided
        for element_id, element_dofs, element_conns in zip(
            ids, degrees_of_freedom, connectivities
        ):
            self.elements.append(
                StaggeredElement2D(
                    element_id,
                    nodal_coordinates[element_conns],
                    element_dofs,
                    element_conns,
                    body_force_x=body_force[0],
                    body_force_y=body_force[1],
                    constitutive_matrix_mode=constitutive_matrix_mode,
                    G_c=G_c,
                    l_0=l_0,
                    H=H,
                    k=k,
                    poisson_ratio=poisson_ratio,
                    youngs_modulus=youngs_modulus,
                    thickness=thickness,
                )
            )

        ## Also save the global values for later use
        self.ids = ids
        self.nodal_coordinates = nodal_coordinates
        self.degrees_of_freedom = degrees_of_freedom
        self.connectivities = connectivities
        self.k = k
