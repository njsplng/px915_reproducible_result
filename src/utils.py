import numpy as np


def print_structure_level_data(obj, excluded_keys=[], precision=5, linewidth=200):
    """
    Print the data of the object at the structure level.
    """
    ## Set custom print options
    np.set_printoptions(precision=precision, linewidth=linewidth)

    ## Initialize the dict where the data will be stored
    print_dict = vars(obj)

    ## Sort the dict keys
    print_dict = dict(sorted(print_dict.items()))

    ## Print the dict
    for key, value in print_dict.items():
        if key in excluded_keys:
            continue
        print(f"{key}: {value}")


def print_integration_point_level_data(
    obj, excluded_keys=[], precision=5, linewidth=200
):
    """
    Prints the data at the structure's integration points level.
    """
    ## Set custom print options
    np.set_printoptions(precision=precision, linewidth=linewidth)

    ## Initialize the dict where the data will be stored
    print_dict = {}
    for element in obj.elements:
        for point in element.integration_points:
            for key, value in vars(point).items():
                if key in excluded_keys:
                    continue
                if key not in print_dict:
                    print_dict[key] = []

    ## Fill in the dict
    for element in obj.elements:
        for point in element.integration_points:
            for key, value in vars(point).items():
                if key in excluded_keys:
                    continue
                print_dict[key].append(value)

    ## Sort the dict keys
    print_dict = dict(sorted(print_dict.items()))

    ## Print the dict
    for key, value in print_dict.items():
        print(f"{key}: {np.array(value)}")


def macaulay_brackets(x):
    return np.maximum(x, 0)


def convert_voigt_to_matrix_2d(x):
    return np.array([[x[0], x[2]], [x[2], x[1]]])


def convert_matrix_2d_to_voigt(x):
    return np.array([x[0, 0], x[1, 1], x[0, 1]])


def convert_voigt_to_matrix_3d(x):
    return np.array([[x[0], x[5], x[4]], [x[5], x[1], x[3]], [x[4], x[3], x[2]]])


def convert_matrix_3d_to_voigt(x):
    return np.array([x[0, 0], x[1, 1], x[2, 2], x[1, 2], x[0, 2], x[0, 1]])


def macaulay_brackets_plus_miehe(x):
    return (x + np.abs(x)) / 2


def macaulay_brackets_minus_miehe(x):
    return (x - np.abs(x)) / 2


def convert_strain_to_tensor(x):
    return np.array(
        [[x[0], x[2] / 2, x[4]], [x[2] / 2, x[1], x[3]], [x[4], x[3], x[5]]]
    )


def flatten_structure_object(structure):
    properties_dict = {}
    for element in structure.elements:
        for point in element.integration_points:
            for key, value in vars(point).items():
                key = "ip_" + key
                if key not in properties_dict:
                    properties_dict[key] = value
                    properties_dict[f"{key}_shape"] = np.array(value).shape
                properties_dict[key] = np.hstack((properties_dict[key], value))

        for key, value in vars(element).items():
            key = "el_" + key
            if key not in properties_dict:
                properties_dict[key] = value
                properties_dict[f"{key}_shape"] = np.array(value).shape
            properties_dict[key] = np.hstack((properties_dict[key], value))

    for key, value in vars(structure).items():
        key = "st_" + key
        properties_dict[key] = value
        properties_dict[f"{key}_shape"] = np.array(value).shape

    return properties_dict


def replace_placeholders(data, placeholders):
    if isinstance(data, dict):
        return {k: replace_placeholders(v, placeholders) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(v, placeholders) for v in data]
    elif isinstance(data, str):
        for placeholder, value in placeholders.items():
            data = data.replace(placeholder, str(value))
        return eval(data) if "*" in data else data
    else:
        return data


import subprocess
import sys


def setup_fortran_link(recompile=True):
    """
    Rebuild the fortran code and add the f90 directory to the python path
    """
    if recompile:
        subprocess.call(["cd f90 && make &> /dev/null"], shell=True)
    sys.path.append("f90")


def generate_fortran_subroutine(template_name, arrays, scalars=None):
    if scalars is not None:
        variable_declaration = ", ".join(
            [f"{arr}, {arr}_dim_x, {arr}_dim_y" for arr in arrays]
            + [f"{scalar}" for scalar in scalars]
        )
    else:
        variable_declaration = ", ".join(
            [f"{arr}, {arr}_dim_x, {arr}_dim_y" for arr in arrays]
        )

    subroutine_template = f"subroutine {template_name}({variable_declaration})\n"

    for arr in arrays:
        subroutine_template += f"    ! {arr} declaration\n"
        subroutine_template += f"    integer, intent(in) :: {arr}_dim_x, {arr}_dim_y\n"
        subroutine_template += (
            f"    real(dp), intent(inout) :: {arr}({arr}_dim_x, {arr}_dim_y)\n\n"
        )
    if scalars is not None:
        subroutine_template += "    ! Scalar variable declaration\n"
        for scalar in scalars:
            subroutine_template += f"    integer, intent(inout) :: {scalar}\n"
        subroutine_template += "\n"
    subroutine_template += "    ! Loop variables\n"
    subroutine_template += "    integer :: i, j, start_index, end_index\n\n"
    subroutine_template += "    ! code goes here\n\n"
    subroutine_template += f"end subroutine {template_name}\n"

    return subroutine_template.strip()
