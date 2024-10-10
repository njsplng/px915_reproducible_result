import numpy as np
import warnings


def compute_elastic_tangent_constitutive_matrix(
    lmbda, mu, E, nu, g, strain, mode="plane strain"
):
    if mode == "plane strain":
        return damage_plane_strain(lmbda, mu, E, nu, g, strain)
    elif mode == "plane stress":
        return damage_plane_stress(lmbda, mu, E, nu, g, strain)
    else:
        print("Unrecognized mode")
        return -1


def damage_plane_strain(lmbda, mu, E, nu, g, strain):
    warnings.simplefilter("ignore")

    C1111 = C1111_term(g, lmbda, mu, nu, strain, 1)
    C1122 = C1122_term(g, lmbda, mu, nu, strain, 1)
    C1112 = C1112_term(g, lmbda, mu, nu, strain, 1)
    C2211 = C2211_term(g, lmbda, mu, nu, strain, 1)
    C2222 = C2222_term(g, lmbda, mu, nu, strain, 1)
    C2212 = C2212_term(g, lmbda, mu, nu, strain, 1)
    C1211 = C1211_term(g, lmbda, mu, nu, strain, 1)
    C1222 = C1222_term(g, lmbda, mu, nu, strain, 1)
    C1212 = C1212_term(g, lmbda, mu, nu, strain, 1)

    D = np.array([[C1111, C1122, C1112], [C2211, C2222, C2212], [C1211, C1222, C1212]], dtype=np.float64, order='F')
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.abs(D[i, j]) < 1e-12:
                D[i, j] = 0

    D_inf = D == np.inf
    D_minf = D == -np.inf
    D_nan = np.isnan(D)
    D_zero = np.abs(np.diagonal(D)) == 0

    if np.sum(D_inf) + np.sum(D_minf) + np.sum(D_nan) + np.sum(D_zero) > 0:
        print(
            f"Recalculating to default constitutive matrix. Stats: {np.sum(D_inf), np.sum(D_minf), np.sum(D_nan), np.sum(D_zero)}. Diagonal elements: {np.diagonal(D)}"
        )
        D = calculate_plane_strain(E, nu)

    return D


def damage_plane_stress(lmbda, mu, E, nu, g, strain):
    warnings.simplefilter("ignore")

    C1111 = C1111_term(g, lmbda, mu, nu, strain, 2)
    C1122 = C1122_term(g, lmbda, mu, nu, strain, 2)
    C1112 = C1112_term(g, lmbda, mu, nu, strain, 2)
    C2211 = C2211_term(g, lmbda, mu, nu, strain, 2)
    C2222 = C2222_term(g, lmbda, mu, nu, strain, 2)
    C2212 = C2212_term(g, lmbda, mu, nu, strain, 2)
    C1211 = C1211_term(g, lmbda, mu, nu, strain, 2)
    C1222 = C1222_term(g, lmbda, mu, nu, strain, 2)
    C1212 = C1212_term(g, lmbda, mu, nu, strain, 2)

    D = np.array([[C1111, C1122, C1112], [C2211, C2222, C2212], [C1211, C1222, C1212]], dtype=np.float64, order='F')

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.abs(D[i, j]) < 1e-12:
                D[i, j] = 0

    D_inf = D == np.inf
    D_minf = D == -np.inf
    D_nan = np.isnan(D)
    D_zero = np.abs(np.diagonal(D)) == 0

    if np.sum(D_inf) + np.sum(D_minf) + np.sum(D_nan) + np.sum(D_zero) > 0:
        print(
            f"Recalculating to default constitutive matrix. Stats: {np.sum(D_inf), np.sum(D_minf), np.sum(D_nan), np.sum(D_zero)}. Diagonal elements: {np.diagonal(D)}"
        )
        D = calculate_plane_stress(E, nu)

    return D


def C1111_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = absDerFun(1, strain[1] + strain[0])
        t5 = strain[0] ** 2
        t8 = (strain[2] / 2) ** 2
        t10 = strain[1] ** 2
        t12 = np.sqrt((-2 * strain[1] * strain[0] + t10 + t5 + 4 * t8))
        t13 = strain[1] - strain[0] + t12
        t15 = 0.2e1 / t13 * (strain[2] / 2)
        t16 = abs(t15)
        t17 = t16**2
        t18 = 0.1e1 + t17
        t19 = t18**2
        t21 = t8 * (strain[2] / 2)
        t23 = t13**2 / 0.4e1
        t24 = t23**2
        t26 = 0.1e1 / t24 * t21 / t19
        t27 = strain[1] + strain[0] + t12
        t28 = 0.0e0 < t27 / 0.2e1
        t29 = t27 / 0.2e1 <= 0.0e0
        t30 = piecewiseFun(t28, t27 / 0.2e1, t29, 0)
        t32 = absDerFun(1, t15)
        t36 = (strain[0] - strain[1]) / t12 / 0.2e1
        t37 = -0.1e1 / 0.2e1 + t36
        t38 = t37 * t32
        t43 = t8 / t18
        t46 = 0.2e1 / t23 / t13
        t51 = 0.1e1 / t23
        t52 = 0.1e1 / 0.2e1 + t36
        t53 = piecewiseFun(t28, t52, t29, 0)
        t56 = strain[1] - strain[0] - t12
        t58 = 0.2e1 / t56 * (strain[2] / 2)
        t59 = abs(t58)
        t60 = t59**2
        t61 = 0.1e1 + t60
        t62 = t61**2
        t65 = t56**2 / 0.4e1
        t66 = t65**2
        t68 = 0.1e1 / t66 * t21 / t62
        t69 = strain[1] + strain[0] - t12
        t70 = 0.0e0 < t69 / 0.2e1
        t71 = t69 / 0.2e1 <= 0.0e0
        t72 = piecewiseFun(t70, t69 / 0.2e1, t71, 0)
        t74 = absDerFun(1, t58)
        t75 = -t52 * t74
        t80 = t8 / t61
        t83 = 0.2e1 / t65 / t56
        t88 = 0.1e1 / t65
        t89 = piecewiseFun(t70, -t37, t71, 0)
        t99 = strain[1] / 0.2e1
        t100 = strain[0] / 0.2e1
        t101 = t12 / 0.2e1
        t102 = t99 + t100 + t101 - t30
        t114 = t99 + t100 - t101 - t72
        C1111 = (
            (
                ((1 + t2) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t38 * t16 * t30 * t26
                    - 0.2e1 * t37 * t30 * t46 * t43
                    + t53 * t51 * t43
                    + 0.2e1 * t75 * t59 * t72 * t68
                    + 0.2e1 * t52 * t72 * t83 * t80
                    + t89 * t88 * t80
                )
                * mu
            )
            * g
            + ((1 - t2) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t38 * t16 * t102 * t26
                - 0.2e1 * t37 * t102 * t46 * t43
                + (0.1e1 / 0.2e1 + t36 - t53) * t51 * t43
                + 0.2e1 * t75 * t59 * t114 * t68
                + 0.2e1 * t52 * t114 * t83 * t80
                + (0.1e1 / 0.2e1 - t36 - t89) * t88 * t80
            )
            * mu
        )
    elif mode == 2:

        t3 = nu / (1 - nu)
        t7 = absDerFun(1, strain[0] + strain[1] - (strain[0] + strain[1]) * t3)
        t9 = (1 - t3) * t7
        t12 = strain[0] ** 2
        t15 = (strain[2] / 2) ** 2
        t17 = strain[1] ** 2
        t19 = np.sqrt((-2 * strain[1] * strain[0] + t12 + 4 * t15 + t17))
        t20 = strain[1] - strain[0] + t19
        t22 = 0.2e1 / t20 * (strain[2] / 2)
        t23 = abs(t22)
        t24 = t23**2
        t25 = 0.1e1 + t24
        t26 = t25**2
        t28 = t15 * (strain[2] / 2)
        t30 = t20**2 / 0.4e1
        t31 = t30**2
        t33 = 0.1e1 / t31 * t28 / t26
        t34 = strain[1] + strain[0] + t19
        t35 = 0.0e0 < t34 / 0.2e1
        t36 = t35 * (t34 / 0.2e1)
        t38 = absDerFun(1, t22)
        t42 = (strain[0] - strain[1]) / t19 / 0.2e1
        t43 = -0.1e1 / 0.2e1 + t42
        t44 = t43 * t38
        t49 = t15 / t25
        t52 = 0.2e1 / t30 / t20
        t57 = 0.1e1 / t30
        t58 = 0.1e1 / 0.2e1 + t42
        t59 = t35 * t58
        t62 = strain[1] - strain[0] - t19
        t64 = 0.2e1 / t62 * (strain[2] / 2)
        t65 = abs(t64)
        t66 = t65**2
        t67 = 0.1e1 + t66
        t68 = t67**2
        t71 = t62**2 / 0.4e1
        t72 = t71**2
        t74 = 0.1e1 / t72 * t28 / t68
        t75 = strain[1] + strain[0] - t19
        t76 = 0.0e0 < t75 / 0.2e1
        t77 = t76 * (t75 / 0.2e1)
        t79 = absDerFun(1, t64)
        t80 = -t58 * t79
        t85 = t15 / t67
        t88 = 0.2e1 / t71 / t62
        t93 = 0.1e1 / t71
        t94 = t76 * -t43
        t104 = strain[1] / 0.2e1
        t105 = strain[0] / 0.2e1
        t106 = t19 / 0.2e1
        t107 = t104 + t105 + t106 - t36
        t119 = t104 + t105 - t106 - t77
        C1111 = (
            (
                ((1 - t3 + t9) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t44 * t23 * t36 * t33
                    - 0.2e1 * t43 * t36 * t52 * t49
                    + t59 * t57 * t49
                    + 0.2e1 * t80 * t65 * t77 * t74
                    + 0.2e1 * t58 * t77 * t88 * t85
                    + t94 * t93 * t85
                )
                * mu
            )
            * g
            + ((1 - t3 - t9) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t44 * t23 * t107 * t33
                - 0.2e1 * t43 * t107 * t52 * t49
                + (0.1e1 / 0.2e1 + t42 - t59) * t57 * t49
                + 0.2e1 * t80 * t65 * t119 * t74
                + 0.2e1 * t58 * t119 * t88 * t85
                + (0.1e1 / 0.2e1 - t42 - t94) * t93 * t85
            )
            * mu
        )

    return C1111


def C1122_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = absDerFun(1, strain[1] + strain[0])
        t5 = strain[0] ** 2
        t8 = (strain[2] / 2) ** 2
        t10 = strain[1] ** 2
        t12 = np.sqrt((-2 * strain[1] * strain[0] + t10 + t5 + 4 * t8))
        t13 = strain[1] - strain[0] + t12
        t15 = 0.2e1 / t13 * (strain[2] / 2)
        t16 = abs(t15)
        t17 = t16**2
        t18 = 0.1e1 + t17
        t19 = t18**2
        t21 = t8 * (strain[2] / 2)
        t23 = t13**2 / 0.4e1
        t24 = t23**2
        t26 = 0.1e1 / t24 * t21 / t19
        t27 = strain[1] + strain[0] + t12
        t28 = 0.0e0 < t27 / 0.2e1
        t29 = t27 / 0.2e1 <= 0.0e0
        t30 = piecewiseFun(t28, t27 / 0.2e1, t29, 0)
        t32 = absDerFun(1, t15)
        t36 = (-strain[0] + strain[1]) / t12 / 0.2e1
        t37 = 0.1e1 / 0.2e1 + t36
        t38 = t37 * t32
        t43 = t8 / t18
        t46 = 0.2e1 / t23 / t13
        t51 = 0.1e1 / t23
        t52 = piecewiseFun(t28, t37, t29, 0)
        t55 = strain[1] - strain[0] - t12
        t57 = 0.2e1 / t55 * (strain[2] / 2)
        t58 = abs(t57)
        t59 = t58**2
        t60 = 0.1e1 + t59
        t61 = t60**2
        t64 = t55**2 / 0.4e1
        t65 = t64**2
        t67 = 0.1e1 / t65 * t21 / t61
        t68 = strain[1] + strain[0] - t12
        t69 = 0.0e0 < t68 / 0.2e1
        t70 = t68 / 0.2e1 <= 0.0e0
        t71 = piecewiseFun(t69, t68 / 0.2e1, t70, 0)
        t73 = absDerFun(1, t57)
        t74 = 0.1e1 / 0.2e1 - t36
        t75 = t74 * t73
        t80 = t8 / t60
        t83 = 0.2e1 / t64 / t55
        t88 = 0.1e1 / t64
        t89 = piecewiseFun(t69, t74, t70, 0)
        t99 = strain[1] / 0.2e1
        t100 = strain[0] / 0.2e1
        t101 = t12 / 0.2e1
        t102 = t99 + t100 + t101 - t30
        t114 = t99 + t100 - t101 - t71
        C1122 = (
            (
                ((1 + t2) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t38 * t16 * t30 * t26
                    - 0.2e1 * t37 * t30 * t46 * t43
                    + t52 * t51 * t43
                    + 0.2e1 * t75 * t58 * t71 * t67
                    - 0.2e1 * t74 * t71 * t83 * t80
                    + t89 * t88 * t80
                )
                * mu
            )
            * g
            + ((1 - t2) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t38 * t16 * t102 * t26
                - 0.2e1 * t37 * t102 * t46 * t43
                + (0.1e1 / 0.2e1 + t36 - t52) * t51 * t43
                + 0.2e1 * t75 * t58 * t114 * t67
                - 0.2e1 * t74 * t114 * t83 * t80
                + (0.1e1 / 0.2e1 - t36 - t89) * t88 * t80
            )
            * mu
        )
    elif mode == 2:
        t3 = nu / (1 - nu)
        t7 = absDerFun(1, strain[0] + strain[1] - (strain[0] + strain[1]) * t3)
        t9 = (1 - t3) * t7
        t12 = strain[0] ** 2
        t15 = (strain[2] / 2) ** 2
        t17 = strain[1] ** 2
        t19 = np.sqrt((-2 * strain[1] * strain[0] + t12 + 4 * t15 + t17))
        t20 = strain[1] - strain[0] + t19
        t22 = 0.2e1 / t20 * (strain[2] / 2)
        t23 = abs(t22)
        t24 = t23**2
        t25 = 0.1e1 + t24
        t26 = t25**2
        t28 = t15 * (strain[2] / 2)
        t30 = t20**2 / 0.4e1
        t31 = t30**2
        t33 = 0.1e1 / t31 * t28 / t26
        t34 = strain[1] + strain[0] + t19
        t35 = 0.0e0 < t34 / 0.2e1
        t36 = t35 * (t34 / 0.2e1)
        t38 = absDerFun(1, t22)
        t42 = (-strain[0] + strain[1]) / t19 / 0.2e1
        t43 = 0.1e1 / 0.2e1 + t42
        t44 = t43 * t38
        t49 = t15 / t25
        t52 = 0.2e1 / t30 / t20
        t57 = 0.1e1 / t30
        t58 = t35 * t43
        t61 = strain[1] - strain[0] - t19
        t63 = 0.2e1 / t61 * (strain[2] / 2)
        t64 = abs(t63)
        t65 = t64**2
        t66 = 0.1e1 + t65
        t67 = t66**2
        t70 = t61**2 / 0.4e1
        t71 = t70**2
        t73 = 0.1e1 / t71 * t28 / t67
        t74 = strain[1] + strain[0] - t19
        t75 = 0.0e0 < t74 / 0.2e1
        t76 = t75 * (t74 / 0.2e1)
        t78 = absDerFun(1, t63)
        t79 = 0.1e1 / 0.2e1 - t42
        t80 = t79 * t78
        t85 = t15 / t66
        t88 = 0.2e1 / t70 / t61
        t93 = 0.1e1 / t70
        t94 = t75 * t79
        t104 = strain[1] / 0.2e1
        t105 = strain[0] / 0.2e1
        t106 = t19 / 0.2e1
        t107 = t104 + t105 + t106 - t36
        t119 = t104 + t105 - t106 - t76
        C1122 = (
            (
                ((1 - t3 + t9) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t44 * t23 * t36 * t33
                    - 0.2e1 * t43 * t36 * t52 * t49
                    + t58 * t57 * t49
                    + 0.2e1 * t80 * t64 * t76 * t73
                    - 0.2e1 * t79 * t76 * t88 * t85
                    + t94 * t93 * t85
                )
                * mu
            )
            * g
            + ((1 - t3 - t9) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t44 * t23 * t107 * t33
                - 0.2e1 * t43 * t107 * t52 * t49
                + (0.1e1 / 0.2e1 + t42 - t58) * t57 * t49
                + 0.2e1 * t80 * t64 * t119 * t73
                - 0.2e1 * t79 * t119 * t88 * t85
                + (0.1e1 / 0.2e1 - t42 - t94) * t93 * t85
            )
            * mu
        )

    return C1122


def C1112_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t20 = 0.1e1 / t19
        t21 = t20 * t5 / t16
        t22 = strain[1] + strain[0] + t9
        t23 = 0.0e0 < t22 / 0.2e1
        t24 = t22 / 0.2e1 <= 0.0e0
        t25 = piecewiseFun(t23, t22 / 0.2e1, t24, 0)
        t27 = absDerFun(1, t12)
        t29 = 0.1e1 / t9
        t33 = (-0.2e1 * t29 * t20 * t5 + t11) * t27
        t37 = 0.1e1 / t15
        t38 = (strain[2] / 2) * t37
        t42 = t5 * (strain[2] / 2)
        t43 = t42 * t37
        t46 = 0.2e1 / t19 / t10
        t51 = t5 * t37
        t53 = 0.2e1 * (strain[2] / 2) * t29
        t54 = piecewiseFun(t23, t53, t24, 0)
        t57 = strain[1] - strain[0] - t9
        t58 = 0.2e1 / t57
        t59 = t58 * (strain[2] / 2)
        t60 = abs(t59)
        t61 = t60**2
        t62 = 0.1e1 + t61
        t63 = t62**2
        t66 = t57**2 / 0.4e1
        t67 = 0.1e1 / t66
        t68 = t67 * t5 / t63
        t69 = strain[1] + strain[0] - t9
        t70 = 0.0e0 < t69 / 0.2e1
        t71 = t69 / 0.2e1 <= 0.0e0
        t72 = piecewiseFun(t70, t69 / 0.2e1, t71, 0)
        t74 = absDerFun(1, t59)
        t79 = (0.2e1 * t29 * t67 * t5 + t58) * t74
        t83 = 0.1e1 / t62
        t84 = (strain[2] / 2) * t83
        t88 = t42 * t83
        t91 = 0.2e1 / t66 / t57
        t96 = t5 * t83
        t97 = piecewiseFun(t70, -t53, t71, 0)
        t102 = strain[1] / 0.2e1
        t103 = strain[0] / 0.2e1
        t104 = t9 / 0.2e1
        t105 = t102 + t103 + t104 - t25
        t120 = t102 + t103 - t104 - t72
        C1112 = (
            0.2e1
            * (
                -0.2e1 * t33 * t13 * t25 * t21
                + 0.2e1 * t25 * t20 * t38
                - 0.4e1 * t29 * t25 * t46 * t43
                + t54 * t20 * t51
                - 0.2e1 * t79 * t60 * t72 * t68
                + 0.2e1 * t72 * t67 * t84
                + 0.4e1 * t29 * t72 * t91 * t88
                + t97 * t67 * t96
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t33 * t13 * t105 * t21
                + 0.2e1 * t105 * t20 * t38
                - 0.4e1 * t29 * t105 * t46 * t43
                + (t53 - t54) * t20 * t51
                - 0.2e1 * t79 * t60 * t120 * t68
                + 0.2e1 * t120 * t67 * t84
                + 0.4e1 * t29 * t120 * t91 * t88
                + (-t53 - t97) * t67 * t96
            )
            * mu
        )
        C1112 = 0.5 * C1112
    elif mode == 2:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t20 = 0.1e1 / t19
        t21 = t20 * t5 / t16
        t22 = strain[1] + strain[0] + t9
        t23 = 0.0e0 < t22 / 0.2e1
        t24 = t23 * (t22 / 0.2e1)
        t26 = absDerFun(1, t12)
        t28 = 0.1e1 / t9
        t32 = (-0.2e1 * t28 * t20 * t5 + t11) * t26
        t36 = 0.1e1 / t15
        t37 = (strain[2] / 2) * t36
        t41 = t5 * (strain[2] / 2)
        t42 = t41 * t36
        t45 = 0.2e1 / t19 / t10
        t50 = t5 * t36
        t52 = 0.2e1 * (strain[2] / 2) * t28
        t53 = t23 * t52
        t56 = strain[1] - strain[0] - t9
        t57 = 0.2e1 / t56
        t58 = t57 * (strain[2] / 2)
        t59 = abs(t58)
        t60 = t59**2
        t61 = 0.1e1 + t60
        t62 = t61**2
        t65 = t56**2 / 0.4e1
        t66 = 0.1e1 / t65
        t67 = t66 * t5 / t62
        t68 = strain[1] + strain[0] - t9
        t69 = 0.0e0 < t68 / 0.2e1
        t70 = t69 * (t68 / 0.2e1)
        t72 = absDerFun(1, t58)
        t77 = (0.2e1 * t28 * t66 * t5 + t57) * t72
        t81 = 0.1e1 / t61
        t82 = (strain[2] / 2) * t81
        t86 = t41 * t81
        t89 = 0.2e1 / t65 / t56
        t94 = t5 * t81
        t95 = t69 * -t52
        t100 = strain[1] / 0.2e1
        t101 = strain[0] / 0.2e1
        t102 = t9 / 0.2e1
        t103 = t100 + t101 + t102 - t24
        t118 = t100 + t101 - t102 - t70
        C1112 = (
            0.2e1
            * (
                -0.2e1 * t32 * t13 * t24 * t21
                + 0.2e1 * t24 * t20 * t37
                - 0.4e1 * t28 * t24 * t45 * t42
                + t53 * t20 * t50
                - 0.2e1 * t77 * t59 * t70 * t67
                + 0.2e1 * t70 * t66 * t82
                + 0.4e1 * t28 * t70 * t89 * t86
                + t95 * t66 * t94
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t32 * t13 * t103 * t21
                + 0.2e1 * t103 * t20 * t37
                - 0.4e1 * t28 * t103 * t45 * t42
                + (t52 - t53) * t20 * t50
                - 0.2e1 * t77 * t59 * t118 * t67
                + 0.2e1 * t118 * t66 * t82
                + 0.4e1 * t28 * t118 * t89 * t86
                + (-t52 - t95) * t66 * t94
            )
            * mu
        )
        C1112 = 0.50 * C1112

    return C1112


def C2211_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = absDerFun(1, strain[1] + strain[0])
        t5 = strain[0] ** 2
        t8 = (strain[2] / 2) ** 2
        t10 = strain[1] ** 2
        t12 = np.sqrt((-2 * strain[1] * strain[0] + t10 + t5 + 4 * t8))
        t13 = strain[1] - strain[0] + t12
        t15 = 0.2e1 / t13 * (strain[2] / 2)
        t16 = abs(t15)
        t17 = t16**2
        t18 = 0.1e1 + t17
        t19 = t18**2
        t20 = 0.1e1 / t19
        t21 = strain[1] + strain[0] + t12
        t22 = 0.0e0 < t21 / 0.2e1
        t23 = t21 / 0.2e1 <= 0.0e0
        t24 = piecewiseFun(t22, t21 / 0.2e1, t23, 0)
        t27 = absDerFun(1, t15)
        t34 = (strain[0] - strain[1]) / t12 / 0.2e1
        t35 = -0.1e1 / 0.2e1 + t34
        t37 = 0.4e1 * t35 / t13**2 * (strain[2] / 2) * t27
        t40 = 0.1e1 / t18
        t41 = 0.1e1 / 0.2e1 + t34
        t42 = piecewiseFun(t22, t41, t23, 0)
        t44 = strain[1] - strain[0] - t12
        t46 = 0.2e1 / t44 * (strain[2] / 2)
        t47 = abs(t46)
        t48 = t47**2
        t49 = 0.1e1 + t48
        t50 = t49**2
        t51 = 0.1e1 / t50
        t52 = strain[1] + strain[0] - t12
        t53 = 0.0e0 < t52 / 0.2e1
        t54 = t52 / 0.2e1 <= 0.0e0
        t55 = piecewiseFun(t53, t52 / 0.2e1, t54, 0)
        t58 = absDerFun(1, t46)
        t63 = -0.4e1 * t41 / t44**2 * (strain[2] / 2) * t58
        t66 = 0.1e1 / t49
        t67 = piecewiseFun(t53, -t35, t54, 0)
        t76 = strain[1] / 0.2e1
        t77 = strain[0] / 0.2e1
        t78 = t12 / 0.2e1
        C2211 = (
            (
                ((1 + t2) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t37 * t16 * t24 * t20
                    + t42 * t40
                    + 0.2e1 * t63 * t47 * t55 * t51
                    + t67 * t66
                )
                * mu
            )
            * g
            + ((1 - t2) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t37 * t16 * (t76 + t77 + t78 - t24) * t20
                + (0.1e1 / 0.2e1 + t34 - t42) * t40
                + 0.2e1 * t63 * t47 * (t76 + t77 - t78 - t55) * t51
                + (0.1e1 / 0.2e1 - t34 - t67) * t66
            )
            * mu
        )
    elif mode == 2:
        t3 = nu / (1 - nu)
        t7 = absDerFun(1, strain[0] + strain[1] - (strain[0] + strain[1]) * t3)
        t9 = (1 - t3) * t7
        t12 = strain[0] ** 2
        t15 = (strain[2] / 2) ** 2
        t17 = strain[1] ** 2
        t19 = np.sqrt((-2 * strain[1] * strain[0] + t12 + 4 * t15 + t17))
        t20 = strain[1] - strain[0] + t19
        t22 = 0.2e1 / t20 * (strain[2] / 2)
        t23 = abs(t22)
        t24 = t23**2
        t25 = 0.1e1 + t24
        t26 = t25**2
        t27 = 0.1e1 / t26
        t28 = strain[1] + strain[0] + t19
        t29 = 0.0e0 < t28 / 0.2e1
        t30 = t29 * (t28 / 0.2e1)
        t33 = absDerFun(1, t22)
        t40 = (strain[0] - strain[1]) / t19 / 0.2e1
        t41 = -0.1e1 / 0.2e1 + t40
        t43 = 0.4e1 * t41 / t20**2 * (strain[2] / 2) * t33
        t46 = 0.1e1 / t25
        t47 = 0.1e1 / 0.2e1 + t40
        t48 = t29 * t47
        t50 = strain[1] - strain[0] - t19
        t52 = 0.2e1 / t50 * (strain[2] / 2)
        t53 = abs(t52)
        t54 = t53**2
        t55 = 0.1e1 + t54
        t56 = t55**2
        t57 = 0.1e1 / t56
        t58 = strain[1] + strain[0] - t19
        t59 = 0.0e0 < t58 / 0.2e1
        t60 = t59 * (t58 / 0.2e1)
        t63 = absDerFun(1, t52)
        t68 = -0.4e1 * t47 / t50**2 * (strain[2] / 2) * t63
        t71 = 0.1e1 / t55
        t72 = t59 * -t41
        t81 = strain[1] / 0.2e1
        t82 = strain[0] / 0.2e1
        t83 = t19 / 0.2e1
        C2211 = (
            (
                ((1 - t3 + t9) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t43 * t23 * t30 * t27
                    + t48 * t46
                    + 0.2e1 * t68 * t53 * t60 * t57
                    + t72 * t71
                )
                * mu
            )
            * g
            + ((1 - t3 - t9) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t43 * t23 * (t81 + t82 + t83 - t30) * t27
                + (0.1e1 / 0.2e1 + t40 - t48) * t46
                + 0.2e1 * t68 * t53 * (t81 + t82 - t83 - t60) * t57
                + (0.1e1 / 0.2e1 - t40 - t72) * t71
            )
            * mu
        )

    return C2211


def C2222_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = absDerFun(1, strain[1] + strain[0])
        t5 = strain[0] ** 2
        t8 = (strain[2] / 2) ** 2
        t10 = strain[1] ** 2
        t12 = np.sqrt((-2 * strain[1] * strain[0] + t10 + t5 + 4 * t8))
        t13 = strain[1] - strain[0] + t12
        t15 = 0.2e1 / t13 * (strain[2] / 2)
        t16 = abs(t15)
        t17 = t16**2
        t18 = 0.1e1 + t17
        t19 = t18**2
        t20 = 0.1e1 / t19
        t21 = strain[1] + strain[0] + t12
        t22 = 0.0e0 < t21 / 0.2e1
        t23 = t21 / 0.2e1 <= 0.0e0
        t24 = piecewiseFun(t22, t21 / 0.2e1, t23, 0)
        t27 = absDerFun(1, t15)
        t34 = (-strain[0] + strain[1]) / t12 / 0.2e1
        t35 = 0.1e1 / 0.2e1 + t34
        t37 = 0.4e1 * t35 / t13**2 * (strain[2] / 2) * t27
        t40 = 0.1e1 / t18
        t41 = piecewiseFun(t22, t35, t23, 0)
        t43 = strain[1] - strain[0] - t12
        t45 = 0.2e1 / t43 * (strain[2] / 2)
        t46 = abs(t45)
        t47 = t46**2
        t48 = 0.1e1 + t47
        t49 = t48**2
        t50 = 0.1e1 / t49
        t51 = strain[1] + strain[0] - t12
        t52 = 0.0e0 < t51 / 0.2e1
        t53 = t51 / 0.2e1 <= 0.0e0
        t54 = piecewiseFun(t52, t51 / 0.2e1, t53, 0)
        t57 = absDerFun(1, t45)
        t61 = 0.1e1 / 0.2e1 - t34
        t63 = 0.4e1 * t61 / t43**2 * (strain[2] / 2) * t57
        t66 = 0.1e1 / t48
        t67 = piecewiseFun(t52, t61, t53, 0)
        t76 = strain[1] / 0.2e1
        t77 = strain[0] / 0.2e1
        t78 = t12 / 0.2e1
        C2222 = (
            (
                ((1 + t2) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t37 * t16 * t24 * t20
                    + t41 * t40
                    + 0.2e1 * t63 * t46 * t54 * t50
                    + t67 * t66
                )
                * mu
            )
            * g
            + ((1 - t2) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t37 * t16 * (t76 + t77 + t78 - t24) * t20
                + (0.1e1 / 0.2e1 + t34 - t41) * t40
                + 0.2e1 * t63 * t46 * (t76 + t77 - t78 - t54) * t50
                + (0.1e1 / 0.2e1 - t34 - t67) * t66
            )
            * mu
        )
    elif mode == 2:
        t3 = nu / (1 - nu)
        t7 = absDerFun(1, strain[0] + strain[1] - (strain[0] + strain[1]) * t3)
        t9 = (1 - t3) * t7
        t12 = strain[0] ** 2
        t15 = (strain[2] / 2) ** 2
        t17 = strain[1] ** 2
        t19 = np.sqrt((-2 * strain[1] * strain[0] + t12 + 4 * t15 + t17))
        t20 = strain[1] - strain[0] + t19
        t22 = 0.2e1 / t20 * (strain[2] / 2)
        t23 = abs(t22)
        t24 = t23**2
        t25 = 0.1e1 + t24
        t26 = t25**2
        t27 = 0.1e1 / t26
        t28 = strain[1] + strain[0] + t19
        t29 = 0.0e0 < t28 / 0.2e1
        t30 = t29 * (t28 / 0.2e1)
        t33 = absDerFun(1, t22)
        t40 = (-strain[0] + strain[1]) / t19 / 0.2e1
        t41 = 0.1e1 / 0.2e1 + t40
        t43 = 0.4e1 * t41 / t20**2 * (strain[2] / 2) * t33
        t46 = 0.1e1 / t25
        t47 = t29 * t41
        t49 = strain[1] - strain[0] - t19
        t51 = 0.2e1 / t49 * (strain[2] / 2)
        t52 = abs(t51)
        t53 = t52**2
        t54 = 0.1e1 + t53
        t55 = t54**2
        t56 = 0.1e1 / t55
        t57 = strain[1] + strain[0] - t19
        t58 = 0.0e0 < t57 / 0.2e1
        t59 = t58 * (t57 / 0.2e1)
        t62 = absDerFun(1, t51)
        t66 = 0.1e1 / 0.2e1 - t40
        t68 = 0.4e1 * t66 / t49**2 * (strain[2] / 2) * t62
        t71 = 0.1e1 / t54
        t72 = t58 * t66
        t81 = strain[1] / 0.2e1
        t82 = strain[0] / 0.2e1
        t83 = t19 / 0.2e1
        C2222 = (
            (
                ((1 - t3 + t9) * lmbda) / 0.2e1
                + 0.2e1
                * (
                    0.2e1 * t43 * t23 * t30 * t27
                    + t47 * t46
                    + 0.2e1 * t68 * t52 * t59 * t56
                    + t72 * t71
                )
                * mu
            )
            * g
            + ((1 - t3 - t9) * lmbda) / 0.2e1
            + 0.2e1
            * (
                0.2e1 * t43 * t23 * (t81 + t82 + t83 - t30) * t27
                + (0.1e1 / 0.2e1 + t40 - t47) * t46
                + 0.2e1 * t68 * t52 * (t81 + t82 - t83 - t59) * t56
                + (0.1e1 / 0.2e1 - t40 - t72) * t71
            )
            * mu
        )

    return C2222


def C2212_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t17 = 0.1e1 / t16
        t18 = strain[1] + strain[0] + t9
        t19 = 0.0e0 < t18 / 0.2e1
        t20 = t18 / 0.2e1 <= 0.0e0
        t21 = piecewiseFun(t19, t18 / 0.2e1, t20, 0)
        t23 = absDerFun(1, t12)
        t28 = 0.1e1 / t9
        t32 = (t11 - 0.8e1 * t28 / t10**2 * t5) * t23 * t13
        t35 = 0.1e1 / t15
        t37 = 0.2e1 * (strain[2] / 2) * t28
        t38 = piecewiseFun(t19, t37, t20, 0)
        t40 = strain[1] - strain[0] - t9
        t41 = 0.2e1 / t40
        t42 = t41 * (strain[2] / 2)
        t43 = abs(t42)
        t44 = t43**2
        t45 = 0.1e1 + t44
        t46 = t45**2
        t47 = 0.1e1 / t46
        t48 = strain[1] + strain[0] - t9
        t49 = 0.0e0 < t48 / 0.2e1
        t50 = t48 / 0.2e1 <= 0.0e0
        t51 = piecewiseFun(t49, t48 / 0.2e1, t50, 0)
        t53 = absDerFun(1, t42)
        t61 = (t41 + 0.8e1 * t28 / t40**2 * t5) * t53 * t43
        t64 = 0.1e1 / t45
        t65 = piecewiseFun(t49, -t37, t50, 0)
        t69 = strain[1] / 0.2e1
        t70 = strain[0] / 0.2e1
        t71 = t9 / 0.2e1
        C2212 = (
            0.2e1
            * (
                -0.2e1 * t32 * t21 * t17
                - 0.2e1 * t61 * t51 * t47
                + t38 * t35
                + t65 * t64
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t32 * (t69 + t70 + t71 - t21) * t17
                + (t37 - t38) * t35
                - 0.2e1 * t61 * (t69 + t70 - t71 - t51) * t47
                + (-t37 - t65) * t64
            )
            * mu
        )
        C2212 = 0.5 * C2212
    elif mode == 2:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t17 = 0.1e1 / t16
        t18 = strain[1] + strain[0] + t9
        t19 = 0.0e0 < t18 / 0.2e1
        t20 = t19 * (t18 / 0.2e1)
        t22 = absDerFun(1, t12)
        t27 = 0.1e1 / t9
        t31 = (t11 - 0.8e1 * t27 / t10**2 * t5) * t22 * t13
        t34 = 0.1e1 / t15
        t36 = 0.2e1 * (strain[2] / 2) * t27
        t37 = t19 * t36
        t39 = strain[1] - strain[0] - t9
        t40 = 0.2e1 / t39
        t41 = t40 * (strain[2] / 2)
        t42 = abs(t41)
        t43 = t42**2
        t44 = 0.1e1 + t43
        t45 = t44**2
        t46 = 0.1e1 / t45
        t47 = strain[1] + strain[0] - t9
        t48 = 0.0e0 < t47 / 0.2e1
        t49 = t48 * (t47 / 0.2e1)
        t51 = absDerFun(1, t41)
        t59 = (t40 + 0.8e1 * t27 / t39**2 * t5) * t51 * t42
        t62 = 0.1e1 / t44
        t63 = t48 * -t36
        t67 = strain[1] / 0.2e1
        t68 = strain[0] / 0.2e1
        t69 = t9 / 0.2e1
        C2212 = (
            0.2e1
            * (
                -0.2e1 * t31 * t20 * t17
                - 0.2e1 * t59 * t49 * t46
                + t37 * t34
                + t63 * t62
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t31 * (t67 + t68 + t69 - t20) * t17
                + (t36 - t37) * t34
                - 0.2e1 * t59 * (t67 + t68 - t69 - t49) * t46
                + (-t36 - t63) * t62
            )
            * mu
        )
        C2212 = 0.50 * C2212

    return C2212


def C1211_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t23 = 0.2e1 / t19 / t10 * t5 / t16
        t24 = strain[1] + strain[0] + t9
        t25 = 0.0e0 < t24 / 0.2e1
        t26 = t24 / 0.2e1 <= 0.0e0
        t27 = piecewiseFun(t25, t24 / 0.2e1, t26, 0)
        t29 = absDerFun(1, t12)
        t33 = (strain[0] - strain[1]) / t9 / 0.2e1
        t34 = -0.1e1 / 0.2e1 + t33
        t35 = t34 * t29
        t40 = (strain[2] / 2) / t15
        t41 = 0.1e1 / t19
        t45 = 0.1e1 / 0.2e1 + t33
        t46 = piecewiseFun(t25, t45, t26, 0)
        t49 = strain[1] - strain[0] - t9
        t50 = 0.2e1 / t49
        t51 = t50 * (strain[2] / 2)
        t52 = abs(t51)
        t53 = t52**2
        t54 = 0.1e1 + t53
        t55 = t54**2
        t58 = t49**2 / 0.4e1
        t62 = 0.2e1 / t58 / t49 / t55 * t5
        t63 = strain[1] + strain[0] - t9
        t64 = 0.0e0 < t63 / 0.2e1
        t65 = t63 / 0.2e1 <= 0.0e0
        t66 = piecewiseFun(t64, t63 / 0.2e1, t65, 0)
        t68 = absDerFun(1, t51)
        t69 = -t45 * t68
        t74 = (strain[2] / 2) / t54
        t75 = 0.1e1 / t58
        t79 = piecewiseFun(t64, -t34, t65, 0)
        t84 = strain[1] / 0.2e1
        t85 = strain[0] / 0.2e1
        t86 = t9 / 0.2e1
        t87 = t84 + t85 + t86 - t27
        t98 = t84 + t85 - t86 - t66
        C1211 = (
            0.2e1
            * (
                0.2e1 * t35 * t13 * t27 * t23
                - t34 * t27 * t41 * t40
                + t46 * t11 * t40
                + 0.2e1 * t69 * t52 * t66 * t62
                + t45 * t66 * t75 * t74
                + t79 * t50 * t74
            )
            * mu
            * g
            + 0.2e1
            * (
                0.2e1 * t35 * t13 * t87 * t23
                - t34 * t87 * t41 * t40
                + (0.1e1 / 0.2e1 + t33 - t46) * t11 * t40
                + 0.2e1 * t69 * t52 * t98 * t62
                + t45 * t98 * t75 * t74
                + (0.1e1 / 0.2e1 - t33 - t79) * t50 * t74
            )
            * mu
        )
    elif mode == 2:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t23 = 0.2e1 / t19 / t10 * t5 / t16
        t24 = strain[1] + strain[0] + t9
        t25 = 0.0e0 < t24 / 0.2e1
        t26 = t25 * (t24 / 0.2e1)
        t28 = absDerFun(1, t12)
        t32 = (strain[0] - strain[1]) / t9 / 0.2e1
        t33 = -0.1e1 / 0.2e1 + t32
        t34 = t33 * t28
        t39 = (strain[2] / 2) / t15
        t40 = 0.1e1 / t19
        t44 = 0.1e1 / 0.2e1 + t32
        t45 = t25 * t44
        t48 = strain[1] - strain[0] - t9
        t49 = 0.2e1 / t48
        t50 = t49 * (strain[2] / 2)
        t51 = abs(t50)
        t52 = t51**2
        t53 = 0.1e1 + t52
        t54 = t53**2
        t57 = t48**2 / 0.4e1
        t61 = 0.2e1 / t57 / t48 * t5 / t54
        t62 = strain[1] + strain[0] - t9
        t63 = 0.0e0 < t62 / 0.2e1
        t64 = t63 * (t62 / 0.2e1)
        t66 = absDerFun(1, t50)
        t67 = -t44 * t66
        t72 = (strain[2] / 2) / t53
        t73 = 0.1e1 / t57
        t77 = t63 * -t33
        t82 = strain[1] / 0.2e1
        t83 = strain[0] / 0.2e1
        t84 = t9 / 0.2e1
        t85 = t82 + t83 + t84 - t26
        t96 = t82 + t83 - t84 - t64
        C1211 = (
            0.2e1
            * (
                0.2e1 * t34 * t13 * t26 * t23
                - t33 * t26 * t40 * t39
                + t45 * t11 * t39
                + 0.2e1 * t67 * t51 * t64 * t61
                + t44 * t64 * t73 * t72
                + t77 * t49 * t72
            )
            * mu
            * g
            + 0.2e1
            * (
                0.2e1 * t34 * t13 * t85 * t23
                - t33 * t85 * t40 * t39
                + (0.1e1 / 0.2e1 + t32 - t45) * t11 * t39
                + 0.2e1 * t67 * t51 * t96 * t61
                + t44 * t96 * t73 * t72
                + (0.1e1 / 0.2e1 - t32 - t77) * t49 * t72
            )
            * mu
        )

    return C1211


def C1222_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t23 = 0.2e1 / t19 / t10 * t5 / t16
        t24 = strain[1] + strain[0] + t9
        t25 = 0.0e0 < t24 / 0.2e1
        t26 = t24 / 0.2e1 <= 0.0e0
        t27 = piecewiseFun(t25, t24 / 0.2e1, t26, 0)
        t29 = absDerFun(1, t12)
        t33 = (-strain[0] + strain[1]) / t9 / 0.2e1
        t34 = 0.1e1 / 0.2e1 + t33
        t35 = t34 * t29
        t40 = (strain[2] / 2) / t15
        t41 = 0.1e1 / t19
        t45 = piecewiseFun(t25, t34, t26, 0)
        t48 = strain[1] - strain[0] - t9
        t49 = 0.2e1 / t48
        t50 = t49 * (strain[2] / 2)
        t51 = abs(t50)
        t52 = t51**2
        t53 = 0.1e1 + t52
        t54 = t53**2
        t57 = t48**2 / 0.4e1
        t61 = 0.2e1 / t57 / t48 * t5 / t54
        t62 = strain[1] + strain[0] - t9
        t63 = 0.0e0 < t62 / 0.2e1
        t64 = t62 / 0.2e1 <= 0.0e0
        t65 = piecewiseFun(t63, t62 / 0.2e1, t64, 0)
        t67 = absDerFun(1, t50)
        t68 = 0.1e1 / 0.2e1 - t33
        t69 = t68 * t67
        t74 = (strain[2] / 2) / t53
        t75 = 0.1e1 / t57
        t79 = piecewiseFun(t63, t68, t64, 0)
        t84 = strain[1] / 0.2e1
        t85 = strain[0] / 0.2e1
        t86 = t9 / 0.2e1
        t87 = t84 + t85 + t86 - t27
        t98 = t84 + t85 - t86 - t65
        C1222 = (
            0.2e1
            * (
                0.2e1 * t35 * t13 * t27 * t23
                - t34 * t27 * t41 * t40
                + t45 * t11 * t40
                + 0.2e1 * t69 * t51 * t65 * t61
                - t68 * t65 * t75 * t74
                + t79 * t49 * t74
            )
            * mu
            * g
            + 0.2e1
            * (
                0.2e1 * t35 * t13 * t87 * t23
                - t34 * t87 * t41 * t40
                + (0.1e1 / 0.2e1 + t33 - t45) * t11 * t40
                + 0.2e1 * t69 * t51 * t98 * t61
                - t68 * t98 * t75 * t74
                + (0.1e1 / 0.2e1 - t33 - t79) * t49 * t74
            )
            * mu
        )
    elif mode == 2:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t10**2 / 0.4e1
        t23 = 0.2e1 / t19 / t10 * t5 / t16
        t24 = strain[1] + strain[0] + t9
        t25 = 0.0e0 < t24 / 0.2e1
        t26 = t25 * (t24 / 0.2e1)
        t28 = absDerFun(1, t12)
        t32 = (-strain[0] + strain[1]) / t9 / 0.2e1
        t33 = 0.1e1 / 0.2e1 + t32
        t34 = t33 * t28
        t39 = (strain[2] / 2) / t15
        t40 = 0.1e1 / t19
        t44 = t25 * t33
        t47 = strain[1] - strain[0] - t9
        t48 = 0.2e1 / t47
        t49 = t48 * (strain[2] / 2)
        t50 = abs(t49)
        t51 = t50**2
        t52 = 0.1e1 + t51
        t53 = t52**2
        t56 = t47**2 / 0.4e1
        t60 = 0.2e1 / t56 / t47 / t53 * t5
        t61 = strain[1] + strain[0] - t9
        t62 = 0.0e0 < t61 / 0.2e1
        t63 = t62 * (t61 / 0.2e1)
        t65 = absDerFun(1, t49)
        t66 = 0.1e1 / 0.2e1 - t32
        t67 = t66 * t65
        t72 = (strain[2] / 2) / t52
        t73 = 0.1e1 / t56
        t77 = t62 * t66
        t82 = strain[1] / 0.2e1
        t83 = strain[0] / 0.2e1
        t84 = t9 / 0.2e1
        t85 = t82 + t83 + t84 - t26
        t96 = t82 + t83 - t84 - t63
        C1222 = (
            0.2e1
            * (
                0.2e1 * t34 * t13 * t26 * t23
                - t33 * t26 * t40 * t39
                + t44 * t11 * t39
                + 0.2e1 * t67 * t50 * t63 * t60
                - t66 * t63 * t73 * t72
                + t77 * t48 * t72
            )
            * mu
            * g
            + 0.2e1
            * (
                0.2e1 * t34 * t13 * t85 * t23
                - t33 * t85 * t40 * t39
                + (0.1e1 / 0.2e1 + t32 - t44) * t11 * t39
                + 0.2e1 * t67 * t50 * t96 * t60
                - t66 * t96 * t73 * t72
                + (0.1e1 / 0.2e1 - t32 - t77) * t48 * t72
            )
            * mu
        )

    return C1222


def C1212_term(g, lmbda, mu, nu, strain, mode):
    if mode == 1:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t11 * (strain[2] / 2) / t16
        t20 = strain[1] + strain[0] + t9
        t21 = 0.0e0 < t20 / 0.2e1
        t22 = t20 / 0.2e1 <= 0.0e0
        t23 = piecewiseFun(t21, t20 / 0.2e1, t22, 0)
        t25 = absDerFun(1, t12)
        t27 = 0.4e1 / t10**2
        t29 = 0.1e1 / t9
        t33 = (-0.2e1 * t29 * t27 * t5 + t11) * t25
        t37 = 0.1e1 / t15
        t38 = t11 * t37
        t40 = t5 * t37
        t45 = (strain[2] / 2) * t37
        t47 = 0.2e1 * (strain[2] / 2) * t29
        t48 = piecewiseFun(t21, t47, t22, 0)
        t51 = strain[1] - strain[0] - t9
        t52 = 0.2e1 / t51
        t53 = t52 * (strain[2] / 2)
        t54 = abs(t53)
        t55 = t54**2
        t56 = 0.1e1 + t55
        t57 = t56**2
        t60 = t52 * (strain[2] / 2) / t57
        t61 = strain[1] + strain[0] - t9
        t62 = 0.0e0 < t61 / 0.2e1
        t63 = t61 / 0.2e1 <= 0.0e0
        t64 = piecewiseFun(t62, t61 / 0.2e1, t63, 0)
        t66 = absDerFun(1, t53)
        t68 = 0.4e1 / t51**2
        t73 = (0.2e1 * t29 * t68 * t5 + t52) * t66
        t77 = 0.1e1 / t56
        t78 = t52 * t77
        t80 = t5 * t77
        t85 = (strain[2] / 2) * t77
        t86 = piecewiseFun(t62, -t47, t63, 0)
        t91 = strain[1] / 0.2e1
        t92 = strain[0] / 0.2e1
        t93 = t9 / 0.2e1
        t94 = t91 + t92 + t93 - t23
        t107 = t91 + t92 - t93 - t64
        C1212 = (
            0.2e1
            * (
                -0.2e1 * t33 * t23 * t13 * t19
                + t23 * t38
                - 0.2e1 * t29 * t23 * t27 * t40
                + t48 * t11 * t45
                - 0.2e1 * t73 * t54 * t64 * t60
                + t64 * t78
                + 0.2e1 * t29 * t64 * t68 * t80
                + t86 * t52 * t85
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t33 * t13 * t94 * t19
                + t94 * t38
                - 0.2e1 * t29 * t94 * t27 * t40
                + (t47 - t48) * t11 * t45
                - 0.2e1 * t73 * t54 * t107 * t60
                + t107 * t78
                + 0.2e1 * t29 * t107 * t68 * t80
                + (-t47 - t86) * t52 * t85
            )
            * mu
        )
        C1212 = 0.5 * C1212
    elif mode == 2:
        t2 = strain[0] ** 2
        t5 = (strain[2] / 2) ** 2
        t7 = strain[1] ** 2
        t9 = np.sqrt((-2 * strain[1] * strain[0] + t2 + 4 * t5 + t7))
        t10 = strain[1] - strain[0] + t9
        t11 = 0.2e1 / t10
        t12 = t11 * (strain[2] / 2)
        t13 = abs(t12)
        t14 = t13**2
        t15 = 0.1e1 + t14
        t16 = t15**2
        t19 = t11 * (strain[2] / 2) / t16
        t20 = strain[1] + strain[0] + t9
        t21 = 0.0e0 < t20 / 0.2e1
        t22 = t21 * (t20 / 0.2e1)
        t24 = absDerFun(1, t12)
        t26 = 0.4e1 / t10**2
        t28 = 0.1e1 / t9
        t32 = (-0.2e1 * t28 * t26 * t5 + t11) * t24
        t36 = 0.1e1 / t15
        t37 = t11 * t36
        t39 = t5 * t36
        t44 = (strain[2] / 2) * t36
        t46 = 0.2e1 * (strain[2] / 2) * t28
        t47 = t21 * t46
        t50 = strain[1] - strain[0] - t9
        t51 = 0.2e1 / t50
        t52 = t51 * (strain[2] / 2)
        t53 = abs(t52)
        t54 = t53**2
        t55 = 0.1e1 + t54
        t56 = t55**2
        t59 = t51 * (strain[2] / 2) / t56
        t60 = strain[1] + strain[0] - t9
        t61 = 0.0e0 < t60 / 0.2e1
        t62 = t61 * (t60 / 0.2e1)
        t64 = absDerFun(1, t52)
        t66 = 0.4e1 / t50**2
        t71 = (0.2e1 * t28 * t66 * t5 + t51) * t64
        t75 = 0.1e1 / t55
        t76 = t51 * t75
        t78 = t5 * t75
        t83 = (strain[2] / 2) * t75
        t84 = t61 * -t46
        t89 = strain[1] / 0.2e1
        t90 = strain[0] / 0.2e1
        t91 = t9 / 0.2e1
        t92 = t89 + t90 + t91 - t22
        t105 = t89 + t90 - t91 - t62
        C1212 = (
            0.2e1
            * (
                -0.2e1 * t32 * t22 * t13 * t19
                + t22 * t37
                - 0.2e1 * t28 * t22 * t26 * t39
                + t47 * t11 * t44
                - 0.2e1 * t71 * t53 * t62 * t59
                + t62 * t76
                + 0.2e1 * t28 * t62 * t66 * t78
                + t84 * t51 * t83
            )
            * mu
            * g
            + 0.2e1
            * (
                -0.2e1 * t32 * t13 * t92 * t19
                + t92 * t37
                - 0.2e1 * t28 * t92 * t26 * t39
                + (t46 - t47) * t11 * t44
                - 0.2e1 * t71 * t53 * t105 * t59
                + t105 * t76
                + 0.2e1 * t28 * t105 * t66 * t78
                + (-t46 - t84) * t51 * t83
            )
            * mu
        )
        C1212 = 0.50 * C1212

    return C1212


def piecewiseFun(cond1, res1, cond2, res2):
    if cond1 == 1:
        return res1
    elif cond2 == 1:
        return res2
    else:
        return -1


def piecewiseFun2(cond1, res1):
    if cond1 == 1:
        return res1
    return 0


def absDerFun(n, x):
    if x != 0:
        return x / abs(x)
    else:
        return 0


def calculate_plane_strain(E, nu):
    return (E / ((1 + nu) * (1 - 2 * nu))) * np.array(
        [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]],
        order="F",
        dtype=np.float64,
    )


def calculate_plane_stress(E, nu):
    return (E / (1 - nu**2)) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], order="F", dtype=np.float64
    )
