import numpy as np


def generate_stiffness_matrix(n, eta=1 * pow(10, -5), dpdx=-0.001):
    M = np.zeros([n, n])
    r = np.ones(n)
    delta_z = 2 / n

    # No Slip boundary Condition
    # Equation 32.42 32.47 32.37 CHAPTER 32 Finite Element Method
    # g^{(0)}-g^{(1)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    # -g^{(N-1)}-g^{(N)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    # -g^(j−1) + 2g^(j)-g^(j+1) = \frac{\dp}{\dx}*\frac{1}{2 \eta}(\Delta z)^2
    # M doesn't change becuase it's based on ratios of the g(j) no the addition

    for i in range(n):
        M[i, i] = 2
        if i > 0:
            M[i, i - 1] = -1
        if i < n - 1:
            M[i, i + 1] = -1

    r = -r * dpdx / eta * pow(delta_z, 2)

    return M, r


if __name__ == "__main__":
    M, r = generate_stiffness_matrix(20)
    print(M, r)
