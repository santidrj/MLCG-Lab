from math import exp

from numpy import ndarray

from PyRT_Common import *


class CovarianceFunction(ABC):

    @abstractmethod
    def eval(self, omega_i, omega_j):
        pass


# Sobolev Covariance function
# [Optimal Sample Weights paper (Marques et al. 2019, Computer Graphics Forum)]
class SobolevCov(CovarianceFunction):
    def __init__(self, s=1.4):
        self.s = s  # Smoothness parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        s = self.s
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return (2 ** (2 * s - 1)) / s - r ** (2 * s - 2)


# Squared Exponential (SE) Covariance function
# [Spherical Gaussian Framework paper (Marques et al. 2013, IEEE TVCG)]
class SECov(CovarianceFunction):

    def __init__(self, l, noise):
        super().__init__(noise)
        self.l = l  # Length-scale parameter (controls the smoothness of the GP model)

    def eval(self, omega_i, omega_j):
        r = Length(omega_i - omega_j)  # Euclidean distance between the samples
        return exp(-(r ** 2) / (2 * self.l ** 2))


def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            val *= function_list[j].eval(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))  # for convenience, we'll only use the red channel
    return sample_values


def compute_estimate_cmc(sample_prob_, sample_values_):
    values = [value / prob for prob, value in zip(sample_prob_, sample_values_)]
    if isinstance(values[0], Vector3D) or isinstance(values[0], RGBColor):
        result = BLACK
    else:
        result = 0
    for value in values:
        result += value
    return result / len(sample_prob_)


# Gaussian Process class for the unit hemisphere
class GP:

    # Initializer
    def __init__(self, cov_func, p_func, imp_samp=False, noise_=0.01):

        # Attribute containing the covariance function
        self.cov_func = cov_func

        # Analytically-known part of the integrand, i.e., the function p(x)
        self.p_func = p_func

        # Noise term to be added in the diagonal of the covariance matrix. This typically small value is used to avoid
        #  numerical instabilities when inverting the covariance matrix Q (preempts the matrix Q from being singular or
        #  close to singular, and thus not invertible).
        self.noise = noise_

        # Vector with the sample positions
        self.samples_pos = None

        # Vector with the sample values (Y)
        self.samples_val = None

        # Inverted coveriance matrix Q^{-1}
        self.invQ = None

        # Vector of z coefficients (see theory slides and Practice 3 text for more details)
        self.z = None

        # Sample weights. Contains the value by which each sample y_i must be multiplied to compute the BMC estimate.
        #  See text of Practice 3 for more details.
        self.weights = None

        # PDF used for Importance Sampling
        self.imp_samp = imp_samp
        if self.imp_samp:
            self.pdf = CosinePDF(1)
        else:
            self.pdf = UniformPDF()

    # Method responsible for receiving the vector of sample positions and assigning it to the class attribute sample_pos
    #  Besides that, it also performs the following computations:
    #    - Compute and store the inverse of the covariance matrix (requires knowing the sample positions and the
    #       covariance function)
    #    - Compute and store the z vector (requires knowing the sample positions and the covariance function)
    #    - Compute and store the sample weights given by z^t Q^{-1}
    def add_sample_pos(self, samples_pos_):
        self.samples_pos = samples_pos_
        self.invQ = self.compute_inv_Q()
        self.z = self.compute_z()
        self.weights = self.z @ self.invQ

    # Method which receives and stores the sample values Y (i.e., the observations)
    def add_sample_val(self, samples_val_):
        self.samples_val = samples_val_

    # Method which computes and inverts the covariance matrix Q
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_inv_Q(self):
        n = len(self.samples_pos)
        Q: ndarray = np.zeros((n, n))

        # ################## #
        # ADD YOUR CODE HERE #
        # ################## #
        for i, sample_p in enumerate(self.samples_pos):
            for j, sample_p2 in enumerate(self.samples_pos):
                Q[i, j] = self.cov_func.eval(sample_p, sample_p2)

        # Add a diagonal of a small amount of noise to avoid numerical instability problems
        Q = Q + np.eye(n, n) * self.noise ** 2
        return np.linalg.inv(Q)

    # Method in charge of computing the z vector.
    #  - IMPORTANT: requires that the samples positions are already known
    def compute_z(self):
        # The z vector z  = [z_1, z_2, ..., z_n] is a vector of integrals, where the value of each element is computed
        #  based on the position omega_n of the nth sample. In most of the cases, these integrals do not have an
        #  analytic solution. Therefore, we will use classic Monte Carlo to estimate the value of these integrals
        #  (that is, of each element z_i of z).

        # STEP 1: Set-up the pdf used to sample the integrals. We will use the same pdf for all integral estimates
        # (a uniform pdf). The number of samples used in the estimate is hardcoded (50.000). This is a rather
        # conservative figure which could perhaps be reduced without impairing the final result.
        ns_z = 50000  # number of samples used to estimate z_i

        # STEP 2: Generate a samples set for the MC estimate
        sample_set_z, probab = sample_set_hemisphere(ns_z, self.pdf)
        ns = len(self.samples_pos)
        z_vec = np.zeros(ns)

        # STEP 3: Compute each z_i element of z
        # for each sample in our GP model
        for i in range(ns):
            # STEP 3.1: Fetch the direction omega_i
            omega_i = self.samples_pos[i]

            # STEP 3.2: Use classic Monte Carlo Integration to compute z_i
            # ################## #
            # ADD YOUR CODE HERE #
            # ################## #
            if self.imp_samp:
                sample_values = [self.cov_func.eval(omega_i, sample) * self.p_func.eval(sample) for sample in
                                 sample_set_z]
            else:
                sample_values = [self.cov_func.eval(omega_i, sample) for sample in sample_set_z]
            z_vec[i] = compute_estimate_cmc(probab, sample_values)

        return z_vec

    # Method in charge of computing the BMC integral estimate (assuming that the prior mean function has value 0)
    def compute_integral_BMC(self):
        res = BLACK

        # ################## #
        # ADD YOUR CODE HERE #
        # ################## #
        for w, s in zip(self.weights, self.samples_val):
            res += s * w

        return res
