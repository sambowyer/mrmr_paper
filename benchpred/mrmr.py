import numpy as np
import joblib as jbl
from scipy.special import psi
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.decomposition import PCA
from tqdm import tqdm

from .base import BenchPred, set_random_seed
from .py_irt.training import IrtModelTrainer, IrtConfig
from .tiny_bench import (
    create_irt_dataset,
    create_continuous_irt_dataset,
    item_curve,
    PIRTPred,
    GPIRTPred,
)


class _BinomialGLMRegressor:
    """Ridge regression with logit link for [0, 1] targets.

    Fits Ridge in logit space and applies sigmoid for prediction,
    implementing a regularised quasi-binomial GLM.
    """

    def __init__(self, alpha=10):
        self._ridge = Ridge(alpha=alpha)

    def fit(self, X, y):
        y_flat = np.clip(y.ravel(), 1e-6, 1 - 1e-6)
        y_logit = np.log(y_flat / (1 - y_flat))
        self._ridge.fit(X, y_logit.reshape(-1, 1))
        return self

    def predict(self, X):
        logit_pred = self._ridge.predict(X).ravel()
        return 1.0 / (1.0 + np.exp(-logit_pred))


class _RawAverageRegressor:
    """Predict by returning the mean of input features (coreset questions).

    No model is fitted; the prediction for each model is simply the average
    of its scores on the selected coreset questions.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        if X.ndim == 1:
            return np.array([X.mean()])
        return X.mean(axis=1)


class _LogitTargetScaler:
    """Scale targets to (0, 1) with a buffer, then apply logit.

    During fit, learns y_min and y_max.  transform maps
    [y_min, y_max] -> [buffer, 1-buffer] then applies logit.
    inverse_transform applies sigmoid then unscales.

    The buffer leaves room for extrapolation beyond the training range:
    predictions in logit space that exceed the training extremes will map
    to values slightly outside [y_min, y_max] after inverse_transform.
    """

    def __init__(self, buffer=0.05):
        self.buffer = buffer
        self.y_min_ = None
        self.y_max_ = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        self.y_min_ = float(y.min())
        self.y_max_ = float(y.max())
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        span = self.y_max_ - self.y_min_
        if span < 1e-15:
            y_unit = np.full_like(y, 0.5)
        else:
            y_unit = self.buffer + (1.0 - 2.0 * self.buffer) * (y - self.y_min_) / span
        y_unit = np.clip(y_unit, 1e-7, 1.0 - 1e-7)
        return np.log(y_unit / (1.0 - y_unit))

    def inverse_transform(self, y_logit):
        y_logit = np.asarray(y_logit, dtype=np.float64).ravel()
        y_unit = 1.0 / (1.0 + np.exp(-y_logit))
        span = self.y_max_ - self.y_min_
        if span < 1e-15:
            return np.full_like(y_logit, (self.y_min_ + self.y_max_) / 2.0)
        return self.y_min_ + (y_unit - self.buffer) / (1.0 - 2.0 * self.buffer) * span


class _LogitWrappedRegressor:
    """Wraps a fitted regressor so predictions are inverse-transformed from logit space."""

    def __init__(self, regressor, scaler):
        self.regressor = regressor
        self.scaler = scaler

    def predict(self, X):
        logit_pred = self.regressor.predict(X)
        return self.scaler.inverse_transform(logit_pred)


class MRMRPred(BenchPred):
    """MRMR (Minimum Redundancy Maximum Relevance) feature selection algorithm.

    Implements both MID (Minimum Redundancy Maximum Relevance) and MIQ (Minimum
    Redundancy Maximum Relevance with Quotient) schemes for feature selection.

    The algorithm selects features by:
    1. First selecting the feature with highest relevance to the target
    2. Then greedily selecting features that maximize the MRMR score:
       - MID: relevance - (redundancy / number_of_selected_features)
       - MIQ: relevance / (redundancy + epsilon)

    After feature selection, trains a Ridge regression model on the selected features.
    """

    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self.target_scheme = "average"  # "average" or "pc1"
        self.miq_scheme = False  # False for MID, True for MIQ
        self.mi_k = 3  # k nearest neighbours for Ross MI estimator
        self.selection_metrics = None
        self._binary = None
        self.alpha_ = None
        self.alpha_range_ = None

    def _build_regressor(self, X: np.ndarray, y: np.ndarray):
        """Build and fit the prediction model.

        Override in subclasses to swap the regressor (e.g. GLM).

        Args:
            X: Feature matrix of shape (M, K)
            y: Target vector of shape (M,)

        Returns:
            Fitted regressor with a sklearn-compatible .predict() method.
        """
        # if self._binary:
        #     # rgs = Ridge(alpha=10)
        #     rgs = RidgeCV(alphas=np.logspace(-1, 1, 5))
        # else:
        #     rgs = RidgeCV(alphas=np.logspace(-2, 2, 9))
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        rgs = RidgeCV(alphas=alphas)
        rgs.fit(X, y.reshape(-1, 1))
        self.alpha_ = float(rgs.alpha_)
        self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
        return rgs

    @staticmethod
    def _mutual_information_discrete_discrete(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Mutual Information between two binary response patterns.

        Args:
            x1: First binary array (0/1 valued)
            x2: Second binary array (0/1 valued)

        Returns:
            Mutual information value (non-negative)
        """
        n = len(x1)
        if n == 0:
            return 0.0

        s1 = x1.sum()
        s2 = x2.sum()
        s11 = np.dot(x1, x2)

        p1 = np.array([n - s1, s1]) / n
        p2 = np.array([n - s2, s2]) / n
        p_joint = np.array([
            [n - s1 - s2 + s11, s2 - s11],
            [s1 - s11,          s11],
        ]) / n

        mi = 0.0
        for i in range(2):
            for j in range(2):
                if p_joint[i, j] > 0:
                    mi += p_joint[i, j] * np.log(p_joint[i, j] / (p1[i] * p2[j] + 1e-15))
        return max(0.0, mi)

    @staticmethod
    def _mutual_information_discrete_discrete_batch(
        X: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:
        """Vectorised MI between each column of X and a single vector x2 (all binary).

        Args:
            X: Binary matrix of shape (n_samples, n_features)
            x2: Single binary vector of shape (n_samples,)

        Returns:
            Array of MI values of shape (n_features,)
        """
        n = X.shape[0]
        if n == 0:
            return np.zeros(X.shape[1])

        x2 = x2.astype(np.float64)

        s1 = X.sum(axis=0)            # (n_features,)
        s2 = x2.sum()                  # scalar
        s11 = X.T @ x2                 # (n_features,)

        p1_1 = s1 / n
        p1_0 = 1.0 - p1_1
        p2_1 = s2 / n
        p2_0 = 1.0 - p2_1

        # 2x2 joint probabilities for each feature
        p_11 = s11 / n
        p_10 = p1_1 - p_11
        p_01 = p2_1 - p_11
        p_00 = 1.0 - p1_1 - p2_1 + p_11

        eps = 1e-15
        mi = np.zeros(X.shape[1])
        for p_ij, m_ij in [
            (p_00, p1_0 * p2_0),
            (p_01, p1_0 * p2_1),
            (p_10, p1_1 * p2_0),
            (p_11, p1_1 * p2_1),
        ]:
            mask = p_ij > 0
            mi[mask] += p_ij[mask] * np.log(p_ij[mask] / (m_ij[mask] + eps))

        return np.maximum(mi, 0.0)

    @staticmethod
    def _mutual_information_ross_estimator(
        x: np.ndarray, y: np.ndarray, k: int = 3
    ) -> float:
        """Ross (2014) estimator for MI between binary x and continuous y.

        Args:
            x: Binary array
            y: Continuous array
            k: Number of nearest neighbors to use

        Returns:
            Mutual information estimate (non-negative)
        """
        n = len(x)
        if n == 0:
            return 0.0

        y0 = y[x == 0].reshape(-1, 1)
        y1 = y[x == 1].reshape(-1, 1)
        nx0, nx1 = len(y0), len(y1)

        # Fallback if categories are too small for k-NN
        if nx0 < k + 1 or nx1 < k + 1:
            return 0.0

        tree0 = cKDTree(y0)
        tree1 = cKDTree(y1)
        tree_all = cKDTree(y.reshape(-1, 1))

        m_list = []
        for i in range(n):
            val = y[i].reshape(1, -1)
            tree_same = tree1 if x[i] == 1 else tree0
            # Get distance to kth neighbor in the SAME class
            dist, _ = tree_same.query(val, k=k + 1)

            # Use max distance to capture all neighbors within that radius
            max_dist = np.max(dist)
            # Count neighbors in ALL classes within the same radius
            m = tree_all.query_ball_point(
                val.reshape(1, -1), max_dist - 1e-15, return_length=True
            )
            m_list.append(m)

        avg_psi_nx = (nx0 * psi(nx0) + nx1 * psi(nx1)) / n
        avg_psi_m = np.mean([psi(m) for m in m_list])
        return max(0.0, psi(n) - avg_psi_nx + psi(k) - avg_psi_m)

    @staticmethod
    def _is_binary_scores(scores: np.ndarray) -> bool:
        """Check whether a score matrix contains only binary (0/1) values."""
        finite = scores[np.isfinite(scores)]
        return finite.size > 0 and len(np.unique(finite)) <= 2

    @staticmethod
    def _mutual_information_ksg(
        x: np.ndarray, y: np.ndarray, k: int = 3
    ) -> float:
        """KSG1 (Kraskov-Stogbauer-Grassberger) estimator for MI between two
        continuous variables.

        Args:
            x: Continuous array of shape (n,)
            y: Continuous array of shape (n,)
            k: Number of nearest neighbours

        Returns:
            Mutual information estimate (non-negative)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < k + 1:
            return 0.0

        xy = np.column_stack([x, y])
        tree_xy = cKDTree(xy)
        tree_x = cKDTree(x.reshape(-1, 1))
        tree_y = cKDTree(y.reshape(-1, 1))

        dd, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
        eps = dd[:, -1]

        r = np.maximum(eps - 1e-15, 0.0)
        nx = tree_x.query_ball_point(
            x.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1
        ny = tree_y.query_ball_point(
            y.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1

        nx = np.maximum(nx, 1)
        ny = np.maximum(ny, 1)

        mi = psi(k) + psi(n) - np.mean(psi(nx+1) + psi(ny+1))
        return max(0.0, float(mi))

    @staticmethod
    def _mutual_information_ksg_batch(
        X: np.ndarray, x2: np.ndarray, k: int = 3
    ) -> np.ndarray:
        """Batch KSG1 MI between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features)
            x2: Vector of shape (n_samples,)
            k: Number of nearest neighbours

        Returns:
            Array of MI values of shape (n_features,)
        """
        n_features = X.shape[1]
        mi_values = np.empty(n_features)
        for j in range(n_features):
            mi_values[j] = MRMRPred._mutual_information_ksg(X[:, j], x2, k=k)
        return mi_values

    # ------------------------------------------------------------------
    # Gao et al. (UAI 2015) Local Gaussian Density Estimation MI
    # ------------------------------------------------------------------

    @staticmethod
    def _lgde_entropy(data: np.ndarray, k: int = 5) -> float:
        """LGDE entropy estimator (Gao, Ver Steeg & Galstyan, UAI 2015).

        At each sample point, maximises the penalised local log-likelihood
        (Gao, Ver Steeg & Galstyan, UAI 2015, Eq. 29):

            L(x, mu, Sigma) = (1/N) sum_j K_H(x_j - x) log N(x_j; mu, Sigma)
                              - N(x; mu, H + Sigma)

        where K_H is a Gaussian kernel with per-point bandwidth matrix
        H_i = diag(h_i^2) and h_i = distance from point i to its k-th
        nearest neighbour.  The sum is taken over k_sum neighbours
        (k_sum > k) to capture sufficient kernel mass for the k-NN
        approximation.

        Uses Cholesky parameterisation for Sigma and L-BFGS-B for
        optimisation with analytical gradient.

        Args:
            data: Array of shape (n, d) or (n,) for 1-D.
            k: Number of nearest neighbours (bandwidth & locality).

        Returns:
            Entropy estimate in nats.
        """
        from scipy.optimize import minimize

        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        if data.shape[0] == 1:
            data = data.T
        n, d = data.shape
        if n < k + 1:
            return 0.0

        k_sum = min(n - 1, max(4 * k, 50))

        tree = cKDTree(data)
        dd, ii = tree.query(data, k=k_sum + 1, p=np.inf)

        h_pp = np.maximum(dd[:, k], 1e-15)
        h2 = h_pp ** 2
        const = -0.5 * d * np.log(2 * np.pi)
        tril_idx = np.tril_indices(d)
        diag_mask = tril_idx[0] == tril_idx[1]

        # --- Batch-precompute kernel weights (per-point bandwidth) ---
        all_nbr = data[ii[:, 1:]]                                   # (n, k_sum, d)
        all_diff = all_nbr - data[:, None, :]                       # (n, k_sum, d)
        all_klog = -0.5 * np.sum(all_diff ** 2, axis=2) / h2[:, None]
        all_klog += const - 0.5 * d * np.log(h2)[:, None]
        all_w = np.exp(all_klog)                                    # (n, k_sum)
        all_w_sum = all_w.sum(axis=1)                               # (n,)

        # --- Batch-precompute weighted-MLE initialisations ---
        w_norm = all_w / np.maximum(all_w_sum[:, None], 1e-300)     # (n, k_sum)
        mu_init_all = np.einsum('nk,nkd->nd', w_norm, all_nbr)     # (n, d)
        diff_init = all_nbr - mu_init_all[:, None, :]               # (n, k_sum, d)
        Sig_init = np.einsum('nki,nkj,nk->nij', diff_init, diff_init, w_norm)
        Sig_init += 1e-6 * np.eye(d)

        try:
            L_init_all = np.linalg.cholesky(Sig_init)               # (n, d, d)
        except np.linalg.LinAlgError:
            L_init_all = np.empty((n, d, d))
            for i in range(n):
                try:
                    L_init_all[i] = np.linalg.cholesky(Sig_init[i])
                except np.linalg.LinAlgError:
                    L_init_all[i] = np.diag(np.full(d, h_pp[i]))

        log_dens = np.empty(n)

        for idx in range(n):
            xi = data[idx]
            nbr = all_nbr[idx]
            w = all_w[idx]
            w_sum = all_w_sum[idx]
            Hi = np.diag(np.full(d, h2[idx]))

            if w_sum < 1e-300:
                log_dens[idx] = -30.0
                continue

            _xi, _nbr, _w, _ws, _H = (
                xi.copy(), nbr.copy(), w.copy(), w_sum, Hi,
            )

            def _unpack(theta, _d=d, _tril=tril_idx):
                mu = theta[:_d]
                L = np.zeros((_d, _d))
                L[_tril] = theta[_d:]
                np.fill_diagonal(L, np.abs(np.diag(L)) + 1e-10)
                return mu, L

            def _obj_and_grad(theta):
                mu, L = _unpack(theta)
                Sigma = L @ L.T

                diff_mu = _nbr - mu
                try:
                    S_inv = np.linalg.inv(Sigma)
                except np.linalg.LinAlgError:
                    return 1e10, np.zeros_like(theta)
                sign, logdet = np.linalg.slogdet(Sigma)
                if sign <= 0:
                    return 1e10, np.zeros_like(theta)

                mahal = np.sum(diff_mu @ S_inv * diff_mu, axis=1)
                gauss_log = const - 0.5 * logdet - 0.5 * mahal
                weighted_ll = np.sum(_w * gauss_log) / n

                P = _H + Sigma
                diff_pen = _xi - mu
                try:
                    sign_p, logdet_p = np.linalg.slogdet(P)
                    if sign_p <= 0:
                        return 1e10, np.zeros_like(theta)
                    P_inv = np.linalg.inv(P)
                except np.linalg.LinAlgError:
                    return 1e10, np.zeros_like(theta)
                pen_mahal = diff_pen @ P_inv @ diff_pen
                penalty = np.exp(const - 0.5 * logdet_p - 0.5 * pen_mahal)

                obj = -(weighted_ll - penalty)

                # --- Analytical gradient ---
                w_diff = _w @ diff_mu
                grad_mu = -(1.0 / n) * (S_inv @ w_diff) + penalty * (P_inv @ diff_pen)

                W_scatter = (diff_mu * _w[:, None]).T @ diff_mu
                dobj_dS = (0.5 / n) * (_ws * S_inv - S_inv @ W_scatter @ S_inv)
                P_inv_dp = P_inv @ diff_pen
                dobj_dS += penalty * 0.5 * (np.outer(P_inv_dp, P_inv_dp) - P_inv)

                dobj_dL = 2.0 * dobj_dS @ L
                L_grad_tril = dobj_dL[tril_idx]
                L_grad_tril[diag_mask] *= np.sign(theta[d:][diag_mask])

                grad = np.empty_like(theta)
                grad[:d] = grad_mu
                grad[d:] = L_grad_tril
                return obj, grad

            theta0 = np.concatenate([mu_init_all[idx], L_init_all[idx][tril_idx]])
            res = minimize(_obj_and_grad, theta0, method="L-BFGS-B",
                           jac=True, options={"maxiter": 100, "ftol": 1e-10})
            mu_opt, L_opt = _unpack(res.x)
            Sigma_opt = L_opt @ L_opt.T

            diff_i = xi - mu_opt
            sign, logdet = np.linalg.slogdet(Sigma_opt)
            if sign <= 0:
                log_dens[idx] = -30.0
            else:
                try:
                    log_dens[idx] = (
                        const - 0.5 * logdet
                        - 0.5 * diff_i @ np.linalg.solve(Sigma_opt, diff_i)
                    )
                except np.linalg.LinAlgError:
                    log_dens[idx] = -30.0

        return -float(log_dens.mean())

    @staticmethod
    def _mutual_information_lgde(
        x: np.ndarray, y: np.ndarray, k: int = 5
    ) -> float:
        """MI via local Gaussian density estimation (Gao et al., UAI 2015).

        Computes MI = H(X) + H(Y) - H(X,Y) where each entropy is
        estimated using the LGDE estimator with penalised local
        log-likelihood optimisation at each sample point.

        Args:
            x: Continuous array of shape (n,).
            y: Continuous array of shape (n,).
            k: Number of nearest neighbours.

        Returns:
            Mutual information estimate (non-negative, nats).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < k + 1:
            return 0.0

        h_x = MRMRPred._lgde_entropy(x.reshape(-1, 1), k=k)
        h_y = MRMRPred._lgde_entropy(y.reshape(-1, 1), k=k)
        h_xy = MRMRPred._lgde_entropy(np.column_stack([x, y]), k=k)
        return max(0.0, h_x + h_y - h_xy)

    @staticmethod
    def _mutual_information_lgde_batch(
        X: np.ndarray, x2: np.ndarray, k: int = 5
    ) -> np.ndarray:
        """Batch LGDE MI between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features).
            x2: Vector of shape (n_samples,).
            k: Number of nearest neighbours.

        Returns:
            Array of MI values of shape (n_features,).
        """
        n_features = X.shape[1]
        mi_values = np.empty(n_features)
        for j in range(n_features):
            mi_values[j] = MRMRPred._mutual_information_lgde(X[:, j], x2, k=k)
        return mi_values

    # ------------------------------------------------------------------
    # Quick Gaussian Density Estimation (QGDE) – KDE balloon estimator
    # ------------------------------------------------------------------

    @staticmethod
    def _qgde_entropy(data: np.ndarray, k: int = 5) -> float:
        """Quick Gaussian density entropy estimator via KDE.

        Uses a leave-one-out Gaussian KDE with per-point (balloon)
        bandwidth h_i equal to the Chebyshev distance from point i to
        its k-th nearest neighbour.  Fully vectorised — no Python loops.

        Args:
            data: Array of shape (n, d) or (n,) for 1-D.
            k: Number of nearest neighbours for bandwidth selection.

        Returns:
            Entropy estimate in nats.
        """
        from scipy.special import logsumexp

        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        if data.shape[0] == 1:
            data = data.T
        n, d = data.shape
        if n < k + 1:
            return 0.0

        tree = cKDTree(data)
        dd, _ = tree.query(data, k=k + 1, p=np.inf)
        h = np.maximum(dd[:, k], 1e-15)                              # (n,)

        const = -0.5 * d * np.log(2 * np.pi)

        # Pairwise squared distances: (n, n)
        sq_dist = np.sum(
            (data[:, None, :] - data[None, :, :]) ** 2, axis=2,
        )

        # log N(x_i; x_j, h_i^2 I) for each (i, j) pair
        log_k = const - d * np.log(h)[:, None] - 0.5 * sq_dist / (h ** 2)[:, None]
        np.fill_diagonal(log_k, -np.inf)

        log_dens = logsumexp(log_k, axis=1) - np.log(n - 1)

        return -float(log_dens.mean())

    @staticmethod
    def _mutual_information_qgde(
        x: np.ndarray, y: np.ndarray, k: int = 5
    ) -> float:
        """MI via quick Gaussian density estimation (closed-form weighted MLE).

        Computes MI = H(X) + H(Y) - H(X,Y) where each entropy is
        estimated using the QGDE estimator.

        Args:
            x: Continuous array of shape (n,).
            y: Continuous array of shape (n,).
            k: Number of nearest neighbours.

        Returns:
            Mutual information estimate (non-negative, nats).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < k + 1:
            return 0.0

        h_x = MRMRPred._qgde_entropy(x.reshape(-1, 1), k=k)
        h_y = MRMRPred._qgde_entropy(y.reshape(-1, 1), k=k)
        h_xy = MRMRPred._qgde_entropy(np.column_stack([x, y]), k=k)
        return max(0.0, h_x + h_y - h_xy)

    @staticmethod
    def _mutual_information_qgde_batch(
        X: np.ndarray, x2: np.ndarray, k: int = 5
    ) -> np.ndarray:
        """Batch QGDE MI between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features).
            x2: Vector of shape (n_samples,).
            k: Number of nearest neighbours.

        Returns:
            Array of MI values of shape (n_features,).
        """
        n_features = X.shape[1]
        mi_values = np.empty(n_features)
        for j in range(n_features):
            mi_values[j] = MRMRPred._mutual_information_qgde(X[:, j], x2, k=k)
        return mi_values

    # ------------------------------------------------------------------
    # Gao et al. (AISTATS 2015) PCA-based Local Nonuniformity Correction
    # ------------------------------------------------------------------

    @staticmethod
    def _mutual_information_lnc(
        x: np.ndarray, y: np.ndarray, k: int = 5, alpha: float = 0.25
    ) -> float:
        """MI via KSG + PCA local nonuniformity correction (LNC).

        Gao, Ver Steeg & Galstyan (AISTATS 2015) correct the KSG
        estimate by replacing the axis-aligned max-norm rectangle with a
        PCA-aligned rectangle at each point.  When the PCA volume is
        much smaller than the max-norm volume (ratio < alpha), a
        positive correction is applied.

        Args:
            x: Continuous array of shape (n,).
            y: Continuous array of shape (n,).
            k: Number of nearest neighbours.
            alpha: Threshold for local nonuniformity test.

        Returns:
            Mutual information estimate (non-negative, nats).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < k + 1:
            return 0.0

        intens = 1e-10
        x = x + intens * np.random.default_rng(0).standard_normal(n)
        y = y + intens * np.random.default_rng(1).standard_normal(n)

        xy = np.column_stack([x, y])
        tree = cKDTree(xy)
        dd, ii = tree.query(xy, k=k + 1, p=np.inf)
        eps = dd[:, -1]

        # Gather all neighbour patches: (n, k+1, 2)
        all_nbr = xy[ii]
        centres = all_nbr[:, 0:1, :]
        centred = all_nbr - centres

        # Per-axis max distances (needed for LNC rectangle volume)
        dvec_x = np.maximum(np.max(np.abs(centred[:, :, 0]), axis=1), 1e-15)
        dvec_y = np.maximum(np.max(np.abs(centred[:, :, 1]), axis=1), 1e-15)

        # KSG1 MI — use eps (Chebyshev distance) for both marginal counts
        r = np.maximum(eps - 1e-15, 0.0)
        tree_x = cKDTree(x.reshape(-1, 1))
        tree_y = cKDTree(y.reshape(-1, 1))
        nx = tree_x.query_ball_point(
            x.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1
        ny = tree_y.query_ball_point(
            y.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1
        nx = np.maximum(nx, 1)
        ny = np.maximum(ny, 1)
        mi_ksg = float(
            psi(k) + psi(n) - np.mean(psi(nx + 1) + psi(ny + 1))
        )

        # LNC correction — vectorised
        nbr_only = centred[:, 1:, :]                          # (n, k, 2)
        cov = np.einsum('nki,nkj->nij', nbr_only, nbr_only)  # (n, 2, 2)
        cov /= k

        _, eigvecs = np.linalg.eigh(cov)                      # (n, 2, 2)
        projected = np.einsum('nki,nij->nkj', centred, eigvecs)  # (n, k+1, 2)
        max_proj = np.maximum(np.max(np.abs(projected), axis=1), 1e-30)  # (n, 2)

        log_V_pca = np.sum(np.log(max_proj), axis=1)          # (n,)
        log_V_rect = np.log(dvec_x) + np.log(dvec_y)          # (n,)

        log_alpha = np.log(max(alpha, 1e-30))
        lnc_mask = log_V_pca < log_V_rect + log_alpha
        correction = np.sum(log_V_rect[lnc_mask] - log_V_pca[lnc_mask]) / n

        return max(0.0, mi_ksg + correction)

    @staticmethod
    def _mutual_information_lnc_batch(
        X: np.ndarray, x2: np.ndarray, k: int = 5, alpha: float = 0.25
    ) -> np.ndarray:
        """Batch LNC MI between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features).
            x2: Vector of shape (n_samples,).
            k: Number of nearest neighbours.
            alpha: LNC threshold parameter.

        Returns:
            Array of MI values of shape (n_features,).
        """
        n_features = X.shape[1]
        mi_values = np.empty(n_features)
        for j in range(n_features):
            mi_values[j] = MRMRPred._mutual_information_lnc(
                X[:, j], x2, k=k, alpha=alpha
            )
        return mi_values

    def _select_target(
        self, source_full_scores: np.ndarray, seed: int = 42, irt_dims: int = 5
    ) -> np.ndarray:
        """Select relevance target based on configuration.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            seed: Random seed (used for IRT training)
            irt_dims: Number of IRT latent dimensions (only used when target_scheme="irt")

        Returns:
            Target vector of shape (M models,)
        """
        if source_full_scores is None or len(source_full_scores) == 0:
            raise ValueError("source_full_scores cannot be None or empty")

        if self.target_scheme == "pc1":
            # Use first principal component as target
            pca = PCA(n_components=1)
            return pca.fit_transform(source_full_scores).flatten()
        elif self.target_scheme == "y+pc1":
            # Use normalized sum of average score and PC1 as target.
            # Average score is already in [0, 1]; min-max normalize PC1
            # to [0, 1] so both terms contribute roughly equally.
            avg_score = source_full_scores.mean(-1)
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(source_full_scores).flatten()
            pc1_min, pc1_max = pc1.min(), pc1.max()
            if pc1_max - pc1_min > 1e-15:
                pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min)
            else:
                pc1_norm = np.zeros_like(pc1)
            return avg_score + pc1_norm
        elif self.target_scheme == "irt":
            # Use mean IRT ability as target
            return self._compute_irt_ability(
                source_full_scores, seed=seed, dims=irt_dims
            )
        else:
            # Use average score as target
            return source_full_scores.mean(-1)

    def _compute_irt_ability(
        self, source_full_scores: np.ndarray, seed: int = 42, dims: int = 5
    ) -> np.ndarray:
        """Run IRT inference and return mean ability per model.

        Fits a multidimensional 2PL IRT model on the source scores and extracts
        the per-model ability parameter, averaged across latent dimensions.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            seed: Random seed for reproducibility
            dims: Number of IRT latent dimensions

        Returns:
            Mean ability per model, shape (M models,)
        """
        config = IrtConfig(
            priors="hierarchical",
            dims=dims,
            lr=0.1,
            epochs=2000,
            model_type="multidim_2pl",
            dropout=0.5,
            hidden=100,
            log_every=200,
            deterministic=True,
            seed=seed,
        )

        trainer = IrtModelTrainer(
            config=config,
            dataset=create_irt_dataset(source_full_scores),
            verbose=True,
        )
        trainer.train(device="cpu")
        params = trainer.best_params

        # ability shape: [num_models, dims]; average across dims for a scalar per model
        ability = np.array(params["ability"])
        return ability.mean(axis=1)

    def _get_mi_estimators(self, _binary: bool):
        """Return (relevance_fn, redundancy_batch_fn) for MI estimation.

        Override in subclasses to swap the MI estimator (e.g. local Gaussian).

        Args:
            _binary: True when the score matrix contains only binary values.

        Returns:
            Tuple of (relevance_fn, redundancy_batch_fn).
            - relevance_fn(x, y, k=...) -> float
            - redundancy_batch_fn(X, x2) -> np.ndarray
        """
        _mi_rel = (
            self._mutual_information_ross_estimator if _binary
            else self._mutual_information_ksg
        )
        _mi_red_batch = (
            self._mutual_information_discrete_discrete_batch if _binary
            else lambda X, x2: self._mutual_information_ksg_batch(X, x2, k=self.mi_k)
        )
        return _mi_rel, _mi_red_batch

    def fit(
        self,
        source_full_scores: np.ndarray,
        coreset_size: int,
        seed: int = 42,
        target_scheme: str = "average",
        miq_scheme: bool = False,
        only_relevance: bool = False,
        irt_dims: int = 5,
        *args,
        **kwargs,
    ) -> "MRMRPred":
        """Fit MRMR feature selection and Ridge regression model.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            coreset_size: Number of features to select
            seed: Random seed for reproducibility
            target_scheme: "average", "pc1", or "irt" for relevance target selection
            miq_scheme: False for MID, True for MIQ scheme
            only_relevance: If True, greedily maximise relevance (MI with target only);
                redundancy is neither computed nor used (MID with redundancy 0 /
                MIQ with fixed redundancy denominator).
            irt_dims: Number of IRT latent dimensions (only used when target_scheme="irt")

        Returns:
            self
        """
        # Validate input
        if source_full_scores is None:
            raise ValueError("source_full_scores cannot be None")

        if len(source_full_scores.shape) < 2:
            raise ValueError("source_full_scores must be a 2D array")

        num_model, num_data = source_full_scores.shape

        if num_model <= 1:
            raise ValueError("Need at least 2 models")
        if num_data <= 1:
            raise ValueError("Need at least 2 data points")
        if coreset_size <= 1:
            raise ValueError("coreset_size must be > 1")
        if num_data <= coreset_size:
            raise ValueError("num_data must be > coreset_size")

        set_random_seed(seed)
        self.target_scheme = target_scheme
        self.miq_scheme = miq_scheme
        self.only_relevance = only_relevance

        # Detect binary vs continuous features and choose MI estimators
        _binary = self._is_binary_scores(source_full_scores)
        self._binary = _binary

        # Mean imputation if there are any NaNs
        if not _binary and np.any(np.isnan(source_full_scores)):
            source_full_scores = source_full_scores.copy()
            col_means = np.nanmean(source_full_scores, axis=0)
            inds = np.where(np.isnan(source_full_scores))
            source_full_scores[inds] = col_means[inds[1]]

        _mi_rel, _mi_red_batch = self._get_mi_estimators(_binary)

        # Select target for relevance computation (may be PC1, IRT ability, or mean)
        relevance_target = self._select_target(
            source_full_scores, seed=seed, irt_dims=irt_dims
        )

        # Regression target is always mean score
        regression_target = source_full_scores.mean(-1)

        # Precompute relevance for every point (MI with relevance_target); it does not change during selection
        relevance_per_idx = np.array(
            [
                _mi_rel(
                    source_full_scores[:, idx], relevance_target, k=self.mi_k
                )
                for idx in tqdm(
                    range(num_data), desc="Relevance (MI)", unit="point"
                )
            ]
        )

        if only_relevance:
            remaining_set: set[int] = set(range(num_data))
            selected_indices_flat: list[int] = []
            coreset_rel_only: list[float] = []
            coreset_red_only: list[float] = []
            for _ in tqdm(
                range(coreset_size),
                desc="MRMR relevance-only",
                unit="feature",
            ):
                rem_arr = np.array(sorted(remaining_set))
                pick = int(rem_arr[int(np.argmax(relevance_per_idx[rem_arr]))])
                selected_indices_flat.append(pick)
                remaining_set.remove(pick)
                coreset_rel_only.append(float(relevance_per_idx[pick]))
                coreset_red_only.append(0.0)
            self.compressed_data_indices = np.array(selected_indices_flat)
            self.selection_metrics = {
                "all_relevance_min": float(np.min(relevance_per_idx)),
                "all_relevance_max": float(np.max(relevance_per_idx)),
                "all_relevance_mean": float(np.mean(relevance_per_idx)),
                "all_relevance_median": float(np.median(relevance_per_idx)),
                "all_relevance_std": float(np.std(relevance_per_idx)),
                "coreset_relevance": coreset_rel_only,
                "coreset_redundancy": coreset_red_only,
            }
            X = source_full_scores[:, self.compressed_data_indices]
            self.rgs = self._build_regressor(X, regression_target)
            return self

        # Global relevance statistics (across ALL questions)
        coreset_relevance = []
        coreset_redundancy = []

        # Initialize selection variables
        selected_indices = []
        remaining_indices = set(range(num_data))

        # Running sum of MI(idx, selected) for each candidate; updated incrementally
        redundancy_sum = np.zeros(num_data)

        # First iteration - select feature with highest relevance to target
        best_idx = int(np.argmax(relevance_per_idx))
        selected_indices.append(best_idx)
        remaining_indices.discard(best_idx)
        coreset_relevance.append(float(relevance_per_idx[best_idx]))
        coreset_redundancy.append(0.0)

        # Update redundancy sums: add MI(each remaining, newly selected) for the first selection
        remaining_arr = np.array(sorted(remaining_indices))
        redundancy_sum[remaining_arr] += _mi_red_batch(
            source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
        )

        # Greedy selection of remaining features
        for _ in tqdm(
            range(1, coreset_size),
            desc="MRMR selection",
            unit="feature",
        ):
            num_selected = len(selected_indices)
            remaining_arr = np.array(sorted(remaining_indices))

            mean_red = redundancy_sum[remaining_arr] / num_selected
            rel = relevance_per_idx[remaining_arr]
            if self.miq_scheme:
                scores = rel / (mean_red + 1e-10)
            else:
                scores = rel - mean_red

            winner_pos = np.argmax(scores)
            winner = remaining_arr[winner_pos]
            best_idx = int(winner)
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            coreset_relevance.append(float(rel[winner_pos]))
            coreset_redundancy.append(float(mean_red[winner_pos]))

            remaining_arr = np.array(sorted(remaining_indices))
            if len(remaining_arr) > 0:
                redundancy_sum[remaining_arr] += _mi_red_batch(
                    source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
                )

        self.compressed_data_indices = np.array(selected_indices)
        self.selection_metrics = {
            "all_relevance_min": float(np.min(relevance_per_idx)),
            "all_relevance_max": float(np.max(relevance_per_idx)),
            "all_relevance_mean": float(np.mean(relevance_per_idx)),
            "all_relevance_median": float(np.median(relevance_per_idx)),
            "all_relevance_std": float(np.std(relevance_per_idx)),
            "coreset_relevance": coreset_relevance,
            "coreset_redundancy": coreset_redundancy,
        }

        # Train regressor on selected features (always predicting mean score)
        X = source_full_scores[:, self.compressed_data_indices]
        self.rgs = self._build_regressor(X, regression_target)

        return self

    def get_coreset(self) -> np.ndarray:
        """Get the selected feature indices.

        Returns:
            Array of selected feature indices
        """
        if self.compressed_data_indices is None:
            raise ValueError("Model has not been fitted yet")
        return self.compressed_data_indices

    def predict(self, target_coreset_scores: np.ndarray) -> np.ndarray:
        """Predict using the trained Ridge model.

        Args:
            target_coreset_scores: Matrix of shape (M_target models, K coreset features)

        Returns:
            Predicted scores
        """
        if target_coreset_scores is None:
            raise ValueError("target_coreset_scores cannot be None")

        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores).ravel()

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        return self._build_regressor(X, y)

    def save(self, path_save: str) -> "MRMRPred":
        """Save the model to disk.

        Args:
            path_save: Path to save the model

        Returns:
            self
        """
        if path_save is None:
            raise ValueError("path_save cannot be None")

        jbl.dump((self.compressed_data_indices, self.rgs), path_save)
        return self

    def load(self, path_load: str) -> "MRMRPred":
        """Load the model from disk.

        Args:
            path_load: Path to load the model from

        Returns:
            self
        """
        if path_load is None:
            raise ValueError("path_load cannot be None")

        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class MRMRPred_MID_y(MRMRPred):
    """MRMR with MID scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MID_PC1(MRMRPred):
    """MRMR with MID scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_y(MRMRPred):
    """MRMR with MIQ scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_PC1(MRMRPred):
    """MRMR with MIQ scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_MI_y(MRMRPred):
    """MRMR relevance-only: greedily maximise MI to the target (no redundancy)."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["only_relevance"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_MID_yPC1(MRMRPred):
    """MRMR with MID scheme and normalized sum of avg score + PC1 as relevance target.

    The relevance target is avg_score + normalize(PC1), where PC1 is min-max
    scaled to [0, 1] so both terms contribute roughly equally.  The regression
    target remains the average score.
    """

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_yPC1(MRMRPred):
    """MRMR with MIQ scheme and normalized sum of avg score + PC1 as relevance target.

    The relevance target is avg_score + normalize(PC1), where PC1 is min-max
    scaled to [0, 1] so both terms contribute roughly equally.  The regression
    target remains the average score.
    """

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class AntiMRMRPred(MRMRPred):
    """Anti-MRMR: ignores redundancy and MINIMISES relevance.

    Selects features whose MI with the target is lowest, producing a coreset
    of maximally *irrelevant* questions.  Useful as a negative-control
    baseline to verify that relevance-based selection is actually helping.
    """

    def fit(
        self,
        source_full_scores: np.ndarray,
        coreset_size: int,
        seed: int = 42,
        target_scheme: str = "average",
        miq_scheme: bool = False,
        irt_dims: int = 5,
        *args,
        **kwargs,
    ) -> "AntiMRMRPred":
        if source_full_scores is None:
            raise ValueError("source_full_scores cannot be None")
        if len(source_full_scores.shape) < 2:
            raise ValueError("source_full_scores must be a 2D array")

        num_model, num_data = source_full_scores.shape

        if num_model <= 1:
            raise ValueError("Need at least 2 models")
        if num_data <= 1:
            raise ValueError("Need at least 2 data points")
        if coreset_size <= 1:
            raise ValueError("coreset_size must be > 1")
        if num_data <= coreset_size:
            raise ValueError("num_data must be > coreset_size")

        set_random_seed(seed)
        self.target_scheme = target_scheme
        self.miq_scheme = miq_scheme

        _binary = self._is_binary_scores(source_full_scores)
        self._binary = _binary

        if not _binary and np.any(np.isnan(source_full_scores)):
            source_full_scores = source_full_scores.copy()
            col_means = np.nanmean(source_full_scores, axis=0)
            inds = np.where(np.isnan(source_full_scores))
            source_full_scores[inds] = col_means[inds[1]]

        _mi_rel, _ = self._get_mi_estimators(_binary)

        relevance_target = self._select_target(
            source_full_scores, seed=seed, irt_dims=irt_dims
        )
        regression_target = source_full_scores.mean(-1)

        relevance_per_idx = np.array(
            [
                _mi_rel(
                    source_full_scores[:, idx], relevance_target, k=self.mi_k
                )
                for idx in tqdm(
                    range(num_data), desc="Relevance (MI)", unit="point"
                )
            ]
        )

        # Greedily pick features with LOWEST relevance (no redundancy term)
        coreset_relevance = []
        selected_indices = []
        remaining_indices = set(range(num_data))

        for _ in tqdm(
            range(coreset_size),
            desc="Anti-MRMR selection",
            unit="feature",
        ):
            remaining_arr = np.array(sorted(remaining_indices))
            rel = relevance_per_idx[remaining_arr]
            winner_pos = np.argmin(rel)
            winner = remaining_arr[winner_pos]
            best_idx = int(winner)
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            coreset_relevance.append(float(rel[winner_pos]))

        self.compressed_data_indices = np.array(selected_indices)
        self.selection_metrics = {
            "all_relevance_min": float(np.min(relevance_per_idx)),
            "all_relevance_max": float(np.max(relevance_per_idx)),
            "all_relevance_mean": float(np.mean(relevance_per_idx)),
            "all_relevance_median": float(np.median(relevance_per_idx)),
            "all_relevance_std": float(np.std(relevance_per_idx)),
            "coreset_relevance": coreset_relevance,
            "coreset_redundancy": [0.0] * coreset_size,
        }

        X = source_full_scores[:, self.compressed_data_indices]
        self.rgs = self._build_regressor(X, regression_target)

        return self


class AntiMRMRPred_MID_y(AntiMRMRPred):
    """Anti-MRMR with MID scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class AntiMRMRPred_MIQ_y(AntiMRMRPred):
    """Anti-MRMR with MIQ scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPredAIPW(MRMRPred):
    """MRMR feature selection with AIPW (prediction-powered inference) prediction.

    Uses the same MRMR greedy feature selection as the base class, but replaces
    the Lasso regression with the AIPW prediction scheme: for each target model,
    a Ridge regression is fitted from coreset questions to non-coreset questions
    using source model scores, and a PPI bias correction is applied.
    """

    def __init__(self):
        super().__init__()
        self.source_full_scores = None

    def fit(
        self,
        source_full_scores: np.ndarray,
        coreset_size: int,
        seed: int = 42,
        target_scheme: str = "average",
        miq_scheme: bool = False,
        *args,
        **kwargs,
    ) -> "MRMRPredAIPW":
        """Fit MRMR feature selection (no Lasso); store source scores for AIPW predict.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            coreset_size: Number of features to select
            seed: Random seed for reproducibility
            target_scheme: "average" or "pc1" for target selection
            miq_scheme: False for MID, True for MIQ scheme

        Returns:
            self
        """
        # Run full MRMR selection (Lasso is trained but will be unused)
        super().fit(
            source_full_scores,
            coreset_size,
            seed=seed,
            target_scheme=target_scheme,
            miq_scheme=miq_scheme,
            *args,
            **kwargs,
        )
        # Store source scores for AIPW prediction at inference time
        self.source_full_scores = source_full_scores
        return self

    def predict(self, target_coreset_scores: np.ndarray) -> np.ndarray:
        """Predict using the AIPW / prediction-powered inference scheme.

        For each target model, fits a Ridge regression from source model behaviour
        on coreset questions to predict behaviour on non-coreset questions, then
        applies a PPI bias correction.

        Args:
            target_coreset_scores: Matrix of shape (M_target models, K coreset features)

        Returns:
            Predicted average scores for each target model
        """
        if target_coreset_scores is None:
            raise ValueError("target_coreset_scores cannot be None")

        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        num_data = self.source_full_scores.shape[1]
        rest_indices = np.array(
            [i for i in range(num_data) if i not in self.compressed_data_indices]
        )

        ret = []
        num_target_models = target_coreset_scores.shape[0]
        for i in range(num_target_models):
            # X rows = questions, columns = source model scores
            x_train = self.source_full_scores[:, self.compressed_data_indices].T
            y_train = target_coreset_scores[i]
            x_test = self.source_full_scores[:, rest_indices].T

            rgs = self._build_regressor(x_train, y_train)

            y_pred_train = rgs.predict(x_train).squeeze()
            y_pred_test = rgs.predict(x_test).squeeze()

            # PPI bias correction
            n = len(self.compressed_data_indices)
            N = len(rest_indices)
            ppi_part = (y_train - y_pred_train).mean() / (1 + n / N)
            ppi_part += y_pred_test.mean()
            ret.append(ppi_part)

        return np.array(ret)

    def save(self, path_save: str) -> "MRMRPredAIPW":
        if path_save is None:
            raise ValueError("path_save cannot be None")
        jbl.dump(
            (self.compressed_data_indices, self.source_full_scores), path_save
        )
        return self

    def load(self, path_load: str) -> "MRMRPredAIPW":
        if path_load is None:
            raise ValueError("path_load cannot be None")
        self.compressed_data_indices, self.source_full_scores = jbl.load(path_load)
        return self


class MRMRPred_MID_y_aipw(MRMRPredAIPW):
    """MRMR with MID scheme, average score target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MID_PC1_aipw(MRMRPredAIPW):
    """MRMR with MID scheme, PC1 target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_y_aipw(MRMRPredAIPW):
    """MRMR with MIQ scheme, average score target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_PC1_aipw(MRMRPredAIPW):
    """MRMR with MIQ scheme, PC1 target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


# --- IRT ability variants (Ridge prediction) ---


class MRMRPred_MID_IRT1(MRMRPred):
    """MRMR with MID scheme and 1-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


class MRMRPred_MID_IRT5(MRMRPred):
    """MRMR with MID scheme and 5-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 5
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_IRT1(MRMRPred):
    """MRMR with MIQ scheme and 1-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_IRT5(MRMRPred):
    """MRMR with MIQ scheme and 5-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 5
        return super().fit(*args, **kwargs)


# --- IRT ability variants (AIPW prediction) ---


class MRMRPred_MID_IRT1_aipw(MRMRPredAIPW):
    """MRMR with MID scheme, 1-dim IRT ability relevance target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


class MRMRPred_MID_IRT5_aipw(MRMRPredAIPW):
    """MRMR with MID scheme, 5-dim IRT ability relevance target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 5
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_IRT1_aipw(MRMRPredAIPW):
    """MRMR with MIQ scheme, 1-dim IRT ability relevance target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_IRT5_aipw(MRMRPredAIPW):
    """MRMR with MIQ scheme, 5-dim IRT ability relevance target, and AIPW prediction."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 5
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# F-statistic / Pearson-correlation variants (FCD and FCQ)
# ---------------------------------------------------------------------------
# These replace the MI-based relevance and redundancy with:
#   - Relevance: one-way ANOVA F-statistic between binary question and
#     continuous target (y, PC1, or IRT ability).
#   - Redundancy: absolute Pearson correlation between pairs of binary
#     questions.
# FCD uses the difference scheme (like MID), FCQ uses the quotient scheme
# (like MIQ).  The regression target is always the average score.


class MRMRFCPred(MRMRPred):
    """MRMR feature selection using F-statistic relevance and Pearson correlation redundancy.

    Instead of mutual information, this variant measures:
      - Relevance of each question to the target via the one-way ANOVA F-statistic.
      - Redundancy between questions via the absolute Pearson correlation.

    Combination schemes mirror MID/MIQ:
      - FCD (miq_scheme=False): relevance - (redundancy / num_selected)
      - FCQ (miq_scheme=True):  relevance / (redundancy + epsilon)
    """

    @staticmethod
    def _f_statistic_binary_continuous(x: np.ndarray, y: np.ndarray) -> float:
        """One-way ANOVA F-statistic between binary x and continuous y.

        Args:
            x: Binary array of shape (M,)
            y: Continuous array of shape (M,)

        Returns:
            F-statistic value (non-negative)
        """
        n = len(x)
        if n < 3:
            return 0.0
        mask0 = x == 0
        mask1 = x == 1
        n0, n1 = int(mask0.sum()), int(mask1.sum())
        if n0 < 1 or n1 < 1:
            return 0.0
        grand_mean = y.mean()
        mean0 = y[mask0].mean()
        mean1 = y[mask1].mean()
        ss_between = n0 * (mean0 - grand_mean) ** 2 + n1 * (mean1 - grand_mean) ** 2
        ss_within = ((y[mask0] - mean0) ** 2).sum() + ((y[mask1] - mean1) ** 2).sum()
        if ss_within < 1e-15:
            return 0.0
        f_stat = ss_between / (ss_within / (n - 2))
        return max(0.0, float(f_stat))

    @staticmethod
    def _f_statistic_continuous_continuous(x: np.ndarray, y: np.ndarray) -> float:
        """F-statistic for simple linear regression between continuous x and continuous y.
        
        Args:
            x: Continuous array of shape (M,)
            y: Continuous array of shape (M,)
            
        Returns:
            F-statistic value (non-negative)
        """
        n = len(x)
        # Degrees of freedom for regression is 1, for error is n - 2.
        # We need at least 3 points to have at least 1 degree of freedom for error.
        if n < 3:
            return 0.0

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # np.corrcoef divides by std dev internally; constant x or y gives 0/0 and
        # raises "invalid value encountered in divide" before we can use the result.
        if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
            return 0.0

        # Calculate the Pearson correlation coefficient
        # np.corrcoef returns a correlation matrix; we take the top-right value
        r_matrix = np.corrcoef(x, y)
        
        # Handle cases where x or y are constant (variance is 0)
        if np.any(np.isnan(r_matrix)):
            return 0.0
            
        r = r_matrix[0, 1]
        if not np.isfinite(r):
            return 0.0
        r2 = float(r) ** 2
        if not np.isfinite(r2):
            return 0.0

        # If the correlation is perfect (r2 = 1), the F-statistic is technically infinite.
        # Use a very large finite value to preserve "max relevance" behaviour.
        if r2 >= 1.0 - 1e-12:
            return 1e15

        # F-statistic formula derived from R-squared:
        # F = (R^2 / k) / ((1 - R^2) / (n - k - 1))
        # For simple linear regression, k (number of predictors) = 1.
        denom = max(1.0 - r2, 1e-15)
        f_stat = r2 * (n - 2) / denom

        return max(0.0, float(f_stat))

    @staticmethod
    def _squared_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Squared Pearson correlation (r^2) between two continuous arrays.

        Generalises the F-statistic relevance to continuous features: for binary
        x, r^2 is a monotone transform of the F-statistic, so the feature ranking
        is preserved.

        Args:
            x: Array of shape (M,)
            y: Array of shape (M,)

        Returns:
            r^2 in [0, 1]
        """
        if len(x) < 2:
            return 0.0
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            return 0.0
        return float(r) ** 2

    @staticmethod
    def _abs_pearson_correlation(x1: np.ndarray, x2: np.ndarray) -> float:
        """Absolute Pearson correlation between two arrays.

        Args:
            x1: First array of shape (M,)
            x2: Second array of shape (M,)

        Returns:
            |r| in [0, 1]
        """
        if len(x1) < 2:
            return 0.0
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        if np.ptp(x1) <= 1e-15 or np.ptp(x2) <= 1e-15:
            return 0.0
        r = np.corrcoef(x1, x2)[0, 1]
        if np.isnan(r):
            return 0.0
        return abs(float(r))

    @staticmethod
    def _abs_pearson_correlation_batch(
        X: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:
        """Vectorised |Pearson r| between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features)
            x2: Vector of shape (n_samples,)

        Returns:
            Array of |r| values of shape (n_features,)
        """
        n = X.shape[0]
        if n < 2:
            return np.zeros(X.shape[1])

        x2c = x2 - x2.mean()
        Xc = X - X.mean(axis=0)

        denom_x2 = np.sqrt(np.dot(x2c, x2c))
        if denom_x2 < 1e-15:
            return np.zeros(X.shape[1])

        denom_X = np.sqrt((Xc ** 2).sum(axis=0))
        numer = Xc.T @ x2c

        safe = denom_X > 1e-15
        r = np.zeros(X.shape[1])
        r[safe] = numer[safe] / (denom_X[safe] * denom_x2)
        return np.abs(r)

    def fit(
        self,
        source_full_scores: np.ndarray,
        coreset_size: int,
        seed: int = 42,
        target_scheme: str = "average",
        miq_scheme: bool = False,
        irt_dims: int = 5,
        *args,
        **kwargs,
    ) -> "MRMRFCPred":
        """Fit MRMR-FC feature selection and Ridge regression model.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            coreset_size: Number of features to select
            seed: Random seed for reproducibility
            target_scheme: "average", "pc1", or "irt" for relevance target selection
            miq_scheme: False for FCD, True for FCQ scheme
            irt_dims: Number of IRT latent dimensions (only used when target_scheme="irt")

        Returns:
            self
        """
        if source_full_scores is None:
            raise ValueError("source_full_scores cannot be None")
        if len(source_full_scores.shape) < 2:
            raise ValueError("source_full_scores must be a 2D array")

        num_model, num_data = source_full_scores.shape

        if num_model <= 1:
            raise ValueError("Need at least 2 models")
        if num_data <= 1:
            raise ValueError("Need at least 2 data points")
        if coreset_size <= 1:
            raise ValueError("coreset_size must be > 1")
        if num_data <= coreset_size:
            raise ValueError("num_data must be > coreset_size")

        set_random_seed(seed)
        self.target_scheme = target_scheme
        self.miq_scheme = miq_scheme

        _binary = self._is_binary_scores(source_full_scores)
        self._binary = _binary

        if not _binary and np.any(np.isnan(source_full_scores)):
            source_full_scores = source_full_scores.copy()
            col_means = np.nanmean(source_full_scores, axis=0)
            inds = np.where(np.isnan(source_full_scores))
            source_full_scores[inds] = col_means[inds[1]]

        # Select relevance target (reuses the same logic as MI-based MRMR)
        relevance_target = self._select_target(
            source_full_scores, seed=seed, irt_dims=irt_dims
        )

        # Regression target is always mean score
        regression_target = source_full_scores.mean(-1)

        _linear_f_all = getattr(
            type(self), "_force_linear_fc_relevance_all", False
        )
        _linear_f_binary = getattr(
            type(self), "_force_linear_fc_binary_relevance", False
        )
        if _linear_f_all:
            _rel_fn = self._f_statistic_continuous_continuous
            _rel_label = "Relevance (F-stat lin)"
        elif _binary:
            _rel_fn = (
                self._f_statistic_continuous_continuous
                if _linear_f_binary
                else self._f_statistic_binary_continuous
            )
            _rel_label = (
                "Relevance (F-stat lin)" if _linear_f_binary else "Relevance (F-stat)"
            )
        else:
            _rel_fn = self._squared_pearson_correlation
            _rel_label = "Relevance (r²)"

        # Precompute relevance for every question
        relevance_per_idx = np.array(
            [
                _rel_fn(
                    source_full_scores[:, idx], relevance_target
                )
                for idx in tqdm(
                    range(num_data), desc=_rel_label, unit="point"
                )
            ]
        )

        # Global relevance statistics (across ALL questions)
        coreset_relevance = []
        coreset_redundancy = []

        # Initialize selection variables
        selected_indices = []
        remaining_indices = set(range(num_data))

        # Running sum of |correlation| with selected features for each candidate
        redundancy_sum = np.zeros(num_data)

        # First iteration — select feature with highest relevance
        best_idx = int(np.argmax(relevance_per_idx))
        selected_indices.append(best_idx)
        remaining_indices.discard(best_idx)
        coreset_relevance.append(float(relevance_per_idx[best_idx]))
        coreset_redundancy.append(0.0)

        # Update redundancy sums for the first selection
        remaining_arr = np.array(sorted(remaining_indices))
        redundancy_sum[remaining_arr] += self._abs_pearson_correlation_batch(
            source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
        )

        # Greedy selection of remaining features
        for _ in tqdm(
            range(1, coreset_size),
            desc="MRMR-FC selection",
            unit="feature",
        ):
            num_selected = len(selected_indices)
            remaining_arr = np.array(sorted(remaining_indices))

            mean_red = redundancy_sum[remaining_arr] / num_selected
            rel = relevance_per_idx[remaining_arr]
            if self.miq_scheme:
                scores = rel / (mean_red + 1e-10)
            else:
                scores = rel - mean_red

            winner_pos = np.argmax(scores)
            winner = remaining_arr[winner_pos]
            best_idx = int(winner)
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            coreset_relevance.append(float(rel[winner_pos]))
            coreset_redundancy.append(float(mean_red[winner_pos]))

            remaining_arr = np.array(sorted(remaining_indices))
            if len(remaining_arr) > 0:
                redundancy_sum[remaining_arr] += self._abs_pearson_correlation_batch(
                    source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
                )

        self.compressed_data_indices = np.array(selected_indices)
        self.selection_metrics = {
            "all_relevance_min": float(np.min(relevance_per_idx)),
            "all_relevance_max": float(np.max(relevance_per_idx)),
            "all_relevance_mean": float(np.mean(relevance_per_idx)),
            "all_relevance_median": float(np.median(relevance_per_idx)),
            "all_relevance_std": float(np.std(relevance_per_idx)),
            "coreset_relevance": coreset_relevance,
            "coreset_redundancy": coreset_redundancy,
        }

        # Train regressor on selected features (always predicting mean score)
        X = source_full_scores[:, self.compressed_data_indices]
        self.rgs = self._build_regressor(X, regression_target)

        return self


# --- FCD (difference) variants ---


class MRMRPred_FCD_y(MRMRFCPred):
    """MRMR-FC with difference scheme and average score as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_FCD_PC1(MRMRFCPred):
    """MRMR-FC with difference scheme and PC1 as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_FCD_IRT1(MRMRFCPred):
    """MRMR-FC with difference scheme and 1-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


# --- FCQ (quotient) variants ---


class MRMRPred_FCQ_y(MRMRFCPred):
    """MRMR-FC with quotient scheme and average score as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_FCQ_PC1(MRMRFCPred):
    """MRMR-FC with quotient scheme and PC1 as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_FCQ_IRT1(MRMRFCPred):
    """MRMR-FC with quotient scheme and 1-dim IRT ability as relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


# --- FCD2 / FCQ2: binary features use regression F (continuous treatment of x); same otherwise ---


class MRMRFCPredV2(MRMRFCPred):
    """FCD2/FCQ2 base: always use linear-regression F-stat relevance."""

    _force_linear_fc_relevance_all = True


class MRMRPred_FCD2_y(MRMRFCPredV2):
    """MRMR-FC with difference scheme, linear F for binary x, average score relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_FCD2_PC1(MRMRFCPredV2):
    """MRMR-FC with difference scheme, linear F for binary x, PC1 relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_FCD2_IRT1(MRMRFCPredV2):
    """MRMR-FC with difference scheme, linear F for binary x, IRT relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = False
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


class MRMRPred_FCQ2_y(MRMRFCPredV2):
    """MRMR-FC with quotient scheme, linear F for binary x, average score relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_FCQ2_PC1(MRMRFCPredV2):
    """MRMR-FC with quotient scheme, linear F for binary x, PC1 relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_FCQ2_IRT1(MRMRFCPredV2):
    """MRMR-FC with quotient scheme, linear F for binary x, IRT relevance target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "irt"
        kwargs["miq_scheme"] = True
        kwargs["irt_dims"] = 1
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# GMI variants (Gao et al. local Gaussian MI instead of KSG)
# ---------------------------------------------------------------------------


class MRMRGMIPred(MRMRPred):
    """MRMR using local Gaussian MI (Gao et al., UAI 2015).

    Estimates MI via penalised local log-likelihood density estimation at
    each sample point (LGDE).  Falls back to Ross / discrete-discrete
    estimators for binary features.
    """

    def _get_mi_estimators(self, _binary: bool):
        if _binary:
            return super()._get_mi_estimators(_binary)
        return (
            lambda x, y, k=self.mi_k: self._mutual_information_lgde(x, y, k),
            lambda X, x2: self._mutual_information_lgde_batch(X, x2, k=self.mi_k),
        )


class MRMRPred_GMID_y(MRMRGMIPred):
    """MRMR-GMI with difference scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_GMID_PC1(MRMRGMIPred):
    """MRMR-GMI with difference scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_GMIQ_y(MRMRGMIPred):
    """MRMR-GMI with quotient scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_GMIQ_PC1(MRMRGMIPred):
    """MRMR-GMI with quotient scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_GMID_yPC1(MRMRGMIPred):
    """MRMR-GMI with difference scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_GMIQ_yPC1(MRMRGMIPred):
    """MRMR-GMI with quotient scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# MRMR with PCA-based MI (PMI) – Gao et al. (AISTATS 2015) LNC correction
# ---------------------------------------------------------------------------


class MRMRPMIPred(MRMRPred):
    """MRMR using PCA-corrected KSG MI (Gao et al., AISTATS 2015).

    Applies the Local Nonuniformity Correction (LNC) which adds a PCA-based
    volume ratio correction to the standard KSG estimate.  Falls back to
    Ross / discrete-discrete estimators for binary features.
    """

    def _get_mi_estimators(self, _binary: bool):
        if _binary:
            return super()._get_mi_estimators(_binary)
        return (
            lambda x, y, k=self.mi_k: self._mutual_information_lnc(x, y, k),
            lambda X, x2: self._mutual_information_lnc_batch(X, x2, k=self.mi_k),
        )


class MRMRPred_PMID_y(MRMRPMIPred):
    """MRMR-PMI with difference scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_PMID_PC1(MRMRPMIPred):
    """MRMR-PMI with difference scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_PMIQ_y(MRMRPMIPred):
    """MRMR-PMI with quotient scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_PMIQ_PC1(MRMRPMIPred):
    """MRMR-PMI with quotient scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_PMI_y(MRMRPMIPred):
    """LNC (PCA-corrected) MI with relevance-only greedy selection (continuous case)."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["only_relevance"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_PMID_yPC1(MRMRPMIPred):
    """MRMR-PMI with difference scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_PMIQ_yPC1(MRMRPMIPred):
    """MRMR-PMI with quotient scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# MRMR with Quick Gaussian MI (QGMI) – closed-form weighted MLE
# ---------------------------------------------------------------------------


class MRMRQGMIPred(MRMRPred):
    """MRMR using quick Gaussian MI (closed-form weighted MLE).

    Approximates LGDE by replacing the per-point L-BFGS-B optimisation with
    a closed-form weighted Gaussian fit (weighted mean + weighted covariance
    with ridge regularisation).  Fully vectorised — dramatically faster than
    MRMRGMIPred.  Falls back to Ross / discrete-discrete estimators for
    binary features.
    """

    def _get_mi_estimators(self, _binary: bool):
        if _binary:
            return super()._get_mi_estimators(_binary)
        return (
            lambda x, y, k=self.mi_k: self._mutual_information_qgde(x, y, k),
            lambda X, x2: self._mutual_information_qgde_batch(X, x2, k=self.mi_k),
        )


class MRMRPred_QGMID_y(MRMRQGMIPred):
    """MRMR-QGMI with difference scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_QGMID_PC1(MRMRQGMIPred):
    """MRMR-QGMI with difference scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_QGMIQ_y(MRMRQGMIPred):
    """MRMR-QGMI with quotient scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "average"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_QGMIQ_PC1(MRMRQGMIPred):
    """MRMR-QGMI with quotient scheme and first principal component as target."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_QGMID_yPC1(MRMRQGMIPred):
    """MRMR-QGMI with difference scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_QGMIQ_yPC1(MRMRQGMIPred):
    """MRMR-QGMI with quotient scheme and normalized sum of avg score + PC1."""

    def fit(self, *args, **kwargs):
        kwargs["target_scheme"] = "y+pc1"
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# k=5 and k=7 MI nearest-neighbour variants
# ---------------------------------------------------------------------------
# Every MRMR class above uses k=3 (default) in the Ross MI estimator.
# The following programmatically creates copies that use k=5 or k=7 instead.

_MRMR_BASE_CLASSES = [
    MRMRPred,
    MRMRPred_MID_y,
    MRMRPred_MID_PC1,
    MRMRPred_MIQ_y,
    MRMRPred_MIQ_PC1,
    MRMRPred_MI_y,
    MRMRPred_MID_yPC1,
    MRMRPred_MIQ_yPC1,
    MRMRPred_MID_y_aipw,
    MRMRPred_MID_PC1_aipw,
    MRMRPred_MIQ_y_aipw,
    MRMRPred_MIQ_PC1_aipw,
    MRMRPred_MID_IRT1,
    MRMRPred_MID_IRT5,
    MRMRPred_MIQ_IRT1,
    MRMRPred_MIQ_IRT5,
    MRMRPred_MID_IRT1_aipw,
    MRMRPred_MID_IRT5_aipw,
    MRMRPred_MIQ_IRT1_aipw,
    MRMRPred_MIQ_IRT5_aipw,
    MRMRGMIPred,
    MRMRPred_GMID_y,
    MRMRPred_GMID_PC1,
    MRMRPred_GMIQ_y,
    MRMRPred_GMIQ_PC1,
    MRMRPred_GMID_yPC1,
    MRMRPred_GMIQ_yPC1,
    MRMRPMIPred,
    MRMRPred_PMID_y,
    MRMRPred_PMID_PC1,
    MRMRPred_PMIQ_y,
    MRMRPred_PMIQ_PC1,
    MRMRPred_PMI_y,
    MRMRPred_PMID_yPC1,
    MRMRPred_PMIQ_yPC1,
    MRMRQGMIPred,
    MRMRPred_QGMID_y,
    MRMRPred_QGMID_PC1,
    MRMRPred_QGMIQ_y,
    MRMRPred_QGMIQ_PC1,
    MRMRPred_QGMID_yPC1,
    MRMRPred_QGMIQ_yPC1,
]


def _make_mi_k_variant(base_cls, k):
    """Create a subclass that overrides mi_k for the Ross MI estimator."""

    class _Variant(base_cls):
        def __init__(self):
            super().__init__()
            self.mi_k = k

    new_name = base_cls.__name__.replace("MRMR", f"MRMR{k}", 1)
    _Variant.__name__ = new_name
    _Variant.__qualname__ = new_name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using k={k} MI neighbours)"
    )
    return _Variant


for _base_cls in _MRMR_BASE_CLASSES:
    for _k in (4, 5, 6, 7, 8, 9):
        _variant = _make_mi_k_variant(_base_cls, _k)
        globals()[_variant.__name__] = _variant


# ---------------------------------------------------------------------------
# GLM variants (Binomial GLM with logit link instead of Ridge regression)
# ---------------------------------------------------------------------------
# For every MRMR class (including k-variants and FC variants), create a
# version that uses _BinomialGLMRegressor instead of Ridge.  These are
# registered under "gmrmr_*" keys in __init__.py.


def _make_glm_variant(base_cls):
    """Create a subclass that uses Binomial GLM instead of Ridge."""

    class _GLMVariant(base_cls):
        def _build_regressor(self, X, y):
            rgs = _BinomialGLMRegressor(alpha=10)
            rgs.fit(X, y)
            return rgs

    new_name = base_cls.__name__.replace("MRMR", "GMRMR", 1)
    _GLMVariant.__name__ = new_name
    _GLMVariant.__qualname__ = new_name
    _GLMVariant.__module__ = __name__
    _GLMVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using Binomial GLM instead of Ridge regression)"
    )
    return _GLMVariant


def _make_raw_variant(base_cls):
    """Create a subclass that predicts via raw coreset average (no regression)."""

    class _RawVariant(base_cls):
        def _build_regressor(self, X, y):
            rgs = _RawAverageRegressor()
            rgs.fit(X, y)
            return rgs

    new_name = base_cls.__name__.replace("MRMR", "RawMRMR", 1)
    _RawVariant.__name__ = new_name
    _RawVariant.__qualname__ = new_name
    _RawVariant.__module__ = __name__
    _RawVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using raw coreset average instead of regression)"
    )
    return _RawVariant


def _make_krr_variant(base_cls, degree=2):
    """Create a subclass that uses Kernel Ridge Regression instead of Ridge."""

    _degree = degree  # capture for closure

    class _KRRVariant(base_cls):
        def _build_regressor(self, X, y):
            alphas = (
                np.logspace(-1, 1, 5) if self._binary
                else np.logspace(-2, 2, 9)
            )
            rgs = GridSearchCV(
                KernelRidge(kernel="poly", degree=_degree, coef0=1),
                param_grid={"alpha": alphas},
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
            )
            rgs.fit(X, y.ravel())
            self.alpha_ = float(rgs.best_params_["alpha"])
            self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
            return rgs

    suffix = "" if _degree == 2 else str(_degree)
    new_name = base_cls.__name__.replace("MRMR", f"K{suffix}MRMR", 1)
    _KRRVariant.__name__ = new_name
    _KRRVariant.__qualname__ = new_name
    _KRRVariant.__module__ = __name__
    _KRRVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using Kernel Ridge Regression with degree-{_degree} polynomial kernel)"
    )
    return _KRRVariant


def _make_cv_variant(base_cls):
    """Create a subclass that uses RidgeCV (5-fold cross-validated alpha) instead of Ridge."""

    class _CVVariant(base_cls):
        def _build_regressor(self, X, y):
            rgs = RidgeCV(
                alphas=np.logspace(0, 2, 9), # default is (0.1, 1, 10.0)
                cv=5, # 5-fold cross-validation
                # cv=None, # None means leave-one-out cross-validation
            )
            rgs.fit(X, y.reshape(-1, 1))
            self.alpha_ = float(rgs.alpha_)
            alphas = np.logspace(0, 2, 9)
            self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
            return rgs

    new_name = base_cls.__name__.replace("MRMR", "CVMRMR", 1) 
    _CVVariant.__name__ = new_name
    _CVVariant.__qualname__ = new_name
    _CVVariant.__module__ = __name__
    _CVVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using RidgeCV with 5-fold cross-validation)"
    )
    return _CVVariant


def _make_rf_variant(base_cls):
    """Create a subclass that uses Random Forest Regression instead of Ridge."""

    class _RFVariant(base_cls):
        def _build_regressor(self, X, y):
            rgs = RandomForestRegressor(
                n_estimators=100, max_depth=None, random_state=42,
            )
            rgs.fit(X, y.ravel())
            return rgs

    new_name = base_cls.__name__.replace("MRMR", "RFMRMR", 1)
    _RFVariant.__name__ = new_name
    _RFVariant.__qualname__ = new_name
    _RFVariant.__module__ = __name__
    _RFVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using Random Forest Regression)"
    )
    return _RFVariant


def _make_syn_variant(base_cls):
    """Create a subclass with 10% random coreset points and synthetic boundary datapoints.

    The variant modifies base MRMR behaviour in two ways:

    1. **10 % random coreset points** – during ``fit``, only 90 % of the
       requested coreset is chosen via MRMR; the remaining 10 % are picked
       uniformly at random from the remaining question pool.
    2. **Synthetic boundary datapoints** – just before the regressor is
       fitted (in ``_build_regressor``), two synthetic rows are injected:
       a *perfect* model (all-ones features, target = 1.0) and a *useless*
       model (all-zeros features, target = 0.0).  These anchor the
       regression at the boundary extremes without affecting any
       downstream outputs (coreset indices, predictions, etc.).
    """

    class _SynVariant(base_cls):
        def _build_regressor(self, X, y):
            """Add synthetic perfect/useless datapoints then delegate to parent."""
            n_features = X.shape[1]
            perfect = np.ones((1, n_features))
            useless = np.zeros((1, n_features))
            X_aug = np.vstack([X, perfect, useless])
            y_aug = np.concatenate([y.ravel(), [1.0, 0.0]])
            return super()._build_regressor(X_aug, y_aug)

        def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
            """MRMR selection with 10 % of coreset points chosen at random."""
            random_size = max(1, int(0.1 * coreset_size))
            mrmr_size = coreset_size - random_size
            if mrmr_size < 2:
                # Coreset too small for random replacement; use full MRMR
                mrmr_size = coreset_size
                random_size = 0

            # Run parent MRMR selection with reduced coreset size
            super().fit(
                source_full_scores, mrmr_size, seed=seed, *args, **kwargs
            )

            if random_size > 0:
                # Pick random points from the pool not already selected
                num_data = source_full_scores.shape[1]
                mrmr_indices = set(self.compressed_data_indices.tolist())
                remaining = [
                    i for i in range(num_data) if i not in mrmr_indices
                ]
                random_indices = np.random.choice(
                    remaining, size=random_size, replace=False
                )
                self.compressed_data_indices = np.concatenate(
                    [self.compressed_data_indices, random_indices]
                )

                # Retrain regressor on the full coreset (synthetic boundary
                # points are added inside _build_regressor)
                regression_target = source_full_scores.mean(-1)
                X = source_full_scores[:, self.compressed_data_indices]
                self.rgs = self._build_regressor(X, regression_target)

            return self

    new_name = base_cls.__name__.replace("MRMR", "SynMRMR", 1)
    _SynVariant.__name__ = new_name
    _SynVariant.__qualname__ = new_name
    _SynVariant.__module__ = __name__
    _SynVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(with 10% random coreset points and synthetic boundary datapoints)"
    )
    return _SynVariant


def _make_logit_variant(base_cls):
    """Create a subclass that fits the regressor in logit-scaled target space.

    Targets are scaled to the unit interval with a small buffer (so that the
    logit is finite and the model can extrapolate slightly beyond the training
    range), then logit-transformed before fitting.  Predictions are
    inverse-transformed (sigmoid + unscale) back to the original target space.

    Composes with any parent regressor type (Ridge, KRR, RidgeCV, etc.)
    because the transform is applied around the parent's _build_regressor.
    """

    class _LogitVariant(base_cls):
        def _build_regressor(self, X, y):
            scaler = _LogitTargetScaler(buffer=0.05)
            scaler.fit(y.ravel())
            y_logit = scaler.transform(y.ravel())
            inner_rgs = super()._build_regressor(X, y_logit)
            return _LogitWrappedRegressor(inner_rgs, scaler)

    new_name = "L" + base_cls.__name__
    _LogitVariant.__name__ = new_name
    _LogitVariant.__qualname__ = new_name
    _LogitVariant.__module__ = __name__
    _LogitVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(with logit-scaled target regression)"
    )
    return _LogitVariant


# ---------------------------------------------------------------------------
# PIRT / GPIRT variants with MRMR coreset selection
# ---------------------------------------------------------------------------
# These replace the KMeans anchor-point selection in PIRTPred / GPIRTPred
# with MRMR feature selection on the raw question-level scores.  Everything
# else (IRT training, ability estimation, prediction) stays identical.


class PIRTMRMRPred(PIRTPred):
    """PIRT with MRMR coreset selection instead of KMeans on IRT parameters.

    Trains the same IRT model as PIRTPred but replaces the KMeans anchor
    point selection with MRMR feature selection on the raw question scores.
    Anchor weights are set to uniform (1/K) since MRMR does not produce
    cluster assignments.
    """

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = False

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        super().fit(source_full_scores, coreset_size, seed=seed, *args, **kwargs)

        mrmr = MRMRPred()
        mrmr.fit(
            source_full_scores,
            coreset_size,
            seed=seed,
            target_scheme=self._mrmr_target_scheme,
            miq_scheme=self._mrmr_miq_scheme,
        )
        self.anchor_points = mrmr.get_coreset()
        self.anchor_weights = np.ones(coreset_size) / coreset_size
        return self


class PIRTMRMRPred_MID_y(PIRTMRMRPred):
    """PIRT-MRMR with MID scheme and average score target."""

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = False


class PIRTMRMRPred_MID_PC1(PIRTMRMRPred):
    """PIRT-MRMR with MID scheme and PC1 target."""

    _mrmr_target_scheme = "pc1"
    _mrmr_miq_scheme = False


class PIRTMRMRPred_MIQ_y(PIRTMRMRPred):
    """PIRT-MRMR with MIQ scheme and average score target."""

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = True


class PIRTMRMRPred_MIQ_PC1(PIRTMRMRPred):
    """PIRT-MRMR with MIQ scheme and PC1 target."""

    _mrmr_target_scheme = "pc1"
    _mrmr_miq_scheme = True


class GPIRTMRMRPred(GPIRTPred):
    """GPIRT with MRMR coreset selection instead of KMeans on IRT parameters.

    Trains the same IRT model(s) as GPIRTPred and computes lambda identically,
    but replaces the KMeans anchor point selection with MRMR feature selection
    on the raw question scores.  Anchor weights are set to uniform (1/K).
    """

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = False

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        super().fit(source_full_scores, coreset_size, seed=seed, *args, **kwargs)

        mrmr = MRMRPred()
        mrmr.fit(
            source_full_scores,
            coreset_size,
            seed=seed,
            target_scheme=self._mrmr_target_scheme,
            miq_scheme=self._mrmr_miq_scheme,
        )
        self.anchor_points = mrmr.get_coreset()
        self.anchor_weights = np.ones(coreset_size) / coreset_size
        return self


class GPIRTMRMRPred_MID_y(GPIRTMRMRPred):
    """GPIRT-MRMR with MID scheme and average score target."""

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = False


class GPIRTMRMRPred_MID_PC1(GPIRTMRMRPred):
    """GPIRT-MRMR with MID scheme and PC1 target."""

    _mrmr_target_scheme = "pc1"
    _mrmr_miq_scheme = False


class GPIRTMRMRPred_MIQ_y(GPIRTMRMRPred):
    """GPIRT-MRMR with MIQ scheme and average score target."""

    _mrmr_target_scheme = "average"
    _mrmr_miq_scheme = True


class GPIRTMRMRPred_MIQ_PC1(GPIRTMRMRPred):
    """GPIRT-MRMR with MIQ scheme and PC1 target."""

    _mrmr_target_scheme = "pc1"
    _mrmr_miq_scheme = True


# ---------------------------------------------------------------------------
# IRT-representation MRMR variants
# ---------------------------------------------------------------------------
# Instead of computing MI between raw M-long score vectors during MRMR
# coreset selection, first fit an IRT model and use the model's predicted
# response probabilities P(theta_m, A_j, b_j) as "smoothed" question
# representations.  The relevance target y (mean raw score per model) and
# the final Ridge regression remain unchanged.


def _make_irt_mrmr_variant(
    base_cls,
    irt_model_type="multidim_2pl",
    irt_dims=10,
    continuous=False,
    gaussian=False,
    name_prefix="IRT",
):
    """Create a MRMR variant that uses IRT-predicted scores for MI computation.

    Wraps *base_cls* (any MRMR strategy class) so that ``fit()`` first trains
    an IRT model, computes predicted response probabilities for every
    (model, question) pair, and feeds those predictions — instead of raw
    scores — into the parent MRMR algorithm for relevance / redundancy MI.

    The relevance target and the downstream regressor are always trained on
    the **raw** scores, not the IRT predictions.
    """
    _irt_model_type = irt_model_type
    _irt_dims = irt_dims
    _irt_continuous = continuous
    _irt_gaussian = gaussian

    class _IRTMRMRVariant(base_cls):

        def _compute_irt_predicted_scores(self, source_full_scores, seed):
            M, N = source_full_scores.shape

            if _irt_continuous:
                _score_min = float(source_full_scores.min())
                _score_max = float(source_full_scores.max())
                _EPS = 1e-4
                span = _score_max - _score_min + 1e-12
                normalized = _EPS + (1 - 2 * _EPS) * (
                    source_full_scores - _score_min
                ) / span
                dataset = create_continuous_irt_dataset(normalized)
            else:
                dataset = create_irt_dataset(source_full_scores)

            config = IrtConfig(
                priors="hierarchical",
                dims=_irt_dims,
                lr=0.1,
                epochs=2000,
                model_type=_irt_model_type,
                dropout=0.5,
                hidden=100,
                log_every=200,
                deterministic=True,
                seed=seed,
            )
            trainer = IrtModelTrainer(
                config=config, dataset=dataset, verbose=True,
            )
            trainer.train(device="cpu")
            params = trainer.best_params

            if _irt_gaussian:
                A = np.ones((1, 1, N))
            else:
                A = np.array(params["disc"]).T[None, :, :]
            B = np.array(params["diff"]).T[None, :, :]
            ability = np.array(params["ability"])

            theta_all = ability[:, :, np.newaxis]          # (M, D, 1)
            predicted = item_curve(theta_all, A, B)        # (M, N)

            if _irt_continuous:
                predicted = (
                    (predicted - _EPS) / (1 - 2 * _EPS)
                    * (_score_max - _score_min)
                    + _score_min
                )

            return predicted

        def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
            self._raw_source_full_scores = source_full_scores

            irt_predicted = self._compute_irt_predicted_scores(
                source_full_scores, seed,
            )

            result = super().fit(
                irt_predicted, coreset_size, seed=seed, *args, **kwargs,
            )

            regression_target = source_full_scores.mean(-1)
            X = source_full_scores[:, self.compressed_data_indices]
            self.rgs = self._build_regressor(X, regression_target)

            return result

        def _select_target(self, source_full_scores, seed=42, irt_dims=5):
            raw = getattr(self, "_raw_source_full_scores", source_full_scores)
            return super()._select_target(raw, seed=seed, irt_dims=irt_dims)

    base_name = base_cls.__name__
    pred_pos = base_name.find("Pred_")
    if pred_pos >= 0:
        insert_pos = pred_pos + len("Pred_")
        new_name = base_name[:insert_pos] + name_prefix + base_name[insert_pos:]
    else:
        new_name = base_name + "_" + name_prefix

    _IRTMRMRVariant.__name__ = new_name
    _IRTMRMRVariant.__qualname__ = new_name
    _IRTMRMRVariant.__module__ = __name__
    _IRTMRMRVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using {irt_model_type} IRT with {irt_dims} dims for question representations)"
    )
    return _IRTMRMRVariant


# --- Generate IRT-representation MRMR classes ---

_IRT_CONFIGS = [
    # (name_prefix_root, model_type, dims, continuous, gaussian)
    ("IRT",     "multidim_2pl",  10, False, False),
    ("BetaIRT", "beta_2pl_nd",   10, True,  False),
    ("B3IRT",   "beta_cubed",     1, True,  False),
    ("LEGOIRT", "lego_cm",       10, True,  False),
    ("GIRT",    "gaussian_irt",   1, True,  True),
]

_IRT_STRATEGY_CLASSES = [
    MRMRPred_MID_y,
    MRMRPred_MIQ_y,
    MRMRPred_FCD_y,
    MRMRPred_FCQ_y,
    MRMRPred_PMID_y,
    MRMRPred_PMIQ_y,
    MRMRPred_GMID_y,
    MRMRPred_GMIQ_y,
    MRMRPred_QGMID_y,
    MRMRPred_QGMIQ_y,
]

_IRT_MRMR_BASE_CLASSES = []

for _irt_prefix, _irt_model_type, _irt_dims, _irt_cont, _irt_gauss in _IRT_CONFIGS:
    for _strategy_cls in _IRT_STRATEGY_CLASSES:
        _full_prefix = f"{_irt_prefix}{_irt_dims}"
        _variant = _make_irt_mrmr_variant(
            _strategy_cls,
            irt_model_type=_irt_model_type,
            irt_dims=_irt_dims,
            continuous=_irt_cont,
            gaussian=_irt_gauss,
            name_prefix=_full_prefix,
        )
        globals()[_variant.__name__] = _variant
        _IRT_MRMR_BASE_CLASSES.append(_variant)

# Generate k-variants (k=4,5,6,7,8,9) for IRT-representation MRMR classes
for _base_cls in _IRT_MRMR_BASE_CLASSES:
    for _k in (4, 5, 6, 7, 8, 9):
        _variant = _make_mi_k_variant(_base_cls, _k)
        globals()[_variant.__name__] = _variant
