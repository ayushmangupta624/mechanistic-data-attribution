"""EK-FAC covariance accumulation, eigendecomposition, and inverse-Hessian-vector products
for the Q/K, V, and O weight blocks of a single attention head."""

import torch


class EKFACQKVOHead:
    def __init__(self, d_model, d_head, damping=1e-5, damping_alpha=0.1):
        self.d_model = d_model
        self.d_head  = d_head
        self.damping = damping
        self.damping_alpha = damping_alpha

        # Three blocks: QK, V, O
        self.blocks = [
            {'name': 'W_QK', 'd_in': d_model, 'd_out': 2 * d_head},
            {'name': 'W_V',  'd_in': d_model, 'd_out': d_head},
            {'name': 'W_O',  'd_in': d_head,  'd_out': d_model},
        ]
        self.A_accum     = [None, None, None]
        self.S_accum     = [None, None, None]
        self.token_count = 0
        self.Q_A         = [None, None, None]
        self.Q_S         = [None, None, None]
        self.Lambda      = [None, None, None]

    def accumulate_AS(self, X_flat, dQ, dK, dV, Z_flat, dR):
        G_qk = torch.cat([dQ, dK], dim=-1)

        A0 = X_flat.t() @ X_flat;  S0 = G_qk.t() @ G_qk
        A1 = X_flat.t() @ X_flat;  S1 = dV.t()   @ dV
        A2 = Z_flat.t() @ Z_flat;  S2 = dR.t()   @ dR

        for i, (A, S) in enumerate([(A0,S0),(A1,S1),(A2,S2)]):
            if self.A_accum[i] is None:
                self.A_accum[i] = A
                self.S_accum[i] = S
            else:
                self.A_accum[i].add_(A)
                self.S_accum[i].add_(S)

        self.token_count += X_flat.shape[0]

    def finalize_eigendecomposition(self):
        for i in range(3):
            A = self.A_accum[i] / self.token_count
            S = self.S_accum[i] / self.token_count

            # Symmetrize and regularize
            A = 0.5 * (A + A.t())
            S = 0.5 * (S + S.t())
            eps_A = (1e-6 * A.trace().abs() / A.shape[0])
            eps_S = (1e-6 * S.trace().abs() / S.shape[0])
            A = A + eps_A * torch.eye(A.shape[0], device=A.device)
            S = S + eps_S * torch.eye(S.shape[0], device=S.device)

            _, self.Q_A[i] = torch.linalg.eigh(A.float())
            _, self.Q_S[i] = torch.linalg.eigh(S.float())

    def fit_lambda(self, X_flat, dQ, dK, dV, Z_flat, dR, weight):
        G_qk = torch.cat([dQ, dK], dim=-1)

        dW0 = X_flat.t() @ G_qk
        dW1 = X_flat.t() @ dV
        dW2 = Z_flat.t() @ dR

        for i, dW in enumerate([dW0, dW1, dW2]):
            ge = self.Q_A[i].t() @ dW @ self.Q_S[i]
            if self.Lambda[i] is None:
                self.Lambda[i] = ge.pow(2) * weight
            else:
                self.Lambda[i].add_(ge.pow(2) * weight)

    def finalize_lambda(self, total_weight):
        for i in range(3):
            self.Lambda[i] = (self.Lambda[i] / total_weight).flatten()

    def inverse_hvp(self, block_idx, grad_matrix):
        G  = grad_matrix.float()
        QA = self.Q_A[block_idx]
        QS = self.Q_S[block_idx]
        ge = QA.t() @ G @ QS

        lam   = self.Lambda[block_idx]
        denom = lam + self.damping_alpha * lam.mean()
        denom = torch.clamp(denom, min=self.damping)

        d_in  = self.blocks[block_idx]['d_in']
        d_out = self.blocks[block_idx]['d_out']
        ihvp  = QA @ (ge.flatten() / denom).reshape(d_in, d_out) @ QS.t()
        return ihvp
