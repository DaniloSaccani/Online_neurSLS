import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# -------------------------------
# Helper: Discrete-time LQR solver
# -------------------------------
def dlqr(A, B, Q, R):
    """
    Solve the discrete time LQR controller for a system:
         x_{t+1} = A x_t + B u_t
    cost = sum x'Qx + u'Ru
    Returns gain K.
    """
    A = np.array(A)
    B = np.array(B)
    Q = np.array(Q)
    R = np.array(R)
    # solve Riccati equation iteratively
    max_iter = 1000
    eps = 1e-8
    P = Q
    for i in range(max_iter):
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_next - P)) < eps:
            P = P_next
            break
        P = P_next
    # Compute the LQR gain
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return torch.tensor(K, dtype=torch.float32)


# -------------------------------
# Class 1: OptimalTrajectory
# -------------------------------
class OptimalTrajectory(nn.Module):
    def __init__(self, x0, x_ref, Ts, b1, b2, N, device='cpu'):
        """
        x0: initial state, shape (4,)
        x_ref: desired state trajectory, tensor shape (N+1, 4)
        Ts: sampling time
        b1, b2: system parameters
        N: horizon length
        """
        super(OptimalTrajectory, self).__init__()
        self.x0 = x0  # tensor shape (4,)
        self.x_ref = x_ref  # tensor shape (N+1, 4)
        self.Ts = Ts
        self.b1 = b1
        self.b2 = b2
        self.N = N
        self.device = device

        # Initialize the control sequence as parameters: shape (N, 2)
        # We use a small random initial guess.
        self.u_seq = nn.Parameter(torch.zeros(N, 2, device=self.device))

    def dynamics(self, x, u):
        """
        Noise-free dynamics of the 2D double integrator with nonlinear damping.
        x: state tensor of shape (..., 4) where x = [p_x, p_y, v_x, v_y]
        u: control tensor of shape (..., 2)
        Returns next state.
        """
        Ts = self.Ts
        b1 = self.b1
        b2 = self.b2
        # Split state into position and velocity:
        p = x[..., :2]
        v = x[..., 2:]
        p_next = p + Ts * v
        # Note: tanh is applied elementwise.
        v_next = v + Ts * (u - b1 * v + b2 * torch.tanh(v))
        x_next = torch.cat((p_next, v_next), dim=-1)
        return x_next

    def forward(self):
        """
        Simulate the noise-free trajectory over the horizon using current u_seq.
        Returns the trajectory as a tensor of shape (N+1, 4).
        """
        traj = [self.x0]
        x = self.x0
        for k in range(self.N):
            u = self.u_seq[k]
            x = self.dynamics(x, u)
            traj.append(x)
        traj = torch.stack(traj, dim=0)  # shape (N+1, 4)
        return traj

    def cost(self):
        """
        Computes the cost as the sum over stages of:
           cost = (position error)^T Qp (position error) + u^T R u.
        A terminal cost on the final position error is also added.
        """
        Qp = 10.0 * torch.eye(2, device=self.device)
        R_u = 0.1 * torch.eye(2, device=self.device)

        traj = self.forward()  # shape (N+1, 4)
        cost_val = 0.0
        # Sum over horizon (stage cost)
        for k in range(self.N):
            pos_error = traj[k, :2] - self.x_ref[k, :2]
            u = self.u_seq[k]
            cost_val = cost_val + pos_error @ Qp @ pos_error + u @ R_u @ u
        # Terminal cost
        pos_error = traj[self.N, :2] - self.x_ref[self.N, :2]
        cost_val = cost_val + pos_error @ Qp @ pos_error
        return cost_val

    def optimize(self, num_iters=500, lr=0.01, verbose=True):
        """
        Optimize the control sequence using gradient descent.
        Returns the optimized control sequence and the resulting trajectory.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for it in range(num_iters):
            optimizer.zero_grad()
            cost_val = self.cost()
            cost_val.backward()
            optimizer.step()
            if verbose and it % 50 == 0:
                print(f"Iteration {it}, cost = {cost_val.item():.4f}")
        # Return the optimized trajectory and control sequence
        traj_opt = self.forward().detach()
        u_opt = self.u_seq.detach()
        return u_opt, traj_opt


# -------------------------------
# Class 2: ReferenceTrajectory
# -------------------------------
class ReferenceTrajectory:
    def __init__(self, x_traj, Ts):
        """
        x_traj: numpy array or tensor of shape (N+1, 4) representing the optimal trajectory.
        Ts: sampling time.
        """
        if isinstance(x_traj, torch.Tensor):
            self.x_traj = x_traj.cpu().numpy()
        else:
            self.x_traj = x_traj
        self.Ts = Ts
        self.N = self.x_traj.shape[0] - 1
        self.lengthtraj = np.shape(self.x_traj)[0]-1

    def get_reference(self, t):
        """
        Given time t (in seconds), returns the corresponding reference state.
        For simplicity, we use zero–order hold (i.e. nearest index).
        """
        # idx = int(round(t / self.Ts))
        # idx = min(max(idx, 0), self.N)
        if t>self.lengthtraj:
            return self.x_traj[0]
        else:
            return self.x_traj[t]


# -------------------------------
# Class 3: ClosedLoopSystem
# -------------------------------
class TrackingRobot(nn.Module):
    def __init__(self, A, B, K, Ts, b2, device='cpu'):
        """
        A, B: system matrices (as torch tensors) for the linear part.
        K: LQR gain (as torch tensor) computed from the linearized error dynamics.
        Ts: sampling time.
        b2: nonlinear damping gain.
        The closed-loop dynamics (for the error state) are assumed to be:
            x_{t+1} = (A - B K)x + f(x) + B u2 + w,
        where
            f(x) = [0, 0, -Ts * b2 * tanh(v1), -Ts * b2 * tanh(v2)]^T,
        with x = [p_x, p_y, v_x, v_y].
        """
        super(TrackingRobot, self).__init__()
        self.A = A.to(device)
        self.B = B.to(device)
        self.K = K.to(device)
        self.Ts = Ts
        self.b2 = b2
        self.device = device
        self.n_agents = 1
        self.n = 4
        self.m = 2


    def f_nonlinear(self, x):
        """
        Compute the nonlinear term f(x) for the error dynamics.
        x: tensor of shape (..., 4)
        Returns: tensor of shape (..., 4) with first two components zero
                 and last two equal to -Ts*b2*tanh(velocity error).
        """
        Ts = self.Ts
        b2 = self.b2
        zeros = torch.zeros_like(x[..., :2])
        v = x[..., 2:]
        nonlinear = +Ts * b2 * torch.tanh(v)
        return torch.cat([zeros, nonlinear], dim=-1)

    def forward(self, t, x, u2, w):
        """
        Simulate one time step of the closed-loop error dynamics.
        t: current time (unused here, but kept for interface consistency)
        x: current state (error), tensor shape (batch, 4) or (4,) if batchless.
        u2: additional input, tensor shape (batch, 2) or (2,)
        w: process noise, tensor of shape matching x.
        Returns: next state.
        Computation:
            x_next = (A - B*K)x + f_nonlinear(x) + B*u2 + w.
        """
        # Ensure x, u2, w are 2D (batch size 1) if necessary.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u2.dim() == 1:
            u2 = u2.unsqueeze(0)
        if w.dim() == 1:
            w = w.unsqueeze(0)

        Acl = self.A - self.B @ self.K
        x_next = (Acl @ x.T).T + self.f_nonlinear(x) + (self.B @ u2.T).T + w
        # Remove batch dimension if it was added
        if x_next.shape[0] == 1:
            x_next = x_next.squeeze(0)
        return x_next
