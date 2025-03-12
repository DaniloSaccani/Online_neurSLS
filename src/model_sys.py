import torch
import numpy as np
import control
import torch.nn as nn



class DynamicObstacle:
    def __init__(self, amplitude=1.0, frequency=0.3, init_pos = 3.0, phase=0, noise_level=0.05, device=torch.device('cpu'),initial_time = None):
        """
        Creates a dynamic obstacle that oscillates between (0,3) and (0,-3) along the y-axis with random speed variations.
        - amplitude: Maximum y deviation (3 → -3)
        - frequency: Base oscillation frequency (Hz)
        - phase: Initial phase of oscillation
        - noise_level: Magnitude of random velocity noise
        """
        self.device = device
        self.amplitude = amplitude
        self.frequency = frequency  # Controls base oscillation speed
        self.phase = phase
        self.noise_level = noise_level  # Random noise intensity
        if initial_time is None:
            self.time = 0  # Time counter
        else:
            self.time = initial_time
        self.init_pos = init_pos
        self.velocity = 2 * np.pi * frequency * amplitude  # Initial speed

    def update_position(self, dt=0.1, velocity_noise = None):
        """
        Updates the obstacle position following a periodic motion with noise.
        """
        self.time += dt
        # Add noise to the velocity (random acceleration)
        if velocity_noise is None:
            velocity_noise = np.random.uniform(-self.noise_level, self.noise_level)
        vel = self.velocity + velocity_noise  # Update velocity with random variation
        #self.velocity = max(self.velocity, 0)  # Ensure velocity stays non-negative

        # Compute y-position with updated velocity
        y_pos = self.amplitude * np.sin(vel * self.time + self.phase) + self.init_pos

        # Store position as a tensor
        self.pos = torch.tensor([0.0, y_pos], dtype=torch.float32, device=self.device)

        return self.pos

    def predict_future_positions(self, dt=0.1, Horizon=10):
        """
        Predicts future positions over a given horizon with random velocity variations.
        - dt: Time step for each prediction step
        - Horizon: Number of future steps to predict
        """
        Horizon = int(Horizon)
        future_positions = torch.zeros((1, 2, Horizon), dtype=torch.float32, device=self.device)
        temp_time = self.time  # Store the current time to avoid modifying the actual state

        for i in range(Horizon):
            velocity_noise = np.random.uniform(-self.noise_level, self.noise_level)
            vel = self.velocity + velocity_noise
            temp_time += dt  # Advance time

            # Compute future y-position
            y_pos = self.amplitude * np.sin(vel * temp_time + self.phase) + self.init_pos

            # Store the x=0 and y_pos in the tensor
            future_positions[0, :, i] = torch.tensor([0.0, y_pos], dtype=torch.float32, device=self.device)

        return future_positions


class RobotsSystem(nn.Module):
    def __init__(self, xbar, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.xbar = xbar
        self.n_agents = 2
        self.n = 8
        self.m = 4
        self.h = 0.05
        self.A = torch.zeros([self.n, self.n])
        self.B = torch.zeros([self.n, self.m])
        self.mass = 1
        self.k = 1
        self.b1 = 2
        self.b2 = 0.5

    def f(self, t, x, u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        A = torch.tensor([
            [1, 0, self.h, 0, 0, 0, 0, 0],
            [0, 1, 0, self.h, 0, 0, 0, 0],
            [-(self.h * self.k / self.mass), 0, 1 - (self.h * self.b1 / self.mass), 0, 0, 0, 0, 0],
            [0, -(self.h * self.k / self.mass), 0, 1 - (self.h * self.b1 / self.mass), 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, self.h, 0],
            [0, 0, 0, 0, 0, 1, 0, self.h],
            [0, 0, 0, 0, -(self.h * self.k / self.mass), 0, 1 - (self.h * self.b1 / self.mass), 0],
            [0, 0, 0, 0, 0, -(self.h * self.k / self.mass), 0, 1 - (self.h * self.b1 / self.mass)],
        ])
        B = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [self.h/self.mass, 0, 0, 0],
            [0, self.h/self.mass, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, self.h/self.mass, 0],
            [0, 0, 0, self.h/self.mass],
        ])


        maskb2 = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, (self.h*self.b2)/self.mass, 0, 0, 0, 0, 0],
            [0, 0, 0, (self.h*self.b2)/self.mass, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, (self.h*self.b2)/self.mass, 0],
            [0, 0, 0, 0, 0, 0, 0, (self.h*self.b2)/self.mass],
        ])

        maskk = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [(self.h*self.k /self.mass), 0, 0, 0, 0, 0, 0, 0],
            [0, (self.h*self.k /self.mass), 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, (self.h*self.k /self.mass), 0, 0, 0],
            [0, 0, 0, 0, 0, (self.h*self.k /self.mass), 0, 0],
        ])


        x1 = torch.matmul(A, x) + torch.matmul(maskk,self.xbar) + torch.matmul(B, u) + torch.tanh(torch.matmul(maskb2,x))
        return x1

    def forward(self, t, x, u, w):
        # if t == 0:
        #    x1 = w
        # else:
        x1 = self.f(t, x, u, w) + w
        return x1


def computePriccati(sys, Q,R):
    A = sys.A
    B = sys.B
    (P, L, K) = control.dare(A, B, Q, R)
    P = torch.tensor(P)
    return P.float()

def disturbance(t, w, dmax, decayd, tmax):
    # Ensure t and w are tensors if not scalars
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)


    # If t and w are scalars, perform scalar computation
    if t.dim() == 0:
        if t > tmax:
            d = 0
        else:
            dideal = dmax * w * torch.exp(-t / decayd)
            d = torch.zeros(dideal.size())
            d[0] = dideal[0]
            d[1] = dideal[1]
            d[2] = 0.1 * dideal[2]
            d[3] = 0.1 * dideal[3]
            d[4] = dideal[4]
            d[5] = dideal[5]
            d[6] = 0.1 * dideal[6]
            d[7] = 0.1 * dideal[7]

    else:
        # Only unsqueeze if t is a 1D tensor, to make it [201, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # Change t from [201] to [201, 1]

        # Compute disturbance for tensor inputs
        dideal = dmax * w * torch.exp(-t / decayd)
        d = torch.zeros(dideal.size())
        d[:,0] = dideal[:,0]
        d[:,1] = dideal[:,1]
        d[:,2] = 0.1 * dideal[:,2]
        d[:,3] = 0.1 * dideal[:,3]
        d[:,4] = dideal[:,4]
        d[:,5] = dideal[:,5]
        d[:,6] = 0.1 * dideal[:,6]
        d[:,7] = 0.1 * dideal[:,7]

        # Apply the tmax condition: set d to 0 where t > tmax
        d[t.squeeze() > tmax, :] = 0

    return d

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