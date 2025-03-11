import torch
import numpy as np
import control


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


class TwoRobots(torch.nn.Module):
    def __init__(self, xbar, linear=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.xbar = xbar
        self.n_agents = 2
        self.n = 8
        self.m = 4
        self.h = 0.05
        self.A = torch.zeros([self.n,self.n])
        self.B = torch.zeros([self.n, self.m])
        self.Mx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mx[0, 0] = 1
        self.Mx[1, 4] = 1
        self.My = torch.zeros(self.n_agents, self.n, device=device)
        self.My[0, 1] = 1
        self.My[1, 5] = 1

        self.Mvx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvx[0, 2] = 1
        self.Mvx[1, 6] = 1
        self.Mvy = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvy[0, 3] = 1
        self.Mvy[1, 7] = 1

        self.mv1 = torch.zeros(2, self.n, device=device)
        self.mv1[0, 2] = 1
        self.mv1[1, 3] = 1
        self.mv2 = torch.zeros(2, self.n, device=device)
        self.mv2[0, 6] = 1
        self.mv2[1, 7] = 1
        self.mp = torch.zeros(2, self.n, device=device)
        self.Mp = torch.cat((self.mv1, self.mp, self.mv2, self.mp), 0)

    def f(self, t, x, u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        m1, m2 = 1, 1
        kspringGround = 2
        cdampGround = 2


        k1, k2 = kspringGround, kspringGround
        c1, c2 = cdampGround, cdampGround

        px = torch.matmul(self.Mx, x)
        py = torch.matmul(self.My, x)
        vx = torch.matmul(self.Mvx, x)
        vy = torch.matmul(self.Mvy, x)



        B = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m1, 0, 0, 0],
            [0, 1 / m1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1 / m2, 0],
            [0, 0 ,0, 1 / m2],
        ])
        Act = torch.tensor([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [-k1/m1, 0, -c1/m1, 0, 0, 0, 0, 0],
            [0, -k1/m1, 0, -c1/m1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, -k2/m2, 0, -c2/m2, 0],
            [0, 0, 0, 0, 0, -k2/m2, 0, -c2/m2],
        ])
        self.B = self.h*B
        self.A = torch.eye(self.n)+self.h*Act
        xt = torch.matmul(self.Mx.float(), self.xbar.float())
        yt = torch.matmul(self.My, self.xbar.float())

        deltaxt = px - xt
        deltayt = py - yt

        projxt = torch.cos(torch.atan2(deltayt, deltaxt))
        projyt = torch.sin(torch.atan2(deltayt, deltaxt))
        projvxt = torch.cos(torch.atan2(vy, vx))
        projvyt = torch.sin(torch.atan2(vy, vx))

        Fc01 = c1 * torch.sqrt(vx[0] ** 2 + vy[0] ** 2)
        Fc02 = c2 * torch.sqrt(vx[1] ** 2 + vy[1] ** 2)

        Fk01 = k1 * torch.sqrt(deltaxt[0] ** 2 + deltayt[0] ** 2)
        Fk02 = k2 * torch.sqrt(deltaxt[1] ** 2 + deltayt[1] ** 2)


        Fground1x = -Fk01 * projxt[0] - Fc01 * projvxt[0]
        Fground1y = -Fk01 * projyt[0] - Fc01 * projvyt[0]

        Fground2x = -Fk02 * projxt[1] - Fc02 * projvxt[1]
        Fground2y = -Fk02 * projyt[1] - Fc02 * projvyt[1]


        A1x = torch.tensor([
            0,
            0,
            (Fground1x) / m1,
            (Fground1y) / m1,
            0,
            0,
            (Fground2x) / m2,
            (Fground2y) / m2,
        ])

        A2x = torch.matmul(self.Mp, x)

        Ax = A1x + A2x

        x1 = x + (Ax + torch.matmul(B, u)) * self.h
        return x1

    def f_withnoise(self, t, x, u, w, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        #matrix used to apply the noise only to the position and scaled with time
        x1 = self.f(t, x, u, w)+w
        return x1

    def forward(self, t, x, u, w):
        #if t == 0:
        #    x1 = w
        #else:
        x1 = self.f_withnoise(t, x, u, w)
        return x1


import torch

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