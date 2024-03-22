import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import spsolve

# Define the Dirac gamma matrices
gamma0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, -1]])

gamma1 = np.array([[0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, -1, 0, 0],
               [-1, 0, 0, 0]])

gamma2 = np.array([[0, 0, 0, -1j],
               [0, 0, 1j, 0],
               [0, 1j, 0, 0],
               [-1j, 0, 0, 0]])

gamma3 = np.array([[0, 0, 1, 0],
               [0, 0, 0, -1],
               [-1, 0, 0, 0],
               [0, 1, 0, 0]])

class Solver:
    def __init__(self, init, params, grid):
        self.m = params[0]
        self.T,self.X,_,_ = grid
        self.dx = 1/self.X
        self.dt = 1/self.T
        self.init = init
        self.setup()

    def loc(self,t,x,c):
        return c + 4*t + 4*self.T*x

    def setup(self):
        X = self.X
        dx = self.dx
        T = self.T
        dt = self.dt
        m = self.m
        loc = self.loc

        M = np.zeros([4*T*X, 4*T*X], dtype = np.complex64) # 4 components x T elements x X elements 
        b = np.zeros([4*T*X])

        # Time derivative
        for t in range(1,T-1):
            for x in range(X):
                M[loc(t,x,0) : loc(t,x,4), loc(t+1,x,0) : loc(t+1,x,4)] = 1j/2 * gamma0 / dt
                M[loc(t,x,0) : loc(t,x,4), loc(t-1,x,0) : loc(t-1,x,4)] = -1j/2 * gamma0 / dt

        # Space derivative
        for t in range(1,T):
            for x in range(1,X-1):
                M[loc(t,x,0) : loc(t,x,4), loc(t,x+1,0) : loc(t,x+1,4)] = -1j/2 * gamma1 / dx
                M[loc(t,x,0) : loc(t,x,4), loc(t,x-1,0) : loc(t,x-1,4)] = 1j/2 * gamma1 / dx

        # Mass term
        for t in range(1,T):
            for x in range(X):
                M[loc(t,x,0) : loc(t,x,4), loc(t,x,0) : loc(t,x,4)] = m*np.identity(4)

        # Make the system circular
        for t in range(1,T-1):
            M[loc(t,0,0) : loc(t,0,4), loc(t+1,0,0) : loc(t+1,0,4)] = 1j/2 * gamma0 / dt
            M[loc(t,0,0) : loc(t,0,4), loc(t-1,0,0) : loc(t-1,0,4)] = -1j/2 * gamma0 / dt
            M[loc(t,0,0) : loc(t,0,4), loc(t,1,0) : loc(t,1,4)] = -1j/2 * gamma1 / dx
            M[loc(t,0,0) : loc(t,0,4), loc(t,X-1,0) : loc(t,X-1,4)] = 1j/2 * gamma1 / dx
            M[loc(t,0,0) : loc(t,0,4), loc(t,0,0) : loc(t,0,4)] = m*np.identity(4)

            M[loc(t,X-1,0) : loc(t,X-1,4), loc(t+1,X-1,0) : loc(t+1,X-1,4)] = 1j/2 * gamma0 / dt
            M[loc(t,X-1,0) : loc(t,X-1,4), loc(t-1,X-1,0) : loc(t-1,X-1,4)] = -1j/2 * gamma0 / dt
            M[loc(t,X-1,0) : loc(t,X-1,4), loc(t,0,0) : loc(t,0,4)] = -1j/2 * gamma1 / dx
            M[loc(t,X-1,0) : loc(t,X-1,4), loc(t,X-2,0) : loc(t,X-2,4)] = 1j/2 * gamma1 / dx
            M[loc(t,X-1,0) : loc(t,X-1,4), loc(t,X-1,0) : loc(t,X-1,4)] = m*np.identity(4)

        # Set initial condition
        for x in range(X):
            M[loc(0,x,0):loc(0,x,4), loc(0,x,0):loc(0,x,4)] = np.identity(4)
            b[loc(0,x,0):loc(0,x,4)] = self.init(x)*np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])

        self.M = M
        self.b = b

    def solve(self):
        psi_matrix = spsolve(self.M, self.b)
        self.psi = np.zeros([self.T,self.X,4], dtype=np.complex64)
        for t in range(self.T):
            for x in range(self.X):
                for c in range(4):
                    self.psi[t,x,c] = psi_matrix[self.loc(t,x,c)]

    def animate(self):
        # Create data
        norm = np.sum(np.multiply(self.psi[0],np.conj(self.psi[0])), axis=1).real

        # Create a figure and axis
        fig, ax = plt.subplots()
        line, = ax.plot(norm)
        ax.set_ylim(0,0.5)
        frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        # Update function for animation
        def update(frame):
            # Update data for each frame
            norm = np.sum(np.multiply(self.psi[frame],np.conj(self.psi[frame])), axis=1).real
            line.set_ydata(norm)
            frame_text.set_text('t: {}'.format(frame))  # Update frame indicator text
            return line,frame_text

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(self.T), interval=100)