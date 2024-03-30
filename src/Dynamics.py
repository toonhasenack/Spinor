import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .Spinor import *

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
        (self.tmax, self.T),(self.xmax, self.X),_,_ = grid
        self.dx = self.xmax/self.X
        self.dt = self.tmax/self.T
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

        b = np.zeros([4*T*X], dtype = np.complex64)

        row_indices = []
        column_indices = []
        values = []

        # Time derivative
        for t in range(1,T):
            for x in range(X):
                for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(t,x,row)) 
                        column_indices.append(loc(t-1,x,col)) 
                        values.append(-1j * gamma0[row,col] / dt)

        # Space derivative
        for t in range(1,T):
            for x in range(1,X):
                for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(t,x,row)) 
                        column_indices.append(loc(t,x-1,col)) 
                        values.append(1j * gamma1[row,col] / dx - 2/m*np.identity(4)[row,col]/dx**2)

            for x in range(2,X):
                for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(t,x,row)) 
                        column_indices.append(loc(t,x-2,col)) 
                        values.append(1/m*np.identity(4)[row,col]/dx**2)
                

        # Mass term
        for t in range(1,T):
            for x in range(X):
                for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(t,x,row)) 
                        column_indices.append(loc(t,x,col)) 
                        values.append(1j * gamma0[row,col] / dt - 1j * gamma1[row,col] / dx + m*np.identity(4)[row,col] + 1/m*np.identity(4)[row,col]/dx**2)
                
        # Make the system circular
        for t in range(1,T):
                for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(t,0,row)) 
                        column_indices.append(loc(t-1,0,col)) 
                        values.append(-1j * gamma0[row,col] / dt)
                        
                        row_indices.append(loc(t,0,row)) 
                        column_indices.append(loc(t,X-1,col)) 
                        values.append(1j * gamma1[row,col] / dx - 2/m*np.identity(4)[row,col]/dx**2)

                        row_indices.append(loc(t,0,row)) 
                        column_indices.append(loc(t,X-2,col)) 
                        values.append(1/m*np.identity(4)[row,col]/dx**2)

                        row_indices.append(loc(t,0,row)) 
                        column_indices.append(loc(t,0,col)) 
                        values.append(1j * gamma0[row,col] / dt - 1j * gamma1[row,col] / dx + m*np.identity(4)[row,col] + 1/m*np.identity(4)[row,col]/dx**2)


                        row_indices.append(loc(t,X-1,row)) 
                        column_indices.append(loc(t-1,X-1,col)) 
                        values.append(-1j * gamma0[row,col] / dt)
                        
                        row_indices.append(loc(t,X-1,row)) 
                        column_indices.append(loc(t,X-2,col)) 
                        values.append(1j * gamma1[row,col] / dx - 2/m*np.identity(4)[row,col]/dx**2)

                        row_indices.append(loc(t,X-1,row)) 
                        column_indices.append(loc(t,X-3,col)) 
                        values.append(1/m*np.identity(4)[row,col]/dx**2)

                        row_indices.append(loc(t,X-1,row)) 
                        column_indices.append(loc(t,X-1,col)) 
                        values.append(1j * gamma0[row,col] / dt - 1j * gamma1[row,col] / dx + m*np.identity(4)[row,col] + 1/m*np.identity(4)[row,col]/dx**2)


        # Set initial condition
        for x in range(X):
            for row in range(4):
                    for col in range(4):
                        row_indices.append(loc(0,x,row)) 
                        column_indices.append(loc(0,x,col)) 
                        values.append(np.identity(4)[row,col])

            b[loc(0,x,0):loc(0,x,4)] = self.init(x*dx)*np.array([1/np.sqrt(2), 1/np.sqrt(2),0,0])

        self.M = coo_matrix((values, (row_indices, column_indices)))
        self.b = b

    def solve(self):
        psi_matrix = spsolve(self.M.tocsc(), self.b)
        self.psi = np.zeros([self.T,self.X,4], dtype=np.complex64)
        for t in range(self.T):
            for x in range(self.X):
                for c in range(4):
                    self.psi[t,x,c] = psi_matrix[self.loc(t,x,c)]

    def animate(self):
        # Create data
        norm = np.sum(np.multiply(self.psi[0],np.conj(self.psi[0])), axis=1).real

        # Create a figure and axis
        fig = plt.figure(figsize=(9,6))
        ax = plt.axes()
        ax.plot(np.arange(len(norm))*self.dx, norm,c="k")
        ax.fill_between(np.arange(len(norm))*self.dx, norm, 0, color='k', alpha=.1)
        ax.set_xlim(0,self.X)
        ax.set_ylim(0,0.5)
        ax.set_xlabel(r"$x$", size = 18)
        ax.set_ylabel(r"$\|\psi\|$", size = 18)
        ax.grid(True, which="both")

        # Update function for animation
        def update(frame):
            # Update data for each frame
            ax.cla()  # Clear the current axes
            norm = np.sum(np.multiply(self.psi[frame],np.conj(self.psi[frame])), axis=1).real
            ax.plot(np.arange(len(norm))*self.dx, norm,c="k")
            ax.fill_between(np.arange(len(norm))*self.dx, norm, 0, color='k', alpha=.1)
            ax.set_xlim(0,self.xmax)
            ax.set_ylim(0,0.5)
            ax.set_xlabel(r"$x$", size = 18)
            ax.set_ylabel(r"$\|\psi\|$", size = 18)
            ax.grid(True, which="both")
            ax.text(0.02, 0.95, f'step={frame}', transform=ax.transAxes,size=18)

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(self.T), interval=int(1e4/self.T))

        return ani