import numpy as np
from clifford import Cl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Spinor:
    def __init__(self, clifford, vector, axial):
        self.clifford = clifford
        self.vector = vector
        self.axial = axial
        self.dim = self.clifford[0]+self.clifford[1]

        D, D_blades = Cl(self.clifford[0], self.clifford[1], firstIdx=0, names='g')
        globals().update(D_blades)
        I = ""
        for i in range(self.dim):
            I += f"{i}"
        g_ax = eval(f"g{I}")
        D_blades["g_ax"] = g_ax
        globals().update(D_blades)

        self.gamma0 = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, -1]])

        self.gamma1 = np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0]])

        self.gamma2 = np.array([[0, 0, 0, -1j],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [-1j, 0, 0, 0]])

        self.gamma3 = np.array([[0, 0, 1, 0],
                            [0, 0, 0, -1],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.spinor = self.vector_to_spinor(self.vector) + self.axial_to_spinor(self.axial)

    def __mul__(self, other):
        product = self.spinor*other.spinor
        spinor = Spinor(self.clifford, self.spinor_to_vector(product))
        return spinor
    
    def tr(self, spinor):
        return spinor.value[0]
    
    def sgn(self, i):
        if i <= self.clifford[0] - 1:
            return 1
        else:
            return -1

    def vector_to_spinor(self, vector):
        spinor = np.sum([self.sgn(i)*vector[i]*eval(f"g{i}") for i in range(self.dim)])
        return spinor
    
    def axial_to_spinor(self, axial):
        spinor = np.sum([self.sgn(i)*axial[i]*g_ax*eval(f"g{i}") for i in range(self.dim)])
        return spinor

    def spinor_to_vector(self, spinor):
        vector = np.array([self.tr(spinor*eval(f"g{i}")) for i in range(self.dim)])
        return vector
    
    def to_vector(self):
        self.vector = self.spinor_to_vector(self.spinor)
        return self.vector
    
    def spinor_to_axial(self, spinor):
        axial = np.array([self.tr(spinor*g_ax*eval(f"g{i}")) for i in range(self.dim)])
        return axial
    
    def to_axial(self):
        self.axial = self.spinor_to_axial(self.spinor)
        return self.axial

    def spinor_to_column(self, spinor):
        vector = self.spinor_to_vector(spinor)
        spinor = np.sum([self.sgn(i)*vector[i]*eval(f"self.gamma{i}") for i in range(self.dim)], axis = 0)
        evals, evects = np.linalg.eigh(spinor)
        for i in range(len(evals)):
            print(evals[i])
            evects[:,i] /= np.sqrt(evals[i])
        return evects

    def to_column(self):
        self.evects = self.spinor_to_column(self.spinor)


    def column_to_spinor(self, column):
        matrix = np.outer(column, np.conj(column))
        vector = [np.trace(np.dot(g, matrix)) for g in [self.gamma0, self.gamma1, self.gamma2, self.gamma3]]
        print(vector)
        spinor = self.vector_to_spinor(vector)
        return spinor

    def spinor_project(self, spinor):
        return np.array([(1+g_ax)*spinor/2, (1-g_ax)*spinor/2])
    
    def project(self):
        self.projections = self.spinor_project(self.spinor)
        return self.projections
    
    def lorentz(self, rotation = [0,0,0], boost = [0,0,0]):
        # This function only makes sense in Cl(1,3)
        self.spinor = np.exp(-rotation[0]/2*g2*g3)*self.spinor*np.exp(rotation[0]/2*g2*g3)
        self.spinor = np.exp(-rotation[1]/2*g3*g1)*self.spinor*np.exp(rotation[1]/2*g3*g1)
        self.spinor = np.exp(-rotation[2]/2*g1*g2)*self.spinor*np.exp(rotation[2]/2*g1*g2)

        self.spinor = np.exp(-boost[0]/2*g0*g1)*self.spinor*np.exp(boost[0]/2*g0*g1)
        self.spinor = np.exp(-boost[1]/2*g0*g2)*self.spinor*np.exp(boost[1]/2*g0*g2)
        self.spinor = np.exp(-boost[2]/2*g0*g3)*self.spinor*np.exp(boost[2]/2*g0*g3)

        self.to_vector()
        self.to_axial()

    def helicity(self):
        # This function only makes sense in Cl(1,3)
        self.helicity = np.sum(self.vector[1:]*self.axial[1:])/(np.linalg.norm(self.vector[1:])*np.linalg.norm(self.axial[1:]))
        return self.helicity
    
    def gamma_product(self):
        return self.tr(self.spinor)

    def plot(self, which = [1,2,3]):
        soa = np.array([[0, 0, 0, self.vector[which[0]], self.vector[which[1]], self.vector[which[2]]],\
                        [0, 0, 0, self.axial[which[0]], self.axial[which[1]], self.axial[which[2]]]])

        X, Y, Z, U, V, W = zip(*soa)
        colors = ['r', 'b']
        fig = plt.figure(figsize = (9,6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(X)):
            ax.quiver(X[i], Y[i], Z[i], U[i], V[i], W[i], color=colors[i], linewidth=5, arrow_length_ratio=0.1)

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="k", alpha=0.5)

        ax.legend(["P", "S"])
        ax.set_xlabel(r"$X_1$")
        ax.set_ylabel(r"$X_2$")
        ax.set_zlabel(r"$X_3$")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        return fig
    
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plot App")
        self.geometry("800x600")

        # Create a Spinor instance with initial vectors
        self.spinor = Spinor(clifford=(1, 3), vector=np.array([1, 0, 0, 0]), axial=np.array([0, 0, 0, 1]))

        # Create input fields
        self.create_input_fields()

        # Plot button
        self.plot_button = ttk.Button(self, text="Plot", command=self.plot)
        self.plot_button.pack()

        # Plot area
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(expand=True, fill=tk.BOTH)

        # Output fields
        self.chirality_label = ttk.Label(self, text="Chirality: ")
        self.chirality_label.pack()
        self.gamma_product_label = ttk.Label(self, text="Gamma Product: ")
        self.gamma_product_label.pack()

        # Initial plot
        self.plot()

    def create_input_fields(self):
        # Momentum vector P
        self.momentum_var = [tk.DoubleVar(value=0) for _ in range(4)]
        self.create_input_row("Enter momentum vector P:", self.momentum_var)

        # Spin vector S
        self.spin_var = [tk.DoubleVar(value=0) for _ in range(4)]
        self.create_input_row("Enter spin vector S:", self.spin_var)

        # Lorentz rotation
        self.rotation_var = [tk.DoubleVar(value=0) for _ in range(3)]
        self.create_input_row("Enter Lorentz rotation:", self.rotation_var)

        # Lorentz boost
        self.boost_var = [tk.DoubleVar(value=0) for _ in range(3)]
        self.create_input_row("Enter Lorentz boost:", self.boost_var)

    def create_input_row(self, label_text, var_list):
        label = ttk.Label(self, text=label_text)
        label.pack()

        frame = ttk.Frame(self)
        frame.pack()

        for var in var_list:
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side=tk.LEFT)

    def plot(self):
        # Parse input vectors and parameters
        P = [self.momentum_var[i].get() for i in range(4)]
        S = [self.spin_var[i].get() for i in range(4)]
        rotation = [self.rotation_var[i].get() for i in range(3)]
        boost = [self.boost_var[i].get() for i in range(3)]

        # Update spinor with new vectors
        self.spinor = Spinor((1, 3), np.array(P), np.array(S))

        # Apply Lorentz transformation
        self.spinor.lorentz(rotation=rotation, boost=boost)

        # Plot
        fig = self.spinor.plot()

         # Update output fields
        chirality = self.spinor.helicity()
        self.chirality_label.config(text=f"Chirality: {chirality}")

        gamma_product = self.spinor.gamma_product()
        self.gamma_product_label.config(text=f"Gamma Product: {gamma_product}")

        # If canvas exists, delete it
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create new canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()