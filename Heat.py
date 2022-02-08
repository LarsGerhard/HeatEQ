import numpy as np
from matplotlib import cm
from matplotlib.pyplot import plot, xlabel, ylabel, figure, title, pause
from numpy import linspace
from numpy import zeros, eye, matmul

# Solve Heat Equation numerically using a time and space mesh


# Parameters for the problem
K = 0.5  # heat conduction K = 0.5

# Spatial Grid
L = 15.  # Length of bar
dx = 0.1  # Spacing of points on bar
nx = int(L / dx + 1)  # **** fill in

x = linspace(0, L, nx)  # **** fill in

# Time grid
stopTime = 20  # Time to run the simulation
dt = .01  # Size of time step
nt = int(20 / dt + 1)  # **** fill in

time = linspace(0, stopTime, nt)  # **** fill in

# Create empty array to contain T(t,x)
T = zeros((nt, nx))  # nt times, nx positions

# Set the initial condition at t=0
T[0, :] = 50.  # uniform initial state

# Boundary Condition at x[0] and x[-1]
Ta = 100.  # left side
Tb = 0.  # right side

# Create Tridiagonal Matrix for iterating forward in time
Tdiag = -2 * eye(nx) + 1 * eye(nx, nx, 1) + 1 * eye(nx, nx, -1)  # **** fill in

M = K * (dt / dx ** 2) * Tdiag + eye(
    nx)  # **** fill in by using Tdiag, I, heat equation and matrix addition. Recall matmul(A,B) for matrix
# multiplication

# loop forward in time and compute change in string
for i in range(len(time) - 1):  # *** fill in loop over it
    # Calculate the future temperature along the center of bar in matrix form. 
    # Note that boundary condition is reapplied at each time step
    T[i + 1,:] = matmul(M, T[i,:])  # *** fill in using matrices above
    # (What is index of temperature at current time? previous time?)

    # apply left side boundary condition
    T[i + 1, 0] = Ta  # *** fill in fixed temperature

    # apply right side boundary condition
    T[i + 1,- 1] = Tb  # *** fill in fixed temperature

# Make some plots -- Leave this as is

# Steady state?
figure(1)
plot(x, T[-1, :])
xlabel('Distance')
ylabel('Temperature')
title('Final State Solution')

# Animated graph for every 10th time step
figure(2)
for it in range(0, nt, 20):
    plot(x, T[it, :])
    xlabel('Distance')
    ylabel('Temperature')
    title('Transient Solution')
    pause(.00001)

# 3-D suface plot

fig = figure(3)
ax = fig.gca(projection='3d')



# make X, Y grids
X, Y = np.meshgrid(x, time)

# Plot the surface.
surf = ax.plot_surface(X, Y, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
xlabel('Distance (cm)')
ylabel('Time (s)')
