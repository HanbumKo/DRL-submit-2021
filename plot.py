import matplotlib.pyplot as plt
import numpy as np

nodes = np.array([[0.47594, 0.56726761],
 [0.90660143, 0.7078028 ],
 [0.62306077, 0.15929967],
 [0.89323792, 0.02490608],
 [0.79478301, 0.6823078 ],
 [0.65056869, 0.60720885],
 [0.22951095, 0.17253584],
 [0.92468209, 0.64155377],
 [0.92847249, 0.01506382],
 [0.46576298, 0.16363963],
 [0.26344217, 0.55366352],
 [0.44111551, 0.05882897],
 [0.61881439, 0.35191443],
 [0.79409749, 0.68200626],
 [0.24147234, 0.94695581],
 [0.57691943, 0.36171269],
 [0.47059612, 0.7927109 ],
 [0.97469525, 0.92288038],
 [0.13029327, 0.76836856],
 [0.602551,   0.87703013]])

upto = 0

xs = nodes[:upto, 0]
ys = nodes[:upto, 1]

plt.scatter(xs, ys, color='b')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.hlines(0.25, 0, 1, colors='r', ls='--')
plt.hlines(0.5, 0, 1, colors='r', ls='--')
plt.hlines(0.75, 0, 1, colors='r', ls='--')
plt.vlines(0.25, 0, 1, colors='r', ls='--')
plt.vlines(0.5, 0, 1, colors='r', ls='--')
plt.vlines(0.75, 0, 1, colors='r', ls='--')

# plt.title("[0, 1] Uniformly generated instance")

plt.savefig("results/instance_" + str(upto) + ".png")

print("done")