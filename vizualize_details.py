import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (11, 8),
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "lines.linewidth": 2,
    "lines.markersize": 9,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

EPOCHS = 51
HIDDEN = 64
H1 = 128
H2 = 64
H3 = 32

with open(f"./details/rbm___784x{H1}___ep_{EPOCHS}_2.json", "rb") as f:
    rbm_results1 = json.loads(f.read())

with open(f"./details/rbm___784x{H2}___ep_{EPOCHS}_2.json", "rb") as f:
    rbm_results2 = json.loads(f.read())

with open(f"./details/rbm___784x{H3}___ep_{EPOCHS}_2.json", "rb") as f:
    rbm_results3 = json.loads(f.read())

# with open(f"./details/frbm___hid_{HIDDEN}___ep_{EPOCHS}", "rb") as ff:
#     frbm_results = json.loads(ff.read())

rbm_errors1 = rbm_results1.get("errors")
rbm_errors2 = rbm_results2.get("errors")
rbm_errors3 = rbm_results3.get("errors")

xs = list(range(1, EPOCHS + 1))
ys_ticks = list(np.arange(0.03, 0.13, step=0.005))

plt.axis([0, xs[-1] + 1, 0.03, 0.11])
plt.grid(True)

plt.yticks(ys_ticks)

if EPOCHS == 10:
    plt.xticks(xs)
else:
    plt.xticks([x for x in xs if x % 2])

plt.plot(xs, rbm_errors1, "b")
plt.plot(xs, rbm_errors1, "xb")

plt.plot(xs, rbm_errors2, "r")
plt.plot(xs, rbm_errors2, "xr")

plt.plot(xs, rbm_errors3, "g")
plt.plot(xs, rbm_errors3, "xg")

plt.savefig(f"./images/rbm_to_rbm_comp___hid_{H1}_vs_{H2}_vs_{H3}___ep_{EPOCHS}.png")
