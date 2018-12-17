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

EPOCHS = 50

with open(f"./details/rbm___784x392___ep_50.json", "rb") as f:
    rbm_1 = json.loads(f.read())

with open(f"./details/frbm___784x392___ep_50.json", "rb") as f:
    rbm_2 = json.loads(f.read())


xs = list(range(1, EPOCHS + 1))
ys_ticks = list(np.arange(0.03, 0.11, step=0.005))

plt.axis([0, xs[-1] + 1, 0.03, 0.11])
plt.grid(True)

plt.yticks(ys_ticks)

if EPOCHS == 10:
    plt.xticks(xs)
else:
    plt.xticks([x for x in xs if x % 2])

plt.plot(xs, rbm_1.get("errors"), "r")
plt.plot(xs, rbm_2.get("errors"), "b")
# plt.plot(xs, rbm_3.get("errors"), "g")

# plt.savefig(f"./images/rbm_to_frbm_comp___hid_{HIDDEN}___ep_{EPOCHS}.png")
plt.show()
