import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
fig, ax = plt.subplots(figsize=(11, 7)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 10)
ax.add_patch(Rectangle((3.2, 0.5), 5.8, 8.8, fill=False, lw=2))
ax.text(6.1, 8.95, "Campus Facility Reservation and Monitoring System", ha="center", weight="bold", fontsize=10)
ucs = [(8.0, "Upload and validate dataset"), (6.8, "View reservation statistics"), (5.6, "Analyze cancellation patterns"),
       (4.4, "Predict reservation risk"), (3.2, "Review risk queue"), (2.0, "Record final decision"), (1.1, "Submit reservation")]
for y, t in ucs:
    ax.add_patch(Ellipse((6.1, y), 4.4, 0.8, fill=False)); ax.text(6.1, y, t, ha="center", va="center", fontsize=9)
def actor(x, y, name):
    ax.plot([x, x], [y, y - 0.9], "k"); ax.plot([x - .5, x + .5], [y - .3, y - .3], "k")
    ax.plot([x, x - .5], [y - .9, y - 1.6], "k"); ax.plot([x, x + .5], [y - .9, y - 1.6], "k")
    ax.add_patch(plt.Circle((x, y + .25), .25, fill=False)); ax.text(x, y - 2.0, name, ha="center")
actor(1.3, 5.6, "Administrator"); actor(10.8, 5.6, "User")
for y in [8.0, 6.8, 5.6, 4.4, 3.2, 2.0]: ax.plot([1.8, 3.9], [5.0, y], lw=.8)
for y in [6.8, 1.1]: ax.plot([10.3, 8.3], [5.0, y], lw=.8)
ax.plot([6.1, 6.1], [3.6, 3.8], alpha=0)
plt.title("Updated Use Case Diagram", fontsize=14)
plt.savefig("assets/use_case_diagram.png", dpi=150, bbox_inches="tight")
