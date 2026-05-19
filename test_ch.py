import numpy as np

def simulate_overwrite():
    ch = {}
    for i in range(10):
        # simulate top 30 moves, slightly shifting
        moves = np.arange(i, i+30)
        priors = np.random.dirichlet(np.ones(30))
        for m, p in zip(moves, priors):
            ch[m] = p
    print(f"Total children: {len(ch)}")
    print(f"Sum of priors: {sum(ch.values()):.2f}")

simulate_overwrite()
