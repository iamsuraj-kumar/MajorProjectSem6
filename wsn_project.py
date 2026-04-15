import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =========================
# CONFIGURATION
# =========================
NUM_NODES = 60
EDGE_PROB = 0.15
INITIAL_INFECTED = 3
INFECTION_PROB = 0.3
RECOVERY_PROB = 0.05
STEPS = 30

# =========================
# CREATE FIXED NETWORK
# =========================
G = nx.erdos_renyi_graph(NUM_NODES, EDGE_PROB)
pos = nx.spring_layout(G, seed=42)  # fixed layout

# Node states
states = {node: "S" for node in G.nodes()}

# Initial infection
infected_nodes = random.sample(list(G.nodes()), INITIAL_INFECTED)
for node in infected_nodes:
    states[node] = "I"

# =========================
# TRACKING
# =========================
S_hist, I_hist, R_hist = [], [], []

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(7,7))

def draw_graph(step):
    plt.clf()

    colors = []
    for node in G.nodes():
        if states[node] == "S":
            colors.append("skyblue")
        elif states[node] == "I":
            colors.append("red")
        else:
            colors.append("green")

    nx.draw(G, pos,
            node_color=colors,
            node_size=120,
            edge_color="gray")

    plt.title(f"Worm Propagation - Step {step}")
    plt.pause(0.4)

# =========================
# SIMULATION LOOP
# =========================
for step in range(STEPS):
    new_states = states.copy()

    for node in G.nodes():
        if states[node] == "I":

            # Infect neighbors
            for neighbor in G.neighbors(node):
                if states[neighbor] == "S":
                    if random.random() < INFECTION_PROB:
                        new_states[neighbor] = "I"

            # Recovery
            if random.random() < RECOVERY_PROB:
                new_states[node] = "R"

    states = new_states

    # Track counts
    S_hist.append(list(states.values()).count("S"))
    I_hist.append(list(states.values()).count("I"))
    R_hist.append(list(states.values()).count("R"))

    draw_graph(step)

plt.show()

# =========================
# DATASET FOR ML
# =========================
data = pd.DataFrame({
    "time": range(STEPS),
    "susceptible": S_hist,
    "recovered": R_hist
})

X = data
y = np.array(I_hist)

# =========================
# MACHINE LEARNING MODEL
# =========================
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

pred_train = model.predict(X)
mse = mean_squared_error(y, pred_train)

print("\nModel MSE:", mse)

# =========================
# FUTURE PREDICTION
# =========================
future_steps = 15
future_time = list(range(STEPS, STEPS + future_steps))

future_df = pd.DataFrame({
    "time": future_time,
    "susceptible": [S_hist[-1]] * future_steps,
    "recovered": [R_hist[-1]] * future_steps
})

future_pred = model.predict(future_df)

# =========================
# GRAPH 1: SIR CURVES
# =========================
plt.figure()
plt.plot(S_hist, label="Susceptible")
plt.plot(I_hist, label="Infected")
plt.plot(R_hist, label="Recovered")
plt.title("SIR Worm Propagation")
plt.xlabel("Time")
plt.ylabel("Nodes")
plt.legend()
plt.show()

# =========================
# GRAPH 2: PEAK INFECTION
# =========================
peak = max(I_hist)
peak_time = I_hist.index(peak)

plt.figure()
plt.plot(I_hist, label="Infected")
plt.scatter(peak_time, peak)
plt.text(peak_time, peak, f"Peak={peak}")
plt.title("Peak Infection Point")
plt.xlabel("Time")
plt.ylabel("Infected Nodes")
plt.legend()
plt.show()

print(f"Peak Infection: {peak} at time {peak_time}")

# =========================
# GRAPH 3: GROWTH RATE
# =========================
growth = np.diff(I_hist)

plt.figure()
plt.plot(growth)
plt.title("Infection Growth Rate")
plt.xlabel("Time")
plt.ylabel("Change in Infected")
plt.show()

# =========================
# GRAPH 4: ML vs ACTUAL
# =========================
plt.figure()
plt.plot(I_hist, label="Actual")
plt.plot(pred_train, linestyle='dashed', label="ML Prediction")
plt.title("ML vs Actual")
plt.xlabel("Time")
plt.ylabel("Infected Nodes")
plt.legend()
plt.show()

# =========================
# GRAPH 5: ERROR ANALYSIS
# =========================
errors = abs(pred_train - I_hist)

plt.figure()
plt.plot(errors)
plt.title("Prediction Error")
plt.xlabel("Time")
plt.ylabel("Error")
plt.show()

# =========================
# GRAPH 6: FUTURE PREDICTION
# =========================
plt.figure()
plt.plot(range(STEPS), I_hist, label="Actual")
plt.plot(future_time, future_pred,
         linestyle='dashed', label="Future Prediction")
plt.title("Future Worm Spread Prediction")
plt.xlabel("Time")
plt.ylabel("Infected Nodes")
plt.legend()
plt.show()
