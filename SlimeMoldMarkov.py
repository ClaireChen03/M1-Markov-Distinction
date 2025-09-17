"""
File name: SlimeMoldMarkov.py
Author: Claire Chen
Date last modified: 16 September, 2025
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Here are the simulation parameters you can play with 
to change the behavior of the slime mold agents.
"""

# Canvas settings (each seed will generate a different pattern)
HEIGHT = 300
WIDTH = 300
SEED = 5 

# Agent settings
NUM_AGENTS = 50
NUM_STEPS = 10000

# Food pellet settings
NUM_PELLETS_PER_COLOR = 5
PELLET_RADIUS = 6
FOOD_DECAY_RATE = 0.999   # How fast food attraction decays over time

# Agent behavior settings
SENS_DIST = 3   # How far agents sense food
SENSE_BIAS_GAIN = 1.2   # How sensation of food influences turning (can choose this or bias_gain_now in main loop)
INERTIA = 1.5   # Tendency to maintain direction
BASELINE_STAY = 0.02   # Baseline probability to STAY
FOODGAIN_STAY = 1.5   # Food influence on STAY

# Trail settings
COLOR_DEPOSIT = 0.9   # Trail intensity to add per step
COLOR_BLEND = 0.12   # Trail color blend rate


"""
Initialization
"""

# Direction constants
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]], dtype = np.int32)
NORTH, EAST, SOUTH, WEST, STAY = range(5)

# Initialize random number generator
rng = np.random.default_rng(SEED)

# Initialize fields (Food A, Food B, Trail Intensity, Trail Color)
nutrient_A = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
nutrient_B = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
trail_intensity = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
trail_rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

# Initialize agents
agents = np.tile(np.array([HEIGHT // 2, WIDTH // 2], dtype=np.int32), (NUM_AGENTS, 1))
prev_agents = agents.copy()
agent_directions = rng.integers(0, 4, size = NUM_AGENTS)
agent_color_mix = np.full(NUM_AGENTS, 0.5, dtype = np.float32)


"""
Helper Functions
"""

def random_bright_color(rng):
    '''Returns a random bright color as an RGB numpy array.'''
    v = rng.random(3)
    v = v / (np.linalg.norm(v)  + 1e-9)
    return 0.35 + 0.65 * v


def place_pellets(field, n, radius, margin = 40):
    ''' 
    Randomly places food pellets of two colors on the canvas. 
    Each pellet starts with a (food existence) value of 1.0, other points are 0.0.
    Margins are left on sides to avoid placing pellets too close to the edge.
    '''
    for _ in range(n):
        cx = rng.integers(margin, WIDTH - margin)
        cy = rng.integers(margin, HEIGHT - margin)
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        # Generate a circular pellet with given radius
        mask = (Y-cy) ** 2 + (X-cx) ** 2 <= radius ** 2
        field[mask] = 1.0


def in_bounds(pos):
    '''
    Keeps agent positions within canvas boundaries. 
    Returns clipped positions within valid range.
    '''
    pos[:, 0] = np.clip(pos[:, 0], 1, HEIGHT - 2)
    pos[:, 1] = np.clip(pos[:, 1], 1, WIDTH - 2)
    return pos


def sense_food(pos, direction):
    '''
    Agents sense food at three probe locations: ahead, left, and right.
    Returns amount of food sensed (ahead, left, right) for each agent.
    '''
    left = (direction - 1) % 4
    right = (direction + 1) % 4

    pointsAhead = pos + DIRS[direction] * SENS_DIST
    pointsLeft = pos + DIRS[left] * SENS_DIST
    pointsRight = pos + DIRS[right] * SENS_DIST

    for p in (pointsAhead, pointsLeft, pointsRight):
        p[:, 0] = np.clip(p[:, 0], 0, HEIGHT - 1)
        p[:, 1] = np.clip(p[:, 1], 0, WIDTH - 1)
    
    # Food sensed is a combination of both nutrient fields
    foodAhead = nutrient_A[pointsAhead[:, 0], pointsAhead[:, 1]] + nutrient_B[pointsAhead[:, 0], pointsAhead[:, 1]]
    foodLeft = nutrient_A[pointsLeft[:, 0], pointsLeft[:, 1]] + nutrient_B[pointsLeft[:, 0], pointsLeft[:, 1]]
    foodRight = nutrient_A[pointsRight[:, 0], pointsRight[:, 1]] + nutrient_B[pointsRight[:, 0], pointsRight[:, 1]]
    return foodAhead, foodLeft, foodRight


def render_agents(agents, out_path):
    '''
    Renders the positions of food pellets and final agent positions.
    '''
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    canvas[nutrient_A > 0] = Color_A
    canvas[nutrient_B > 0] = Color_B
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, origin='upper')
    # Overlay agent dots so you can see where they ended up
    ay, ax = agents[:, 0], agents[:, 1]
    plt.scatter(ax, ay, s=5, c='white', alpha=0.9, marker='.')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Wrote {out_path} (pellets + final agent positions)")


def render_trail(trail_intensity, trail_rgb, out_path):
    '''
    Renders the final trail network with color blending.
    '''
    # Normalize intensity for contrast and blend with trail RGB
    intensity = trail_intensity / (trail_intensity.max() + 1e-9)
    color_img = np.clip(trail_rgb, 0, 1)
    # Mix a bit of grayscale intensity to enhance structure
    color_out = 0.25 * np.dstack([intensity, intensity, intensity]) + 0.75 * color_img
    color_out = np.clip(color_out, 0, 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(color_out, origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Wrote {out_path} (final colored trail network)")


# Generate initial colors and place food pellets
Color_A = random_bright_color(rng)
Color_B = random_bright_color(rng)
place_pellets(nutrient_A, NUM_PELLETS_PER_COLOR, PELLET_RADIUS)
place_pellets(nutrient_B, NUM_PELLETS_PER_COLOR, PELLET_RADIUS)

"""
Run simulation and render images.
"""

def main():
    '''
    Main simulation loop.
    Each agent senses food, decides on a movement direction based on a transition matrix,
    moves, deposits trail, and updates its color based on the food it encounters.
    '''
    global agents, agent_directions, agent_color_mix
    print("Starting simulation")

    for t in range(NUM_STEPS):
        # Gradually increase food attraction over time (or use fixed SENSE_BIAS_GAIN)
        bias_gain_now = 1.0 + 2.0 * (t / NUM_STEPS)

        # Sense food at three locations per agent
        foodAhead, foodLeft, foodRight = sense_food(agents, agent_directions)
        # Build transition matrix for movement probabilities
        transition_matrix = np.ones((NUM_AGENTS, 5), dtype=np.float32)
        # Tendency to go straight
        transition_matrix[np.arange(NUM_AGENTS), agent_directions] *= INERTIA
        # Tendency to turn towards food
        turnLeft = (agent_directions - 1) % 4
        turnRight = (agent_directions + 1) % 4
        transition_matrix[np.arange(NUM_AGENTS), turnLeft]  *= (1.0 + bias_gain_now * foodLeft)
        transition_matrix[np.arange(NUM_AGENTS), turnRight] *= (1.0 + bias_gain_now * foodRight)
        transition_matrix[np.arange(NUM_AGENTS), agent_directions] *= (1.0 + bias_gain_now * foodAhead)
        # Bias to stay if food is present
        y = agents[:, 0]; x = agents[:, 1]
        onA = nutrient_A[y, x] > 0.5
        onB = nutrient_B[y, x] > 0.5
        onFood = onA | onB
        transition_matrix[:, STAY] = BASELINE_STAY
        transition_matrix[onFood, STAY] *= FOODGAIN_STAY
        # Bias away from walls
        near_top = agents[:, 0] < 4
        near_bottom = agents[:, 0] > HEIGHT - 5
        near_left = agents[:, 1] < 4
        near_right = agents[:, 1] > WIDTH - 5
        transition_matrix[near_top, SOUTH] *= 2.0
        transition_matrix[near_bottom, NORTH] *= 2.0
        transition_matrix[near_left, EAST] *= 2.0
        transition_matrix[near_right, WEST] *= 2.0

        # Normalize probabilities
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        # Sample next direction from transition matrix
        cumulative_matrix = np.cumsum(transition_matrix, axis=1)
        random_values = rng.random((NUM_AGENTS, 1))
        agent_directions = (random_values < cumulative_matrix).argmax(axis=1)

        # Move agents
        prev_agents[:] = agents
        stepVectors = DIRS[agent_directions]
        agents = in_bounds(agents + stepVectors)   # Keep agents in bounds

        # Update agent color
        y = agents[:, 0]; x = agents[:, 1]
        onA = nutrient_A[y, x] > 0.5
        onB = nutrient_B[y, x] > 0.5
        agent_color_mix[onA] = 0.9 * agent_color_mix[onA] + 0.1 * 0.0
        agent_color_mix[onB] = 0.9 * agent_color_mix[onB] + 0.1 * 1.0 

        # Update trail intensity and color
        for i in range(NUM_AGENTS):
            y1, x1 = int(agents[i, 0]), int(agents[i, 1])
            m = float(agent_color_mix[i])
            agent_color = (1.0 - m) * Color_A + m * Color_B
            trail_intensity[y1, x1] += COLOR_DEPOSIT
            trail_rgb[y1, x1] = (1.0 - COLOR_BLEND) * trail_rgb[y1, x1] + COLOR_BLEND * agent_color
        
        # Food attraction decreases over time
        nutrient_A[y, x] *= FOOD_DECAY_RATE
        nutrient_B[y, x] *= FOOD_DECAY_RATE


    render_agents(agents, f"agent_final_positions_seed{SEED}.png")   # Render pellets + final agent positions
    render_trail(trail_intensity, trail_rgb, f"slime_trail_simulate_seed{SEED}.png")   # Render final colored trail network


if __name__ == "__main__":
    main()