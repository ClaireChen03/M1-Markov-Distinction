import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters for the simulation
Height, Width = 300, 300
NumAgents = 50
NumSteps = 20000
NumPellets_PerColor = 5
Pellet_Radius = 6
Sensory_Dist = 3 # how far ahead agents sense food
Sensory_BiasGain = 2.0 # how strongly sensation of food influences turning
Inertia = 2.0 # tendency to keep going in the same direction
Stay_Baseline = 0.05 # baseline probability of staying in a position
Stay_FoodGain = 1.5 # how strongly food presence influences staying in position
Wall_Bounce = True # Keeps away from walls
SEED = 5

Debug_showpellet = True

rng = np.random.default_rng(SEED)

# Directions for transitions (N, E, S, W, Stay)

DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]], dtype = np.int32)
NORTH, EAST, SOUTH, WEST, STAY = range(5)

# Generate two bright, random colors for food pellets

def random_bright_color(rng):
    v = rng.random(3)
    v = v / (np.linalg.norm(v)  + 1e-9)
    return 0.35 + 0.65 * v

Color_A = random_bright_color(rng)
Color_B = random_bright_color(rng)


# Define the pellet fields for each nutrient (food existence map)

nutrient_A = np.zeros((Height, Width), dtype=np.float32)
nutrient_B = np.zeros((Height, Width), dtype=np.float32)

def place_pellets(field, n, radius, margin = 40):
    ''' 
    Randomly places food pellets of two colors on the canvas. 
    Each pellet starts with a (food existence) value of 1.0, other points are 0.0.
    Margins are left on sides to avoid placing pellets too close to the edge.
    '''
    for _ in range(n):
        cx = rng.integers(margin, Width - margin)
        cy = rng.integers(margin, Height - margin)
        Y, X = np.ogrid[:Height, :Width]
        # Generate a circular pellet with given radius
        mask = (Y-cy) ** 2 + (X-cx) ** 2 <= radius ** 2
        field[mask] = 1.0

place_pellets(nutrient_A, NumPellets_PerColor, Pellet_Radius)
place_pellets(nutrient_B, NumPellets_PerColor, Pellet_Radius)


# Keep agents in boundary of canvas
def in_bounds(pos):
    pos[:, 0] = np.clip(pos[:, 0], 1, Height - 2)
    pos[:, 1] = np.clip(pos[:, 1], 1, Width - 2)
    return pos


# Agents probe the food field at three locations: straight ahead, left and right
def sense_food(pos, direction):
    left = (direction - 1) % 4
    right = (direction + 1) % 4

    pointsAhead = pos + DIRS[direction] * Sensory_Dist
    pointsLeft = pos + DIRS[left] * Sensory_Dist
    pointsRight = pos + DIRS[right] * Sensory_Dist

    for p in (pointsAhead, pointsLeft, pointsRight):
        p[:, 0] = np.clip(p[:, 0], 0, Height - 1)
        p[:, 1] = np.clip(p[:, 1], 0, Width - 1)
    
    # food sensed is a combination of both nutrient fields
    foodAhead = nutrient_A[pointsAhead[:, 0], pointsAhead[:, 1]] + nutrient_B[pointsAhead[:, 0], pointsAhead[:, 1]]
    foodLeft = nutrient_A[pointsLeft[:, 0], pointsLeft[:, 1]] + nutrient_B[pointsLeft[:, 0], pointsLeft[:, 1]]
    foodRight = nutrient_A[pointsRight[:, 0], pointsRight[:, 1]] + nutrient_B[pointsRight[:, 0], pointsRight[:, 1]]
    return foodAhead, foodLeft, foodRight


# Generate the trail network created by slime mold agents
# The intensity shows the brightness and rgb shows the visual color of the trail

trail_intensity = np.zeros((Height, Width), dtype=np.float32)
trail_rgb = np.zeros((Height, Width, 3), dtype=np.float32)


# Initialize agent positions, directions and colors
Agents = np.tile(np.array([Height // 2, Width // 2], dtype=np.int32), (NumAgents, 1))
PrevAgents = Agents.copy()

AgentDirections = rng.integers(0, 4, size = NumAgents)

AgentColorMix = np.full(NumAgents, 0.5, dtype = np.float32)

# Visualize pellets on canvas for debugging
def show_pellets(nutrient_A, nutrient_B, Color_A, Color_B, out_path = "initialize.png"):
    """ Visualizes the food pellets on a canvas and saves the image."""
    canvas = np.zeros((Height, Width, 3), dtype=np.float32)
    mask_A = nutrient_A > 0.0 # Shows occurrence of food A
    mask_B = nutrient_B > 0.0 # Shows occurrence of food B
    canvas[mask_A] = Color_A
    canvas[mask_B] = Color_B
    plt.figure(figsize = (8, 8))
    plt.imshow(canvas, origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    print(f"Wrote {out_path}")
    print(f"Color A: {Color_A}, Color B: {Color_B}")

if __name__ == "__main__":
    if Debug_showpellet:
        show_pellets(nutrient_A, nutrient_B, Color_A, Color_B)
    # Main simulation loop
    for t in range(NumSteps):

        BiasGainNow = 1.0 + 3.0 * (t / NumSteps)
        Sensory_BiasGain = BiasGainNow

        # Sense food at three probe locations for each agent
        foodAhead, foodLeft, foodRight = sense_food(Agents, AgentDirections)
        # Build transition matrix for states (N, E, S, W, Stay)
        transition_matrix = np.ones((NumAgents, 5), dtype=np.float32)
        # Tendency to go straight
        transition_matrix[np.arange(NumAgents), AgentDirections] *= Inertia
        # Add bias to turn towards food
        turnLeft = (AgentDirections - 1) % 4
        turnRight = (AgentDirections + 1) % 4
        transition_matrix[np.arange(NumAgents), turnLeft]  *= (1.0 + Sensory_BiasGain * foodLeft)
        transition_matrix[np.arange(NumAgents), turnRight] *= (1.0 + Sensory_BiasGain * foodRight)
        transition_matrix[np.arange(NumAgents), AgentDirections] *= (1.0 + Sensory_BiasGain * foodAhead)

        # Add bias to stay if food is present
        y = Agents[:, 0]; x = Agents[:, 1]
        onA = nutrient_A[y, x] > 0.5
        onB = nutrient_B[y, x] > 0.5
        onFood = onA | onB
        transition_matrix[:, STAY] = Stay_Baseline
        transition_matrix[onFood, STAY] *= Stay_FoodGain

        # Add bias to turn away from walls if close to edge
        near_top = Agents[:, 0] < 4
        near_bottom = Agents[:, 0] > Height - 5
        near_left = Agents[:, 1] < 4
        near_right = Agents[:, 1] > Width - 5
        transition_matrix[near_top, SOUTH] *= 2.0
        transition_matrix[near_bottom, NORTH] *= 2.0
        transition_matrix[near_left, EAST] *= 2.0
        transition_matrix[near_right, WEST] *= 2.0

        # Normalize probabilities
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        # Sample next direction from transition matrix
        cumulative_matrix = np.cumsum(transition_matrix, axis=1)
        random_values = rng.random((NumAgents, 1))
        AgentDirections = (random_values < cumulative_matrix).argmax(axis=1)

        # Move agents
        PrevAgents[:] = Agents
        stepVectors = DIRS[AgentDirections]
        Agents = Agents + stepVectors

        # Keep agents in bounds
        Agents = in_bounds(Agents)

        # Update agent color mix
        y = Agents[:, 0]; x = Agents[:, 1]
        onA = nutrient_A[y, x] > 0.5
        onB = nutrient_B[y, x] > 0.5
        AgentColorMix[onA] = 0.9 * AgentColorMix[onA] + 0.1 * 0.0
        AgentColorMix[onB] = 0.9 * AgentColorMix[onB] + 0.1 * 1.0 

        # Food attraction decreases over time
        nutrient_A[y, x] *= 0.9995
        nutrient_B[y, x] *= 0.9995

        # Draw trail
  

    # Debug render: pellets + final agent positions
    canvas = np.zeros((Height, Width, 3), dtype=np.float32)
    canvas[nutrient_A > 0] = Color_A
    canvas[nutrient_B > 0] = Color_B
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, origin='upper')
    # overlay agent dots so you can see where they ended up
    ay, ax = Agents[:, 0], Agents[:, 1]
    plt.scatter(ax, ay, s=5, c='white', alpha=0.9, marker='.')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("step1_agents_debug.png", dpi=300, bbox_inches='tight', pad_inches=0)
    print("Wrote step1_agents_debug.png (pellets + final agent positions)")


    

