import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters for the simulation
Height, Width = 500, 500
NumAgents = 30
NumSteps = 5000
NumPellets_PerColor = 5
Pellet_Radius = 8

Debug_showpellet = True

rng = np.random.default_rng(7)

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


# Generate the trail network created by slime mold agents
# The intensity shows the brightness and rgb shows the visual color of the trail

trail_intensity = np.zeros((Height, Width), dtype=np.float32)
trail_rgb = np.zeros((Height, Width, 3), dtype=np.float32)


# Visualize pellets on canvas for debugging
def show_pellets(nutrient_A, nutrient_B, Color_A, Color_B, out_path = "initialize.png"):
    """ Visualizes the food pellets on a canvas and saves the image."""
    canvas = np.zeros((Height, Width, 3), dtype=np.float32)
    mask_A = nutrient_A > 0.0 # Shows occurrence of food A
    mask_B = nutrient_B > 0.0 # Shows occurrence of food B
    canvas[mask_A] = Color_A
    canvas[mask_B] = Color_B
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    print(f"Wrote {out_path}")
    print(f"Color A: {Color_A}, Color B: {Color_B}")

if __name__ == "__main__":
    if Debug_showpellet:
        show_pellets(nutrient_A, nutrient_B, Color_A, Color_B)
    else:
        print(f"false show")

