from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# A simulated "fixed" maximum resolution the model can handle
MAX_RESOLUTION = (100, 100)


def create_mock_image(size=(500, 500), detail_pos=(450, 450)):
    """Creates a large mock image with a single red pixel as the 'detail'."""
    image = Image.new('RGB', size, 'white')
    pixels = image.load()
    pixels[detail_pos] = (255, 0, 0)  # Place a red pixel as a detail
    return image, detail_pos


def fixed_resolution_processor(image):
    """
    Simulates a traditional model that resizes the image, losing detail.
    """
    resized_image = image.resize(MAX_RESOLUTION)
    return resized_image


def naive_dynamic_resolution_processor(image, patch_size=MAX_RESOLUTION):
    """
    Simulates Naive Dynamic Resolution by processing patches of the original image.
    Returns the patch that contains the detail for visualization.
    """
    width, height = image.size

    # Iterate through the image in patches
    for i in range(0, width, patch_size[0]):
        for j in range(0, height, patch_size[1]):
            patch = image.crop((i, j, i + patch_size[0], j + patch_size[1]))

            # Check for the red pixel in the high-resolution patch
            if np.any(np.array(patch) == [255, 0, 0]):
                return patch, (i, j)  # Return the patch and its coordinates

    return None, None  # Detail not found


# --- Main Program ---
if __name__ == "__main__":
    large_image, detail_pos = create_mock_image()

    # --- Part 1: Fixed Resolution Model ---
    resized_image = fixed_resolution_processor(large_image)

    # --- Part 2: Dynamic Resolution Model ---
    found_patch, patch_coords = naive_dynamic_resolution_processor(large_image)

    # --- Part 3: Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Original Image
    axes[0].imshow(large_image)
    axes[0].set_title("1. Original High-Resolution Image")
    axes[0].text(detail_pos[0], detail_pos[1], 'X', color='red', fontsize=12, ha='center')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot 2: Downsampled Image (Fixed Resolution)
    axes[1].imshow(resized_image)
    axes[1].set_title("2. Fixed-Resolution Downsampled Image\n(Detail is lost)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Plot 3: Dynamic Resolution with Patches and Highlight
    axes[2].imshow(large_image)
    axes[2].set_title("3. Naive Dynamic Resolution\n(Processing High-Res Patches)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Add a grid to show the patches
    for i in range(0, large_image.size[0], MAX_RESOLUTION[0]):
        for j in range(0, large_image.size[1], MAX_RESOLUTION[1]):
            rect = patches.Rectangle((i, j), MAX_RESOLUTION[0], MAX_RESOLUTION[1], linewidth=1, edgecolor='gray',
                                     facecolor='none')
            axes[2].add_patch(rect)

    # Add the text to plot 3 as well
    axes[2].text(detail_pos[0], detail_pos[1], 'X', color='red', fontsize=12, ha='center')

    # Add a highlight rectangle around the patch with the detail
    if patch_coords:
        rect = patches.Rectangle(patch_coords, MAX_RESOLUTION[0], MAX_RESOLUTION[1], linewidth=3, edgecolor='green',
                                 facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(patch_coords[0] + 5, patch_coords[1] + 15, "Patch with detail", color='green', fontsize=10,
                     weight='bold')

    plt.tight_layout()
    plt.savefig('naive_dynamic_resolution.png')