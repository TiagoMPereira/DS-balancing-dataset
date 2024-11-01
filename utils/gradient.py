from PIL import Image

# Define the colors
colors = [
    (165, 0, 38),    # cbdarkred
    (215, 48, 39),   # cbmediumred
    (244, 109, 67),  # cbdarkorange
    (253, 174, 97),  # cbmediumorange
    (254, 224, 139), # cblightorange
    (217, 239, 139), # cbtintgreen
    (166, 217, 106), # cblightgreen
    (102, 189, 99),  # cbmediumgreen
    (26, 152, 80),   # cbdimgreen
    (0, 104, 55)     # cbdarkgreen
]

# Invert the color order
colors.reverse()

# Set image dimensions
width = 1700  # Width of the image
height = 1000  # Height of the image

# Create a new image with the specified dimensions
img = Image.new('RGB', (width, height))

# Function to interpolate between two colors
def interpolate(color1, color2, t):
    return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))

# Calculate the number of color bands
num_bands = len(colors) - 1
band_height = height // num_bands

# Draw the gradient
for i in range(num_bands):
    start_color = colors[i]
    end_color = colors[i + 1]
    for y in range(band_height):
        t = y / band_height  # Calculate interpolation factor
        current_color = interpolate(start_color, end_color, t)
        for x in range(width):
            img.putpixel((x, i * band_height + y), current_color)

# Handle any remaining pixels if the height is not exactly divisible by the number of bands
remaining_pixels = height % num_bands
if remaining_pixels > 0:
    start_color = colors[-2]
    end_color = colors[-1]
    for y in range(remaining_pixels):
        t = y / remaining_pixels
        current_color = interpolate(start_color, end_color, t)
        for x in range(width):
            img.putpixel((x, num_bands * band_height + y), current_color)

# Save the image
img.save('smooth_vertical_gradient_inverted.png')

# To show the image uncomment the line below
# img.show()
