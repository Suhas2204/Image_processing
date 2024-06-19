import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load the image
img = Image.open('example.jpg').convert('L')
img_array = np.array(img)

# Apply simple edge detection filter
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
edges = np.abs(np.convolve(img_array.flatten(),
                           kernel.flatten(), mode='same'))
edges = edges.reshape(img_array.shape)


# Display the result
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.show()
