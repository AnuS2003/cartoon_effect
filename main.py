import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    img = cv2.imread(filename)
    if img is None:
        print(f"Error: Unable to load image from {filename}")
        return None
    return img

def create_edgeMask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def cartoon_effect(quantized_img, edge_mask):
    edge_mask_colored = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized_img, edge_mask_colored)
    return cartoon

def apply_vignette(img):
    rows, cols = img.shape[:2]
    X, Y = np.arange(cols), np.arange(rows)
    X, Y = np.meshgrid(X, Y)
    X, Y = X - cols / 2, Y - rows / 2
    D = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1 - D / (np.sqrt((cols / 2) ** 2 + (rows / 2) ** 2)), 0, 1)
    vignette = np.expand_dims(vignette, axis=2)
    return np.uint8(img * vignette)

def adjust_saturation(img, factor):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def apply_gaussian_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def save_image(image, filename):
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def display_images(original, edge_mask, quantized, blurred, cartoon, vignette, saturated):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edge_mask, cmap='gray')
    axes[0, 1].set_title('Edge Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Quantized Image')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Blurred Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Cartoon Effect')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(vignette, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Vignette Effect')
    axes[1, 2].axis('off')

    axes[2, 0].imshow(cv2.cvtColor(saturated, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Saturated Image')
    axes[2, 0].axis('off')

    # Add empty subplots if less than 9 images
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main(filename):
    img = read_file(filename)
    if img is not None:
        line_size = 7
        blur_value = 5
        k = 3
        gaussian_blur_size = 7
        saturation_factor = 1.5

        edge_mask = create_edgeMask(img, line_size, blur_value)
        quantized_img = color_quantisation(img, k)
        blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
        cartoon_img = cartoon_effect(quantized_img, edge_mask)
        vignette_img = apply_vignette(cartoon_img)
        saturated_img = adjust_saturation(cartoon_img, saturation_factor)

        save_image(edge_mask, 'edge_mask.png')
        save_image(quantized_img, 'quantized_img.png')
        save_image(blurred, 'blurred_img.png')
        save_image(cartoon_img, 'cartoon_img.png')
        save_image(vignette_img, 'vignette_img.png')
        save_image(saturated_img, 'saturated_img.png')

        display_images(img, edge_mask, quantized_img, blurred, cartoon_img, vignette_img, saturated_img)

filename = r"/home/anupama/Desktop/opencv/images2.jpeg"
main(filename)
