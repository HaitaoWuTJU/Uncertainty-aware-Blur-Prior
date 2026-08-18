import cv2
import numpy as np
from PIL import Image
from scipy.optimize import fsolve
import os

class FoveaBlur:
    """
    Exact FoveaBlur implementation extracted from the repository.
    """
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h,w), np.float32)
        
        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1-c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]
        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]
        
        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1,distance/max_distance)
                y0 = fun_degrade(x0,**kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
        return blended
    
    def __call__(self, img, blur_kernel_size=None): 
        if blur_kernel_size == None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Apply the target blur size
        blured = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        
        # Blend the sharp image and blurred image using the exponential mask
        blended = self.alphaBlend(img, blured, 1- self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
    
    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

def apply_repo_blur(image_path, blur_level):
    """
    Applies the low, medium (middle), or high FoveaBlur to a 224x224 jpg image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # The repo uses a base kernel of 51 and a shift (c) of 6.
    base_kernel_size = 51
    shift_c = 6
    system_g = 3
    
    # Map levels to their exact kernel sizes based on the repository's shift logic
    kernel_sizes = {
        "low": base_kernel_size - shift_c,      # 45
        "middle": base_kernel_size,             # 51 (referred to as 'medium' in repo)
        "high": base_kernel_size + shift_c      # 57
    }

    if blur_level not in kernel_sizes:
        raise ValueError("blur_level must be 'low', 'middle', or 'high'")

    target_kernel_size = kernel_sizes[blur_level]

    # Load the image (assuming it is already 224x224)
    image = Image.open(image_path).convert("RGB")

    # Instantiate the exact blur class used in the repo
    blur_tool = FoveaBlur(
        h=224, 
        w=224, 
        blur_kernel_size=target_kernel_size, 
        curve_type='exp', 
        system_g=system_g
    )

    # Apply the blur
    blurred_image = blur_tool(image)
    return blurred_image

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    # REPLACE this with the actual path to your 224x224 image
    input_image = "/work3/s193209/data/images/Image_set_Resize/test_images/00022_bread/bread_19s.jpg" 
    
    # Process and save an example for each blur level
    for level in ["low", "middle", "high"]:
        try:
            result_img = apply_repo_blur(input_image, level)
            output_filename = f"blurred_{level}.jpg"
            result_img.save(output_filename)
            print(f"Successfully saved: {output_filename}")
            
        except FileNotFoundError:
            print(f"Error: Could not find the image at '{input_image}'. Please update the path.")
            break