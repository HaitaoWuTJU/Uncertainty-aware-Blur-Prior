import cv2
import numpy as np

# Load generated blurred images
img_low = cv2.imread("blurred_low.jpg")
img_high = cv2.imread("blurred_high.jpg")

if img_low is None or img_high is None:
    raise FileNotFoundError("Could not load 'blurred_low.jpg' or 'blurred_high.jpg'. Ensure they exist in the current directory.")

# Calculate absolute pixel-by-pixel difference
diff = cv2.absdiff(img_low, img_high)

# Numerical difference statistics
mean_diff = np.mean(diff)
max_diff = np.max(diff)
differing_pixels = np.count_nonzero(diff)
total_values = diff.size
pct_differing = (differing_pixels / total_values) * 100

print(f"--- Difference Analysis (Low vs High) ---")
print(f"Max pixel difference:       {max_diff}")
print(f"Mean pixel difference:      {mean_diff:.4f}")
print(f"Differing channels/pixels:  {differing_pixels}/{total_values} ({pct_differing:.2f}%)")

# Amplify the difference by 15x for visual clarity and save to disk
amplified_diff = np.clip(diff.astype(np.uint16) * 15, 0, 255).astype(np.uint8)
output_path = "diff_low_vs_high.jpg"
cv2.imwrite(output_path, amplified_diff)

print(f"\nAmplified difference image saved to: {output_path}")