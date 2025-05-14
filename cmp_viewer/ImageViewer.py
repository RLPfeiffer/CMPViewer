import tkinter as tk
from tkinter import ttk


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ImageViewer")

        # Dictionary to organize images by category
        self.images_by_category = {
            "Raw Image Data": [],  # All loaded images
            "Processed Images": [],  # Images used for clustering
            "Image Clusters": []  # Combined cluster overlays (cell type classes)
        }

        # Sample data to simulate your workflow (replace with actual image loading logic)
        self.load_sample_images()

        # GUI Setup
        self.setup_ui()

    def load_sample_images(self):
        # Simulate loading raw images (all loaded images)
        for i in range(5):
            self.images_by_category["Raw Image Data"].append(f"raw_image_{i}.png")

        # Simulate processed images (subset of raw images used for clustering)
        # For this example, let's assume images 0, 1, and 3 were used for clustering
        self.images_by_category["Processed Images"] = [
            self.images_by_category["Raw Image Data"][i] for i in [0, 1, 3]
        ]

        # Simulate image clusters (combined cluster overlays for cell type classes)
        # Based on your log, clusters are toggled and combined; here we simulate the result
        for i in range(3):
            self.images_by_category["Image Clusters"].append(f"cluster_overlay_{i}.png")

    def setup_ui(self):
        # Image Display Frame
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.image_label = tk.Label(self.display_frame)
        self.image_label.pack()

        # Control Panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Category Selection
        tk.Label(self.control_frame, text="Select Image Category").pack()
        self.category_var = tk.StringVar(value="Raw Image Data")
        tk.OptionMenu(self.control_frame, self.category_var, "Raw Image Data", "Processed Images", "Image Clusters",
                      command=self.update_image_list).pack()

        # Image Selection
        tk.Label(self.control_frame, text="Select Image to Display").pack()
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(self.control_frame, textvariable=self.image_var)
        self.update_image_list()
        self.image_dropdown.pack()

        # Display Button
        tk.Button(self.control_frame, text="Display Image", command=self.display_image).pack(pady=5)

    def update_image_list(self, *args):
        selected_category = self.category_var.get()
        self.image_dropdown['values'] = self.images_by_category[selected_category]
        self.image_var.set(
            self.images_by_category[selected_category][0] if self.images_by_category[selected_category] else "")

    def display_image(self):
        selected_image = self.image_var.get()
        if selected_image:
            # Simulate image display (replace with actual image rendering logic)
            self.image_label.config(text=f"Displaying: {selected_image}")
        else:
            self.image_label.config(text="No image selected")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()