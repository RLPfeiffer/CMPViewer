import tkinter as tk
from tkinter import ttk


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ImageViewer")

        # Dictionary to organize images by type
        self.images_by_type = {
            "grayscale": [],
            "rgb": [],
            "clustered": []
        }

        # Sample image data (replace with actual image loading logic)
        self.load_sample_images()

        # GUI Setup
        self.setup_ui()

    def load_sample_images(self):
        # Simulate loading images with types
        for i in range(5):
            self.images_by_type["grayscale"].append(f"gray_image_{i}.png")
            self.images_by_type["rgb"].append(f"rgb_image_{i}.png")
            self.images_by_type["clustered"].append(f"clustered_image_{i}.png")

    def setup_ui(self):
        # Image Display Frame
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.image_label = tk.Label(self.display_frame)
        self.image_label.pack()

        # Control Panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Image Type Selection
        tk.Label(self.control_frame, text="Select Image Type").pack()
        self.type_var = tk.StringVar(value="grayscale")
        tk.OptionMenu(self.control_frame, self.type_var, "grayscale", "rgb", "clustered",
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
        selected_type = self.type_var.get()
        self.image_dropdown['values'] = self.images_by_type[selected_type]
        self.image_var.set(self.images_by_type[selected_type][0] if self.images_by_type[selected_type] else "")

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