import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw

class ImageRegionZoom:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.original_img = None
        self.result_img = None
        self.roi = None  # (x, y, width, height)
        self.zoom_factor = 2

        if self.image_path:
            self.load_image()

    def load_image(self):
        """Load image"""
        try:
            self.original_img = Image.open(self.image_path)
            # Convert to RGB mode to ensure compatibility
            if self.original_img.mode != 'RGB':
                self.original_img = self.original_img.convert('RGB')
            return self.original_img
        except Exception as e:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}. Error: {e}")

    def set_roi(self, x, y, width, height):
        """Set region of interest"""
        self.roi = (x, y, width, height)

    def set_zoom_factor(self, factor):
        """Set zoom magnification"""
        self.zoom_factor = factor

    def select_roi_interactive(self):
        """Interactive ROI selection"""
        if self.original_img is None:
            raise ValueError("Please load image first")

        # Create selection window
        select_window = tk.Tk()
        select_window.title("Select Region of Interest")
        select_window.geometry("800x600")

        # Resize image to fit window
        img_width, img_height = self.original_img.size
        max_width, max_height = 750, 500

        scale = min(max_width / img_width, max_height / img_height)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        display_img = self.original_img.copy()
        display_img = display_img.resize((display_width, display_height), Image.LANCZOS)

        # Convert to tkinter format
        tk_image = ImageTk.PhotoImage(display_img)

        # Create canvas to display image
        canvas = tk.Canvas(select_window, width=display_width, height=display_height)
        canvas.pack(pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        # Add info label
        info_label = tk.Label(select_window, text="Click two diagonal points to select ROI")
        info_label.pack()

        # Store click coordinates
        self.click_points = []
        self.scale_factor = scale
        self.display_width = display_width
        self.display_height = display_height

        def on_click(event):
            if len(self.click_points) < 2:
                x, y = event.x, event.y
                # Convert to original image coordinates
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)

                self.click_points.append((orig_x, orig_y))

                # Draw click point on canvas
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")

                if len(self.click_points) == 2:
                    # Calculate ROI area
                    x1, y1 = self.click_points[0]
                    x2, y2 = self.click_points[1]

                    # Ensure correct coordinate order
                    roi_x = min(x1, x2)
                    roi_y = min(y1, y2)
                    roi_width = abs(x2 - x1)
                    roi_height = abs(y2 - y1)

                    self.set_roi(roi_x, roi_y, roi_width, roi_height)

                    # Draw selection box
                    canvas.create_rectangle(
                        roi_x * self.scale_factor,
                        roi_y * self.scale_factor,
                        (roi_x + roi_width) * self.scale_factor,
                        (roi_y + roi_height) * self.scale_factor,
                        outline="green", width=2
                    )

                    info_label.config(text=f"Selected area: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
                    select_window.after(1000, select_window.destroy)

        # Bind mouse click event
        canvas.bind("<Button-1>", on_click)

        # Keep reference to image
        canvas.image = tk_image

        select_window.mainloop()

        return len(self.click_points) == 2

    def process_image(self):
        """Process image: zoom ROI and display on right side of original image"""
        if self.original_img is None:
            raise ValueError("Please load image first")

        if self.roi is None:
            raise ValueError("Please set region of interest first")

        x, y, width, height = self.roi

        # Check if ROI is within image bounds
        img_width, img_height = self.original_img.size
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            raise ValueError("ROI exceeds image boundaries")

        # Extract region of interest
        roi_img = self.original_img.crop((x, y, x + width, y + height))

        # Zoom in the ROI
        zoomed_roi = roi_img.resize(
            (int(width * self.zoom_factor), int(height * self.zoom_factor)),
            Image.LANCZOS
        )

        # Create result image, display original and zoomed area side by side
        original_width, original_height = self.original_img.size
        zoomed_width, zoomed_height = zoomed_roi.size

        # Calculate result image dimensions
        result_width = original_width + zoomed_width + 20  # 20 is spacing
        result_height = max(original_height, zoomed_height)

        # Create new result image
        self.result_img = Image.new('RGB', (result_width, result_height), (255, 255, 255))

        # Paste original image on left side
        self.result_img.paste(self.original_img, (0, 0))

        # Paste zoomed ROI on right side
        paste_x = original_width + 20
        paste_y = (result_height - zoomed_height) // 2  # Vertical centering
        self.result_img.paste(zoomed_roi, (paste_x, paste_y))

        # Create drawable object for adding borders and labels
        draw = ImageDraw.Draw(self.result_img)

        # Draw border for original ROI on original image
        draw.rectangle([x, y, x + width, y + height], outline=(0, 255, 0), width=2)

        # Draw border around zoomed area
        draw.rectangle([paste_x, paste_y, paste_x + zoomed_width, paste_y + zoomed_height],
                       outline=(255, 0, 0), width=2)

        try:
            # Add labels
            draw.text((x, y - 10), fill=(0, 255, 0))
            draw.text((paste_x, paste_y - 10), fill=(255, 0, 0))
        except:
            # If font is not available, don't add text
            pass

        # Draw connection line
        # From right midpoint of ROI to left midpoint of zoomed area
        start_x = x + width
        start_y = y + height // 2
        end_x = paste_x
        end_y = paste_y + zoomed_height // 2
        draw.line([start_x, start_y, end_x, end_y], fill=(0, 0, 255), width=1)

        return self.result_img

    def show_image(self):
        """Display processed image"""
        if self.result_img is None:
            raise ValueError("Please process image first")

        # Create Tkinter window to display image
        root = tk.Tk()
        root.title("Original Image and Zoomed Region")

        # Resize image to fit screen
        img_width, img_height = self.result_img.size
        max_width, max_height = 1200, 700

        scale = min(max_width / img_width, max_height / img_height, 1.0)  # Don't enlarge
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        display_img = self.result_img.copy()
        display_img = display_img.resize((display_width, display_height), Image.LANCZOS)

        # Convert PIL image to tkinter format
        tk_image = ImageTk.PhotoImage(display_img)

        # Create label to display image
        label = tk.Label(root, image=tk_image)
        label.pack()

        # Add info label
        info_label = tk.Label(root, text="Close window button to exit")
        info_label.pack()

        # Keep reference to image to prevent garbage collection
        label.image = tk_image

        root.mainloop()

    def save_result(self, output_path):
        """Save processed image"""
        if self.result_img is None:
            raise ValueError("No processed image to save")
        self.result_img.save(output_path)
        print(f"Image saved to: {output_path}")

def select_image_path():
    """Use file dialog to select image path"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(
        title="Select image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def get_zoom_factor():
    """Get user input zoom factor"""
    root = tk.Tk()
    root.withdraw()
    try:
        result = simpledialog.askstring("Input", "Please enter zoom factor:", initialvalue="2")
        root.destroy()
        if result:
            return float(result)
        return 2.0
    except (ValueError, TypeError):
        root.destroy()
        return 2.0

def ask_save_option():
    """Ask whether to save result"""
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Save", "Save result image?")
    root.destroy()
    return result

def select_save_path():
    """Select save path"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save image",
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def main():
    """Main function - load one image and process"""
    try:
        # Select image file
        image_path = select_image_path()
        if not image_path:
            print("No image file selected")
            return

        # Create image processing object
        image_zoom = ImageRegionZoom(image_path)

        # Load image
        print("Loading image...")
        image_zoom.load_image()
        print("Image loaded successfully")

        # Interactive ROI selection
        print("Please click two diagonal points in popup window to select ROI...")
        if not image_zoom.select_roi_interactive():
            print("ROI selection failed")
            return

        # Set zoom factor
        zoom_factor = get_zoom_factor()
        image_zoom.set_zoom_factor(zoom_factor)

        # Process image
        print("Processing image...")
        result = image_zoom.process_image()
        print("Image processing completed")

        # Display result
        print("Displaying result image...")
        image_zoom.show_image()

        # Save result
        if ask_save_option():
            output_path = select_save_path()
            if output_path:
                image_zoom.save_result(output_path)
            else:
                print("No save path selected")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure image file exists")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
