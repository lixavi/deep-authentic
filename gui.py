import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from synthesis import DeepFakeSynthesizer
from detection import DeepFakeDetector

class DeepFakeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFake Tool")

        # Create buttons
        self.detect_button = tk.Button(self.root, text="Detect DeepFake", command=self.detect_deepfake)
        self.detect_button.pack(pady=10)

        self.create_button = tk.Button(self.root, text="Create DeepFake", command=self.create_deepfake)
        self.create_button.pack(pady=10)

        # Initialize DeepFakeSynthesizer and DeepFakeDetector
        self.synthesizer = DeepFakeSynthesizer(generator_model_path='path_to_generator_model.pth')
        self.detector = DeepFakeDetector(model_path='path_to_detector_model.pth')

    def detect_deepfake(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        image = Image.open(file_path)
        result = self.detector.detect_deepfake(image)
        # Display detection result

    def create_deepfake(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        face_image = Image.open(file_path)
        generated_image = self.synthesizer.generate_deepfake(face_image)
        # Display generated image

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepFakeGUI(root)
    root.mainloop()
