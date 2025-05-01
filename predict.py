# predict.py
import torch
from PIL import Image
import numpy as np
from src.model import MNISTNet
from src.utils import load_model
from src.visualizer import OutputVisualizer
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from torchvision import transforms

def load_trained_model(model_path):
    model = MNISTNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)

def predict_digit(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
        probs = torch.exp(output).squeeze()
    return pred, probs.numpy()

class DigitRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition")
        self.window.geometry("1200x800")
        self.window.configure(bg='#2b2b2b')
        
        self.model = self.load_model()
        self.setup_ui()
        
    def load_model(self):
        model_path = 'models/mnist_cnn.pth'
        if not os.path.exists(model_path):
            messagebox.showwarning("Model Not Found", 
                                 "Model not found. Please train first.")
            return None
        return load_trained_model(model_path)
    
    def setup_ui(self):
        button_frame = tk.Frame(self.window, bg='#2b2b2b')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Select Image", command=self.predict_image,
                 bg='#4CAF50', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Train Model", command=self.train_model,
                 bg='#2196F3', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        
        self.result_frame = tk.Frame(self.window, bg='#2b2b2b')
        self.result_frame.pack(pady=20, expand=True, fill=tk.BOTH)
    
    def predict_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        
        if file_path:
            try:
                image_tensor = preprocess_image(file_path)
                prediction, probabilities = predict_digit(self.model, image_tensor)
                
                for widget in self.result_frame.winfo_children():
                    widget.destroy()
                
                fig = OutputVisualizer.create_prediction_display(
                    file_path, prediction, probabilities)
                
                canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def train_model(self):
        response = messagebox.askquestion("Train Model", 
            "This will start training a new model. Continue?")
        if response == 'yes':
            import train
            train.main()
            self.model = self.load_model()
            messagebox.showinfo("Success", "Training completed!")
    
    def run(self):
        self.window.mainloop()

def main():
    app = DigitRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()