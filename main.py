import tkinter as tk
from gui import DeepFakeGUI

def main():
    root = tk.Tk()
    app = DeepFakeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
