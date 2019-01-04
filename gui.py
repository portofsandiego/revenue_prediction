import tkinter as tk
from tkinter import Tk, Button, Label, filedialog, Entry, StringVar
from tkinter.ttk import Frame
from PIL import Image, ImageTk
from data_transformer import reformat
from forecaster import HoltWinters
from sklearn.metrics import mean_absolute_error
import numpy as np
import os

def mean_absolute_scaled_error():
    pass

class gui(Frame):

    def __init__(self):
        super().__init__()

        self.master.title("Revenue Predictions")
        self.pack(fill=tk.BOTH,expand=1)
        self.centerWindow()

        self.folder_path = StringVar()
        self.folder_path.set(os.getcwd())

        # Top Frame 
        self.tfm = Frame(self.master)
        # Logo
        self.img = Image.open("posdlogo.png")
        self.img = self.img.resize((40,40),Image.ANTIALIAS)
        self.pic = ImageTk.PhotoImage(self.img)
        self.label = Label(self.tfm, image = self.pic)
        self.label.pack(side=tk.LEFT)
        # Directory Label
        self.m = Label(self.tfm, text="Select Input Data")
        self.m.pack(side=tk.BOTTOM)
        # Pack Top Frame
        self.tfm.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Center Frame
        self.cfm = Frame(self.master)
        # Path Input
        self.t = Entry(self.cfm, width=35, textvariable = self.folder_path)
        self.t.pack(side=tk.LEFT)
        # Browse Folder
        self.cbf = Frame(self.cfm, height=23,width=23)
        self.cbf.pack_propagate(0)
        self.cbf.pack()
        self.folderButton = Button(self.cbf, text="...", command=self.browse_button)
        self.folderButton.pack(side=tk.RIGHT)
        # Pack Center Frame
        self.cfm.pack(anchor=tk.CENTER,padx=5, pady=5)

        # Bottom Frame
        self.bfm = Frame(self.master)
        # Quit Program
        self.closeButton = Button(self.bfm, text="Quit", command=self.master.quit)
        self.closeButton.pack(side=tk.LEFT)
        # Predict
        self.predictButton = Button(self.bfm,text="Predict", command=self.predict)
        self.predictButton.pack(side=tk.RIGHT)
        # Pack Bottom Frame
        self.bfm.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def centerWindow(self):
        w = 425
        h = 120

        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.master.geometry('%dx%d+%d+%d' % (w,h,x,y))

    def browse_button(self):
        self.folder_path.set(filedialog.askopenfilename(initialdir='',title="Select File", filetypes=(("Data Files","*.csv"),("Hyperparameters","*.npy"),("All Files","*.*"))))
        print(self.folder_path.get())

    def predict(self):
        # Format the data
        data = reformat(self.folder_path.get())
        data = data.Revenue[:]
        model = HoltWinters(data, 3, n_preds=12)
        model.train()
        predictions = model.predict()
        print(np.around(predictions, decimals=2))
        # If file exists, read in data
        

        # Input data sheet and predict

def main():
    root = Tk()
    gui()
    root.mainloop()

if __name__ == '__main__':
    main()