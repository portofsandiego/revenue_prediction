from tkinter import Tk, Button, Label, filedialog, Entry, StringVar, BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter.ttk import Frame
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
        self.pack(fill=BOTH,expand=1)
        self.centerWindow()

        self.folder_path = StringVar()
        self.folder_path.set(os.getcwd())

        # Current Folder Label
        self.m = Label(self.master, text="Directory: ")
        self.m.pack(side=TOP)

        self.l = Label(self.master, textvariable = self.folder_path)
        self.l.pack(side=TOP)

        # Path Input
        self.t = Entry(self.master, width=35, textvariable = self.folder_path)
        self.t.pack()

        # Predict
        self.predictButton = Button(self.master,text="Predict", command=self.predict)
        self.predictButton.pack(side=RIGHT)

        # Browse Folder
        self.folderButton = Button(self.master,text="Browse", command=self.browse_button)
        self.folderButton.pack(side=RIGHT)

        # Quit Program
        self.closeButton = Button(self.master, text="Quit", command=self.master.quit)
        self.closeButton.pack(side=LEFT)

    def centerWindow(self):
        w = 475
        h = 100

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