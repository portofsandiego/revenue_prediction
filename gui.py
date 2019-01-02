from tkinter import Tk, Button, Label, filedialog, StringVar, BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter.ttk import Frame
from data_transformer import reformat
from forecaster import HoltWinters
import os

class gui(Frame):

    def __init__(self):
        super().__init__()
        folder_path = os.getcwd()
        Label(self.master,text=folder_path).pack(side = TOP)
        self.initUI()

    def initUI(self):
        self.master.title("Revenue Predictions")
        self.pack(fill=BOTH,expand=1)
        self.centerWindow()

        #Predict
        self.predictButton = Button(self.master,text="Predict", command=predict)
        self.predictButton.pack(side=RIGHT)

        #Browse Folder
        self.folderButton = Button(self.master,text="Browse", command=browse_button)
        self.folderButton.pack(side=RIGHT)

        #Quit Program
        self.closeButton = Button(self.master, text="Quit", command=self.master.quit)
        self.closeButton.pack(side=LEFT)

    def centerWindow(self):
        w = 600
        h = 200

        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.master.geometry('%dx%d+%d+%d' % (w,h,x,y))

def browse_button():
    global folder_path
    folder_path = StringVar()
    filename = filedialog.askopenfilename()
    folder_path = filename
    print(filename)

def predict():
    #Format the data
    reformat()

    #Pull data sheet and predict

def main():
    root = Tk()
    gui()
    root.mainloop()

if __name__ == '__main__':
    main()