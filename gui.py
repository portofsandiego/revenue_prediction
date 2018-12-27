from tkinter import Tk, Button, Label, filedialog, StringVar, BOTH, BOTTOM, LEFT, RIGHT, TOP
from tkinter.ttk import Frame
import os

class gui(Frame):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("Revenue Predictions")
        self.pack(fill=BOTH,expand=1)
        self.centerWindow()

        #Browse Folder
        self.folderButton = Button(self.master,text="Browse", command=browse_button)
        self.folderButton.pack(side=RIGHT)

        #Predict
        self.predictButton = Button(self.master,text="Predict")
        self.predictButton.pack(side=RIGHT)

        #Quit Program
        self.closeButton = Button(self.master, text="Quit", command=self.master.quit)
        self.closeButton.pack(side=LEFT)

    def centerWindow(self):
        w = 300
        h = 100

        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.master.geometry('%dx%d+%d+%d' % (w,h,x,y))

def browse_button():
    folder_path = StringVar()
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def main():
    root = Tk()
    folder_path = StringVar()
    folder_path.set(os.getcwd())
    gui()
    Label(master=root,text=folder_path).pack(side = TOP)
    root.mainloop()

if __name__ == '__main__':
    main()
