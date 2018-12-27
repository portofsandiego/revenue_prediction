from tkinter import Tk, Button, Label, filedialog, BOTH, StringVar
from tkinter.ttk import Frame

class gui(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("Revenue Predictions")
        self.pack(fill=BOTH,expand=1)
        self.centerWindow()
        self.folderButton = Button(self.master,text="Browse", command=browse_button)
        self.folderButton.pack()

    def centerWindow(self):
        w = 290
        h = 150

        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
        self.master.geometry('%dx%d+%d+%d' % (w,h,x,y))

def browse_button():
    global folder_path
    folder_path = StringVar()
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def main():
    root = Tk()
    app = gui()
    root.mainloop()

if __name__ == '__main__':
    main()
