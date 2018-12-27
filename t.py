from tkinter import *
import os
root = Tk()
var = StringVar()
var.set(os.getcwd())

l = Label(root, textvariable = var)
l.pack()

t = Entry(root, textvariable = var)
t.pack()

root.mainloop()