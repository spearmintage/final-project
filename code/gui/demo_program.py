from tkinter import *
from tkinter import ttk

window = Tk()



fileUploadFrame = ttk.Frame(window)
btn = ttk.Button(fileUploadFrame)

fileUploadFrame.grid(column=0, row=0, padx=5, pady=5)

fileUploadFrame["width"] = 800
fileUploadFrame["height"] = 250
fileUploadFrame["padding"] = 5
fileUploadFrame["borderwidth"] = 2

btn["text"] = "nuts"

window.mainloop()