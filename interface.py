import tkinter as tk
import cv2
from PIL import Image, ImageTk

# webcam stream, 0 is the default camera
cap = cv2.VideoCapture(0)

# window
window = tk.Tk()

# canvas setup
# width and height is based on the frame size
webcam = tk.Canvas(window, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# positioning canvas on the grid
webcam.grid(column=0, row=0, padx=10, pady=10)

def update():
    # reads stream and returns true if read, and the frame data
    ret, frame = cap.read()
    # frame returned
    if ret:
        # covert the frame from bgr to rgba
        convert_to_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # convert from array to image and to PhotoImage object for tkinter
        photo = ImageTk.PhotoImage(image=Image.fromarray(convert_to_rgba))
        # anchor the image (otherwise gets deleted by garbage collection)
        webcam.photo = photo
        # display the image on the canvas
        webcam.create_image(0, 0, image=photo, anchor='nw')

    # updates the canvas with a new frame every 30 miliseconds
    window.after(30, update)

# call back function when window is closed
# release camera and destroys window object
def destructor():
    window.destroy()
    cap.release()

# once the porotocol for closing window is initiated a call back function is called
window.protocol('WM_DELETE_WINDOW', destructor)

update()
window.resizable(False, False)
window.mainloop()