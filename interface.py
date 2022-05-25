import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk


# this is where the predicted letter should be stored
predicted = 'N/A'
# TODO: add call back function for taking a screenshot from the webcam
screenshot_path = ''


# button commands
 # when test button is pressed this function will execute
def test():
    # check if file path has been selected and screenshot hasn't been taken
    if len(browse) != 0 and len(screenshot_path) == 0:
        # execute back propagation
        # path to test image -> browse()
        pass
    elif len(browse) == 0 and len(screenshot_path) != 0:
        # execute back propagation
        # path to test image -> screenshot_path
        pass  
    
    print('Test')
def screenshot():
    # when screenshot button is pressed this function will execute
    print('Webcam')

# call back function when window is closed
# for releasing webcam stream and destroying window object
def destructor():
    window.destroy()
    cap.release()

# webcam stream, 0 is the default camera
cap = cv2.VideoCapture(0)

# window
window = tk.Tk()
window.config(padx=10, pady=10)

# canvas setup
# width and height is based on the frame size
webcam = tk.Canvas(window, width=400, height=290)
# positioning canvas on the grid
webcam.grid(column=0, row=0, padx=5, pady=5)

# container for grouping path_entry and browse_btn widgets
path_entry_frame = ttk.Frame(window)
path_entry_frame.grid(column=0, row=1)
path_entry_frame.configure()

# initializing entry widget

path_entry = ttk.Entry(path_entry_frame, text='no image selected', width=54)

# positioniong entry widget on the grid
path_entry.grid(column=0, row=1)

# container for grouping output_box and output_label
output_frame = ttk.Frame(window, width=200, height=200)
output_frame.grid(column=1, row=0, columnspan=1, rowspan=4, padx=5)

# initializing output_box and output_label
output_box = ttk.Label(output_frame, text=f'{predicted}', justify=tk.CENTER)
output_label = ttk.Label(output_frame, text='Predicted Letter')

# positioning output_box and output_label on the grid
output_box.grid(column=0, row=1, pady=5)
output_label.grid(column=0, row=0)

# customizing output box
output_box.config(state=tk.DISABLED, relief='solid', font=('Calibri',40), justify=tk.CENTER, padding=50, background='light yellow')

# for selecting an image, returns file path
def browse():
    path = filedialog.askopenfilename(initialdir='/', title='select an image', filetypes=(('image', '*.jpg'),('all', '*.*')))
    path_entry.delete(0, END)
    path_entry.insert(INSERT, path)
    test_btn['state'] = NORMAL
    return path

# initializing buttons
test_btn = ttk.Button(output_frame, width=30, text='Test', command=test, state=tk.DISABLED)
browse_btn = ttk.Button(path_entry_frame, width=10, text='Browse', command=browse)
screenshot_btn = ttk.Button(output_frame, width=30, text='Screenshot', command=screenshot)
exit_btn = ttk.Button(output_frame, width=30, text='Exit', command=destructor)


# button positioning
test_btn.grid(column=0, row=5)
browse_btn.grid(column=1, row=1)
screenshot_btn.grid(column=0, row=3)
exit_btn.grid(column=0, row=6)

def update():
    # reads stream and returns true if read, and the frame data
    ret, frame = cap.read()
    # frame returned
    if ret:
        # covert the frame from bgr to rgba
        convert_to_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # convert from array to image and then to PhotoImage object for tkinter
        photo = ImageTk.PhotoImage(image=Image.fromarray(convert_to_rgba))
        # anchor the image (otherwise gets deleted by garbage collection)
        webcam.photo = photo
        # display the image on the canvas
        webcam.create_image(0, 0, image=photo, anchor='nw')
    # updates the canvas with a new frame every 30 miliseconds
    window.after(30, update)

# once the porotocol for closing window is initiated a call back function is called
window.protocol('WM_DELETE_WINDOW', destructor)

update()
window.resizable(False, False)
window.mainloop()
