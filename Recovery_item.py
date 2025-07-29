import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import os

tools = {}
matched_tools = set()
lost_tools = {}
running = False
selected_video_path = None

def log_position(label, x, y):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("object_log.txt", "a") as f:
        f.write(f"{timestamp}, LOST {label} at ({x},{y})\n")

def match_tool(frame, label, ref_img, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Morphological filtering
    kernel = np.ones((5, 5), np.uint8)
    gray_masked = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    _, thresh = cv2.threshold(gray_masked, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    if not valid:
        print(f"{label}: No large contours found")
        return False, frame

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_masked, None)

    if des1 is None or des2 is None or len(kp2) < 10:
        print(f"{label}: Not enough features")
        return False, frame

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"{label}: Good matches = {len(good)}")

    if len(good) > 12:
        cv2.putText(frame, f"{label.upper()} Matched", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return True, frame
    return False, frame

def update_lost_box(box):
    box.delete(0, tk.END)
    for tool in lost_tools:
        box.insert(tk.END, tool)

def video_loop(label_widget, matched_box, lost_box, status_label):
    global running, selected_video_path
    if not selected_video_path:
        messagebox.showwarning("No Video", "Please select a video file first.")
        return
    if not tools:
        messagebox.showwarning("No Tools", "Please load reference tools first.")
        return

    cap = cv2.VideoCapture(selected_video_path)
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        current_visible = []

        for label, (ref_img, hsv_l, hsv_u) in tools.items():
            found, frame = match_tool(frame, label, ref_img, hsv_l, hsv_u)
            if found:
                current_visible.append(label)
                if label not in matched_tools:
                    matched_tools.add(label)
                    matched_box.insert(tk.END, label)
                    if label in lost_tools:
                        del lost_tools[label]
                        update_lost_box(lost_box)
            else:
                if label in matched_tools and label not in lost_tools:
                    lost_tools[label] = (time.strftime("%H:%M:%S"), (0, 0))
                    log_position(label, 0, 0)
                    update_lost_box(lost_box)

        status_text = f"Currently Visible: {', '.join(current_visible) if current_visible else 'None'}"
        status_label.config(text=status_text, fg="blue")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.config(image=imgtk)
        label_widget.update()

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()

def start_detection(label, matched_box, lost_box, status_label, start_btn):
    global selected_video_path
    if not selected_video_path:
        messagebox.showerror("Missing Video", "Please select a video file first!")
        return
    if not tools:
        messagebox.showerror("Missing Tools", "Please load at least one reference tool!")
        return

    start_btn.config(state=tk.DISABLED)
    threading.Thread(target=video_loop, args=(label, matched_box, lost_box, status_label), daemon=True).start()

def load_tool(label_entry, hsv_low_entry, hsv_high_entry):
    label = label_entry.get().strip()
    lower_str = hsv_low_entry.get().strip()
    upper_str = hsv_high_entry.get().strip()

    if not label or not lower_str or not upper_str:
        messagebox.showerror("Error", "Please enter label and HSV bounds!")
        return

    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
    if not path:
        return

    img = cv2.imread(path)
    try:
        lower = np.array([int(x.strip()) for x in lower_str.split(',')])
        upper = np.array([int(x.strip()) for x in upper_str.split(',')])
        if lower.shape[0] != 3 or upper.shape[0] != 3:
            raise ValueError
    except:
        messagebox.showerror("Invalid HSV", "HSV values must be 3 comma-separated integers.")
        return

    tools[label.upper()] = (img, lower, upper)
    messagebox.showinfo("Success", f"Tool '{label}' loaded successfully.")

def select_video_file(video_label_var):
    global selected_video_path
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if path:
        selected_video_path = path
        video_label_var.set(f"Selected: {os.path.basename(path)}")

def stop_app(root):
    global running
    running = False
    root.quit()
    root.destroy()

def start_gui():
    root = tk.Tk()
    root.title("Lost Item Recovery System – Multi Tool + HSV")
    root.geometry("1000x800")

    frame_top = tk.Frame(root)
    frame_top.pack()

    tk.Label(frame_top, text="Tool Label").grid(row=0, column=0)
    label_entry = tk.Entry(frame_top)
    label_entry.grid(row=0, column=1)

    tk.Label(frame_top, text="HSV Lower (H,S,V)").grid(row=1, column=0)
    hsv_low = tk.Entry(frame_top)
    hsv_low.grid(row=1, column=1)

    tk.Label(frame_top, text="HSV Upper (H,S,V)").grid(row=2, column=0)
    hsv_high = tk.Entry(frame_top)
    hsv_high.grid(row=2, column=1)

    tk.Button(frame_top, text="Load Reference Tool",
              command=lambda: load_tool(label_entry, hsv_low, hsv_high)).grid(row=0, column=2, rowspan=3, padx=10)

    video_select_frame = tk.Frame(root)
    video_select_frame.pack(pady=5)

    video_path_label_var = tk.StringVar()
    video_path_label_var.set("No video selected")

    tk.Button(video_select_frame, text="Select Video File",
              command=lambda: select_video_file(video_path_label_var)).grid(row=0, column=0, padx=10)

    tk.Label(video_select_frame, textvariable=video_path_label_var, fg="green").grid(row=0, column=1)

    video_label = tk.Label(root)
    video_label.pack(pady=10)

    status_label = tk.Label(root, text="Currently Visible: None", fg="blue")
    status_label.pack()

    list_frame = tk.Frame(root)
    list_frame.pack()

    tk.Label(list_frame, text="Matched Tools").grid(row=0, column=0)
    tk.Label(list_frame, text="Lost Tools").grid(row=0, column=1)

    matched_box = tk.Listbox(list_frame, width=40, height=10)
    matched_box.grid(row=1, column=0, padx=10)
    lost_box = tk.Listbox(list_frame, width=40, height=10)
    lost_box.grid(row=1, column=1, padx=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    start_btn = tk.Button(btn_frame, text="Start Detection",
                          command=lambda: start_detection(video_label, matched_box, lost_box, status_label, start_btn))
    start_btn.grid(row=0, column=0, padx=10)

    tk.Button(btn_frame, text="Exit", command=lambda: stop_app(root)).grid(row=0, column=1)

    root.mainloop()

if __name__ == "__main__":
    start_gui()

