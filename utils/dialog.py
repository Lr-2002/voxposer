# -*- coding: utf-8 -*-
from threading import Thread
import tkinter as tk
from queue import Queue
from tkinter import font
import numpy as np
from PIL import Image, ImageTk
import queue
import speech_recognition as sr


class tkinterUI:
    def __init__(self, agent_id, ai_queue, human_queue, world_observation_queue):
        self.ai_queue = ai_queue
        self.human_queue = human_queue
        self.world_observation_queue = world_observation_queue
        self.update_interval = 1/60
        
        self.root = tk.Tk()
        self.root.title("交互界面")
        text_font = ("song ti", 70)
        
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg="white")
        self.canvas.pack()


        self.output_text = tk.Text(self.root, height=15, width=30, font=text_font)  # window size
        self.output_text.pack()

        self.input_entry = tk.Entry(self.root, width=27, font=text_font)  # window size
        self.input_entry.pack()

        self.submit_button = tk.Button(self.root, text="回车", command=self.handle_user_input, font=text_font)
        self.submit_button.pack()
        self.root.bind('<Return>', self.handle_user_input)

        output_thread = Thread(target=self.display_ai_output)
        output_thread.daemon = True
        output_thread.start()

        self.root.mainloop()
        # self.display_image()



    def handle_user_input(self, event=None):
        user_input = self.input_entry.get()
        output = "User: " + user_input+'\n\n'
        self.human_queue.put(user_input)
        self.output_text.insert(tk.END, output)
        self.output_text.see(tk.END)
        if user_input == "quit":
            self.root.destroy()
        self.input_entry.delete(0, tk.END)


    def display_ai_output(self):
        while True:
            if not self.ai_queue.empty():
                ai_text = self.ai_queue.get()
                output = 'Agent: '+ai_text+'\n\n'
                self.output_text.insert(tk.END, output)
                self.output_text.see(tk.END)
            if not self.world_observation_queue.empty():
                obs = self.world_observation_queue.get().astype(np.uint8)
                img = Image.fromarray(obs)
                img = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                obr = img # hack for solving screen flash
            
                
    
def new_dialog(agent_id, ai_queue, human_queue, world_observation_queue):
    ui = tkinterUI(agent_id, ai_queue, human_queue, world_observation_queue)



def dialog_window(agent_id, ai_queue, human_queue, world_observation_queue):
    # create the main window
    root = tk.Tk()
    root.title("Agent: "+str(agent_id))
    #root.config(encoding='utf-8')
    print(font.families())

    # create a text window
    output_text = tk.Text(root, height=15, width=30)  # window size
    output_text.pack()

    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack()
    # create an input window
    input_entry = tk.Entry(root, width=27)  # window size
    input_entry.pack()

    # text window font and size
    text_font = ("song ti", 70)
    output_text.configure(font=text_font)

    # input window font and size
    input_font = ("song ti", 70)
    input_entry.configure(font=input_font)

    # display the ai_output from the queue
    def display_ai_output():
        while True:
            if not ai_queue.empty():
                ai_text = ai_queue.get()
                output = 'Agent: '+ai_text+'\n\n'
                output_text.insert(tk.END, output)
                output_text.see(tk.END)
        
    
    # handle the user input, print it on the text window and send it to user_queue
    def handle_user_input(event=None):
        user_input = input_entry.get()
        output = "User: " + user_input+'\n\n'
        human_queue.put(user_input)
        output_text.insert(tk.END, output)
        output_text.see(tk.END)
        if user_input == "quit":
            root.destroy()
        input_entry.delete(0, tk.END)
    
    def start_listening():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            text_var.set("请开始说话...")
            root.update()
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio, language="zh-CN")
                text_var.set(f"你说的是: {text}")
                root.update()
            except sr.UnknownValueError:
                text_var.set("语音识别未能理解你说的内容。")
                root.update()
            except sr.RequestError as e:
                text_var.set(f"发生错误: {e}")
                root.update()

    # create an enter button
    button_font = ("song ti", 70)
    submit_button = tk.Button(root, text="输入", command=handle_user_input, font=button_font)
    submit_button.pack()

    say_button = tk.Button(root, text="开始语音输入", command=start_listening)
    say_button.pack()

    text_var = tk.StringVar()
    result_label = tk.Label(root, textvariable=text_var, font=("Arial", 12))
    result_label.pack(pady=20)

    #non-blocking displaying ai-output
    output_thread = Thread(target=display_ai_output)
    output_thread.daemon = True
    output_thread.start()
    
    # start the Tkinter main loop
    root.bind('<Return>', handle_user_input)
    root.mainloop()