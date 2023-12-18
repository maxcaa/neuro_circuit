import tkinter as tk
from tkinter import filedialog, Text
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.backends.backend_tkagg as tkagg
from mat73 import loadmat
import numpy as np
from scipy.signal import find_peaks, stft
import pandas as pd
import datetime

def extract_continuous_trace(data, Mouse, Cell_Counter, Trial_Counter, key='Trial_MembranePotential'):
    """Extracts membrane potential from data structure

    Parameters
    ----------
    data : dict
        Dictionary with the imported datastructure
    Mouse : str
        Name of the mouse we want to process
    Cell_Counter : int
        Count of neurons per mouse
    Trial_Counter : int
        Trial number that we want to process

    Returns
    -------
    selected_trial_vm : np.ndarray
        membrane potential for selected mouse, cell, trial
    """
    # Find correct trial with mouse, cell, and trial constraint.
    # This line is a little bit weird because of the types and how the vectors
    #     are represented.
    selected_trial = (Mouse == data['Mouse_Name']).T & (data['Cell_Counter'] == Cell_Counter) & (data['Trial_Counter'] == Trial_Counter)
    # extract the membrane potential
    continuous_trace = data[key][np.where(selected_trial)[1][0]][0]

    return continuous_trace

def extract_continuous_trace_time(data, Mouse, Cell_Counter, Trial_Counter, key='Trial_MembranePotential'):
    """Extracts membrane potential from data structure

    Parameters
    ----------
    data : dict
        Dictionary with the imported datastructure
    Mouse : str
        Name of the mouse we want to process
    Cell_Counter : int
        Count of neurons per mouse
    Trial_Counter : int
        Trial number that we want to process

    Returns
    -------
    selected_trial_vm : np.ndarray
        membrane potential for selected mouse, cell, trial
    """
    # Find correct trial with mouse, cell, and trial constraint.
    # This line is a little bit weird because of the types and how the vectors
    #     are represented.
    selected_trial = ((Mouse == data['Mouse_Name']).T & (data['Cell_Counter'] == Cell_Counter) & (data['Trial_Counter'] == Trial_Counter)).flatten()
    # extract the membrane potential
    times = data[key][np.where(selected_trial)[0][0]][0]
    start_time = 0
    sr = data['Trial_WhiskerAngle_C2_right_SamplingRate'][selected_trial][0]
    end_time = data['Trial_WhiskerAngle_C2_right'][selected_trial][0][0].shape[0] / sr
    samples = int((end_time - start_time) * sr)
    continuous_trace = np.zeros(samples)

    if type(times) == np.ndarray:
        for time in times:
            continuous_trace[int(time[0]*sr):int(time[1]*sr)] = 1

    return continuous_trace


class Trial_Viewer_Gui():

    def __init__(self, master, data):
        self.master = master
        self.data = data

        self.instructions = ("           Welcome to the python Dataviewer!\n"
            "The instructions are simple just choose first the mouse name you want\n"
            "please do so in that order. When you choose the trial then you will get\n"
            "to inspect then the cell from this mouse and finally the trial,\n"
            "informations about this specific mouse, cell, trial combination and then\n"
            "you can press the button in order to plot the membrane potential the whisker\n"
            "angle and the state of the whisking.")

        self.mouse_names = np.unique(data['Mouse_Name'])
        self.trials = np.unique(data['Trial_Counter'])
        self.cells = np.unique(data['Cell_Counter'])

        self.plot_frame_build()
        self.dropdown()



    def dropdown(self):
        self.frame_popup = tk.Frame(self.master, bg="white")
        self.frame_popup.pack(fill='both', side='left', expand=True)

        self.txt_instructions = Text(self.frame_popup, state=tk.DISABLED)
        self.txt_instructions.pack()
        self.txt_instructions.configure(state=tk.NORMAL)
        self.txt_instructions.insert(tk.INSERT, self.instructions)
        self.txt_instructions.configure(state=tk.DISABLED)

        self.mouse_name = tk.StringVar(self.master)
        self.mouse_name.set('TK355')
        self.mouse_name.trace('w', self.updatecell)

        self.cell_counter = tk.StringVar(self.master)
        self.cell_counter.set('1.0')
        self.cell_counter.trace('w', self.updatetrial)

        self.trial_counter = tk.StringVar(self.master)
        self.trial_counter.set('1.0')
        self.trial_counter.trace('w', self.print_information)

        self.popupMenuMouse = tk.OptionMenu(self.frame_popup, self.mouse_name, *self.mouse_names)
        tk.Label(self.frame_popup, text="Choose a mouse", bg='white').pack()
        self.popupMenuMouse.pack()

        self.popupMenuCells = tk.OptionMenu(self.frame_popup, self.cell_counter, '')
        tk.Label(self.frame_popup, text="Choose a cell", bg='white').pack()
        self.popupMenuCells.pack()

        self.popupMenuTrials = tk.OptionMenu(self.frame_popup, self.trial_counter, '')
        tk.Label(self.frame_popup, text="Choose a trial", bg='white').pack()
        self.popupMenuTrials.pack()

        self.plot_button = tk.Button(self.frame_popup, text="Plot the membrane potential", command=self.plot_signals)
        self.plot_button.pack()

        self.txt = Text(self.frame_popup, state=tk.DISABLED)
        self.txt.pack()

    def print_information(self, *args):
        # import ipdb;ipdb.set_trace()
        mouse_name = str(self.mouse_name.get())
        mouse_Genotype = np.unique(data['Mouse_Genotype'][mouse_name==data["Mouse_Name"]])
        cell_number = str(self.cell_counter.get())
        cell_td_tomato = "tdTomato+" if data['Cell_tdTomatoExpressing'][((mouse_name==data["Mouse_Name"]).flatten()) & (float(cell_number) == data["Cell_Counter"])][0] else "tdTomato+"
        cell_depth = str(data['Cell_Depth'][((mouse_name==data["Mouse_Name"]).flatten()) & (float(cell_number) == data["Cell_Counter"])][0])
        cell_type = str(data['Cell_Type'][((mouse_name==data["Mouse_Name"]).flatten()) & (float(cell_number) == data["Cell_Counter"])][0][0])
        trial_number = str(self.trial_counter.get())
        trial_type = str(data['Trial_Type'][((mouse_name==data["Mouse_Name"]).flatten()) & (float(cell_number) == data["Cell_Counter"]) & (float(trial_number)==data['Trial_Counter'])][0][0])
        trial_start = data['Trial_StartTime'][((mouse_name==data["Mouse_Name"]).flatten()) & (float(cell_number) == data["Cell_Counter"]) & (float(trial_number)==data['Trial_Counter'])]
        trial_start = trial_start[0].astype(int)
        trial_start = datetime.datetime(trial_start[0], trial_start[1], trial_start[2], trial_start[3], trial_start[4], trial_start[5]).strftime("%d %b %Y %H:%M:%S")

        text = ("Mouse Information \nMouse Name    \t{} \nMouse Genotype\t{}\n\t\t{} \n\n"
                "Cell Number   \t{} \nCell tdTomato \t{} \nCell Depth    \t{} \nCell Type      {} \n\nTrial Number  \t{} \n"
                "Trial Type    \t{} \nTrial Start   \t{}").format(mouse_name, mouse_Genotype[0],
                mouse_Genotype[1], cell_number, cell_td_tomato, cell_depth, cell_type, trial_number, trial_type, trial_start)
        self.txt.configure(state=tk.NORMAL)
        self.txt.delete('1.0', tk.END)
        self.txt.insert(tk.INSERT, text)
        self.txt.configure(state=tk.DISABLED)


    def updatetrial(self, *args):
        trials = data['Trial_Counter'][(data['Mouse_Name'] == self.mouse_name.get()).flatten()]
        menu = self.popupMenuTrials['menu']
        menu.delete(0,'end')
        for trial in trials:
            menu.add_command(label=trial, command=lambda x=trial: self.trial_counter.set(x))

    def updatecell(self, *args):
        cells = self.data['Cell_Counter'][(data['Mouse_Name'] == self.mouse_name.get()).flatten()]
        menu = self.popupMenuCells['menu']
        menu.delete(0,'end')
        for cell in cells:
            menu.add_command(label=cell, command=lambda x=cell: self.cell_counter.set(x))


    def plot_frame_build(self):
        self.plot_frame = tk.Frame(self.master, bg="white")
        self.plot_frame.pack(fill='both', side='right', expand=True)

        self.figure = plt.Figure(figsize=(12,10))
        self.chart_type = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.chart_type.get_tk_widget().pack(expand=True, fill='both')
        tkagg.NavigationToolbar2Tk(self.chart_type, self.plot_frame)


    def plot_signals(self, keys=['Trial_MembranePotential', 'Trial_WhiskerAngle_C2_right', 'Trial_WhiskingTime', 'Trial_QuietTime', 'Trial_ContactTime']):

        mouse_name = str(self.mouse_name.get())
        cell = str(self.cell_counter.get())
        trial = str(self.trial_counter.get())
        self.figure.clf()
        axes = self.figure.subplots(len(keys), 1,sharex=True, gridspec_kw={'height_ratios': [2,1.5,1,1,1]})

        for i, key in enumerate(keys):
            if 'Membrane' in key:
                sr = data['Trial_MembranePotential_SamplingRate'][0]
            else:
                sr = data['Trial_WhiskerAngle_C2_right_SamplingRate'][0]
            axes[i].clear()
            if 'Time' in key:
                signal = extract_continuous_trace_time(data, mouse_name, int(float(cell)), int(float(trial)), key=key)
            else:
                signal = extract_continuous_trace(data, mouse_name, int(float(cell)), int(float(trial)), key=key)

            time = np.arange(signal.shape[0]) / sr
            axes[i].plot(time, signal)
            axes[i].set_ylabel(key.split('_')[1])
        axes[i].set_xlabel("Time (seconds)")
        axes[0].set_title('Mouse {} Cell {} Trial {}'.format(mouse_name, trial, cell))
        self.figure.subplots_adjust(left=0.07, bottom=0.05, right=0.98, top=0.95)

# Load data and transfer them in numpy arrays
data = loadmat(os.path.join('MiniProjectData.mat'))
data_descr = data['data_description']
data = data['data']
data = {key: np.array(data[key]) for key in data}

# start the gui
root = tk.Tk()
mygui = Trial_Viewer_Gui(root, data)
root.mainloop()