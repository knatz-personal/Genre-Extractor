"""This module contains user interface components
"""
import configparser
import matplotlib
import numpy as np
import os
import threading
import tkinter as tk

from Utilities import *
from builtins import *
from collections import OrderedDict
from pandas_ml.confusion_matrix.cm import ConfusionMatrix
from sklearn import preprocessing
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import classification_report
from tkinter import *
from tkinter.ttk import *
import webbrowser
from _operator import itemgetter
import matplotlib
#  For embedding plot in tkinter frame
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Classification import ClassifierType
from Classification import GenreClassifier
from CustomControls import ConsoleWidget
from CustomControls import StatusBar
from PIL.ImageTk import Image
from PIL.ImageTk import PhotoImage
from idlelib.ToolTip import ToolTip
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from tkFontChooser import FontChooser
from tkFontChooser import askfont
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

_author__ = "Nathan Zwelibanzi Khupe"
__copyright__ = "Copyright 2017, Nathan Zwelibanzi Khupe"
__credits__ = ["Nathan Zwelibanzi Khupe"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nathan Zwelibanzi Khupe"
__email__ = "001knatz@gmail.com"
__docformat__ = "numpy"

class MainForm(object):
    """The standard GUI screen on which other elements are added  
    """
    corpusForm = None
    mainGenreForm = None
    subGenreForm = None

    def __init__(self, master: Tk, path, **kwargs):
        """Constructor for the Main Form

        Parameters
        ----------
        master : Tk
            tkinter Tk object
        path : str
            The path to the icon file
        **kwargs
            arguments
        """
        self._root = master
        self.directory = path
        master.iconbitmap(True, os.path.join(path, "Images/favicon.ico"))
        self._initGUI()
        self._configUtil = ConfigUtils(configparser.ConfigParser())
        self._configUtil.load()
        self._load_settings()
        self.filename = ""
        return super().__init__(**kwargs)

    def _initGUI(self):
        self._root.title("Fantasy Genre Extractor")
        self._center()
        self._root.withdraw()

        self._make_menubar()
        self._make_toolbar()

        self.statebar = StatusBar(self._root)
        self.statebar.set('{} |'.format('Status Bar'))
        self.statebar.label.config(font=('Calibri', 12, 'normal'))
        self.statebar.pack(side=BOTTOM, fill=X, expand=NO)

        lframe = Frame(self._root, width=500, height=600)
        lframe.pack_propagate(False)
        self._make_text_area(lframe)
        lframe.pack(side=LEFT, fill=BOTH, expand=YES)

        fstyle = Style()
        fstyle.configure('GE.TFrame', background='#334353', foreground="#fff")
        rframe = Frame(self._root, width=300, height=600, style='GE.TFrame')
        rframe.pack_propagate(False)

        self._make_control_area(rframe)
        rframe.pack(side=LEFT, fill=BOTH, expand=YES)
        self._root.deiconify()
        pass

    def _make_toolbar(self):
        container = Frame(self._root)
        self._make_tools(container)
        self._make_search_box(container)
        container.pack(side=TOP, fill=X)
        pass

    def _make_tools(self, parent):
        toolbar = Frame(parent)

        new_img = Image.open(os.path.join(self.directory, "Images/new.png"))
        open_img = Image.open(os.path.join(self.directory, "Images/open.png"))
        save_img = Image.open(os.path.join(self.directory, "Images/save.png"))
        saveas_img = Image.open(os.path.join(self.directory, "Images/saveas.png"))
        cut_img = Image.open(os.path.join(self.directory, "Images/cut.png"))
        copy_img = Image.open(os.path.join(self.directory, "Images/copy.png"))
        paste_img = Image.open(os.path.join(self.directory, "Images/paste.png"))

        # Create a TkInter image to be used in the button
        new_icon = PhotoImage(new_img)
        open_icon = PhotoImage(open_img)
        save_icon = PhotoImage(save_img)
        saveas_icon = PhotoImage(saveas_img)
        cut_icon = PhotoImage(cut_img)
        copy_icon = PhotoImage(copy_img)
        paste_icon = PhotoImage(paste_img)

        # Create buttons for the toolbar
        new_button = Button(toolbar, image=new_icon, command=self.new_file)
        open_button = Button(toolbar, image=open_icon, command=self.open_file)
        save_button = Button(toolbar, image=save_icon, command=self.save_file)
        saveas_button = Button(toolbar, image=saveas_icon,
                               command=self.save_as_file)
        cut_button = Button(toolbar, image=cut_icon, command=self.onCut)
        copy_button = Button(toolbar, image=copy_icon, command=self.onCopy)
        paste_button = Button(toolbar, image=paste_icon, command=self.onPaste)

        ToolTip(new_button, ["New", "Text"])
        ToolTip(open_button, ["Open", "File"])
        ToolTip(save_button, ["Save", "File"])
        ToolTip(saveas_button, ["Save", "As"])
        ToolTip(cut_button, ["Cut", "Text"])
        ToolTip(copy_button, ["Copy", "Text"])
        ToolTip(paste_button, ["Paste", "Text"])

        new_button.image = new_icon
        open_button.image = open_icon
        save_button.image = save_icon
        saveas_button.image = saveas_icon
        cut_button.image = cut_icon
        copy_button.image = copy_icon
        paste_button.image = paste_icon

        # Place buttons in the interface
        px = 1
        py = 1
        new_button.pack(side=LEFT, padx=px, pady=py)
        open_button.pack(side=LEFT, padx=px, pady=py)
        save_button.pack(side=LEFT, padx=px, pady=py)
        saveas_button.pack(side=LEFT, padx=px, pady=py)
        cut_button.pack(side=LEFT, padx=px, pady=py)
        copy_button.pack(side=LEFT, padx=px, pady=py)
        paste_button.pack(side=LEFT, padx=px, pady=py)

        toolbar.pack(side=LEFT, fill=X)
        pass

    def _make_search_box(self, parent):
        searchFrm = Frame(parent,  width=25)

        px = 1
        py = 1

        self.find_text = Entry(searchFrm)
        self.find_text.config(font=('Calibri', 12, 'normal'))
        self.find_text.insert(0, 'find')
        self.find_text.bind('<FocusIn>', self.onEntryFocusIn)
        self.find_text.bind('<FocusOut>', lambda event:
                            self.onEntryFocusOut(event=event, message='find'))

        find_img = Image.open(os.path.join(self.directory, "Images/find.png"))
        clear_img = Image.open(os.path.join(self.directory, "Images/clear.png"))

        find_icon = PhotoImage(find_img)
        clear_icon = PhotoImage(clear_img)

        find_button = Button(searchFrm, image=find_icon, command=self.onFind)
        clear_button = Button(searchFrm, image=clear_icon,
                              command=self.onClearText)

        ToolTip(clear_button, ["Clear", "Found", "Text"])

        find_button.image = find_icon
        clear_button.image = clear_icon

        ToolTip(find_button, ["Find", "Text"])

        self.find_text.pack(side=LEFT, fill=Y, padx=0, pady=3)
        find_button.pack(side=LEFT, padx=0, pady=1)
        clear_button.pack(side=LEFT, padx=px, pady=py)
        searchFrm.pack(side=LEFT)
        pass

    def _make_control_area(self, frame: Frame):
        """Summary

        Parameters
        ----------
        frame : Frame
            Description

        Returns
        -------
        TYPE
            Description
        """
        msglblstyle = Style()
        msglblstyle.configure('GE.TLabel', background='#fff', foreground="#000")
        self.msglbl = Label(frame, text="Hi, Reader", style='GE.TLabel')
        self.msglbl.pack(side=TOP, fill=BOTH, expand=YES)

        btnframe = Frame(frame)
        extractbtn = tk.Button(btnframe, text="Extract Sub-Genres", relief=FLAT,
                                    foreground="#fff", background="#639af2",
                                    activebackground="#14ff33", command=self.onExtractButtonClick)
        extractbtn.pack(side=LEFT, padx=4, pady=4)

        self.spinner = Progressbar(btnframe, orient="horizontal",
                                   length=200, mode="indeterminate")
        self.spinner.pack(side=RIGHT, fill=X, expand=True, padx=5)
        btnframe.pack(side=TOP, fill=BOTH, expand=True)

        note_area = Notebook(frame)

        page1 = self._make_plot_area(note_area)
        page2 = self._make_table_tab(note_area)

        note_area.add(page1, text='Chart')
        note_area.add(page2, text='Table')

        note_area.pack(side=TOP, fill=BOTH, expand=TRUE)
        pass

    def _make_table_tab(self, parentfrm):
        """Summary

        Parameters
        ----------
        parentfrm : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        tableFrame = tk.Frame(parentfrm)
        tableScroll = Scrollbar(tableFrame)

        self.GenreTable = Treeview(tableFrame)
        self.GenreTable['columns'] = ('Genre', 'Probability')

        self.GenreTable.heading("#0", text='#', anchor='center')
        self.GenreTable.heading('Genre', text='Genre', anchor='w')
        self.GenreTable.heading('Probability', text='Probability', anchor='e')

        self.GenreTable.column('#0', anchor='center', width=20)
        self.GenreTable.column('Genre', anchor='w', width=50)
        self.GenreTable.column('Probability', anchor='e', width=50)

        tableScroll.configure(command=self.GenreTable.yview)
        tableScroll.pack(side="right", fill=Y)

        self.GenreTable.configure(yscrollcommand=tableScroll.set)
        self.GenreTable.pack(side="left", fill=BOTH, expand=True)
        tableFrame.pack(side=TOP, fill=BOTH, expand=True)
        return tableFrame

    def _make_plot_area(self, frame: Frame):
        """Summary

        Parameters
        ----------
        frame : Frame
            Description

        Returns
        -------
        TYPE
            Description
        """
        pltframe = Frame(frame)

        self._actualFigure, (self._ax1, self._ax2) = plt.subplots(nrows=2, ncols=1)

        self.canvas = FigureCanvasTkAgg(self._actualFigure, master=pltframe)
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=TRUE)

        self._refresh_plot()

        plt.gca().set_aspect('equal', adjustable='box')

        toolbar = NavigationToolbar2TkAgg(self.canvas, pltframe)
        toolbar.update()

        self.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=TRUE)
        pltframe.pack(side=BOTTOM, fill=BOTH, expand=TRUE)

        return pltframe

    def _make_plot(self, plot1: tuple, plot2: tuple):
        plt.style.use('ggplot')
        colors = plt.rcParams['axes.color_cycle']

        self._ax1.clear()
        self._ax2.clear()

        explode = (0.1, 0)

        if plot1 is None:
            labelz = ["A", "B"]
            data = np.array([0.1, 0.1])
            self._ax1.pie(data, explode=explode, labels=labelz, colors=colors, autopct='%1.1f%%', shadow=False, startangle=45)
        else:
            self._ax1.pie(plot1[1], explode=explode, labels=plot1[0], colors=colors, autopct='%1.1f%%', shadow=False, startangle=45)
            pass

        self._ax1.set_xlabel("Fantasy Filter")
        self._ax1.xaxis.set_label_position('top')
        self._ax1.legend(loc='upper right')
        self._ax1.axis('equal')

        if plot2 is None:
            plot2 = np.array([0.1, 0.1, 0.1, 0.1])
            labelz = ["A", "B", "C", "D"]
            explode = (0.1, 0, 0, 0)
            self._ax2.pie(plot2, explode=explode, labels=labelz, colors=colors, autopct='%1.1f%%', shadow=False, startangle=45)
        elif plot2 == ([], []):
            plot2 = np.array([1.0])
            labelz = ["Generic"]
            self._ax2.pie(plot2, labels=labelz, colors=colors, autopct='%1.1f%%', shadow=False, startangle=45)
            self.msglbl["text"] = "Fantasy, Generic Fantasy"
            pass
        else:
            self._ax2.pie(plot2[1], labels=plot2[0], colors=colors, autopct='%1.1f%%', shadow=False, startangle=45)
            pass

        self._ax2.set_xlabel("Fantasy Sub-Genres", labelpad=20)
        self._ax2.xaxis.set_label_position('top')
        self._ax2.legend(loc='lower right')
        self._ax2.axis('equal')

        pass

    def _make_text_area(self, frame: Frame):
        """Summary

        Parameters
        ----------
        frame : Frame
            Description

        Returns
        -------
        TYPE
            Description
        """
        scrollbar = Scrollbar(frame)
        self.raw_text_area = Text(frame, width=198, height=600,
                                  yscrollcommand=scrollbar.set, wrap=WORD)
        self.raw_text_area.config(font=('Calibri', 12, 'normal'))
        scrollbar.config(command=self.raw_text_area.yview)

        scrollbar.pack(side="right", fill="y")
        self.raw_text_area.pack(side="left", fill="y")
        pass

    def _make_menubar(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        the_menu = Menu(self._root)

        file_menu = Menu(the_menu, tearoff=0)
        file_menu = self._file_menu_options(file_menu)
        the_menu.add_cascade(label="File", menu=file_menu)

        edit_menu = Menu(the_menu, tearoff=0)
        edit_menu = self._edit_menu_options(edit_menu)
        the_menu.add_cascade(label="Edit", menu=edit_menu)

        setting_menu = Menu(the_menu, tearoff=0)
        setting_menu = self._setting_menu_options(setting_menu)
        the_menu.add_cascade(label="Settings", menu=setting_menu)

        training_menu = Menu(the_menu, tearoff=0)
        training_menu = self._training_menu_options(training_menu)
        the_menu.add_cascade(label="Training", menu=training_menu)

        help_menu = Menu(the_menu, tearoff=0)
        help_menu = self._help_menu_options(help_menu)
        the_menu.add_cascade(label="Help", menu=help_menu)

        # Display the menu bar
        self._root.config(menu=the_menu)
        pass

    def _file_menu_options(self, file_menu):
        """Summary

        Parameters
        ----------
        file_menu : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_as_file, accelerator="Alt+S")

        self._root.bind_all("<Control-n>", lambda event: self.new_file(event))
        self._root.bind_all("<Control-o>", lambda event: self.open_file(event))
        self._root.bind_all("<Control-s>", lambda event: self.save_file(event))
        self._root.bind_all("<Alt-s>", lambda event: self.save_as_file(event))

        # Add a horizontal bar to group similar commands
        file_menu.add_separator()

        # Call for the function to execute when clicked
        file_menu.add_command(label="Quit", command=self.exit_application, accelerator="Alt+F4")

        return file_menu

    def _help_menu_options(self, help_menu):
        """Summary

        Parameters
        ----------
        help_menu : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        help_menu.add_command(label="About", command=self._About)
        help_menu.add_command(label="Documentation",
                              command=self._Document)
        return help_menu

    def _edit_menu_options(self, edit_menu):
        """Summary

        Parameters
        ----------
        edit_menu : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        edit_menu.add_command(label="Cut", command=self.onCut, accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=self.onCopy, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.onPaste, accelerator="Ctrl+V")
        return edit_menu

    def _setting_menu_options(self, setting_menu):
        """Summary

        Parameters
        ----------
        setting_menu : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        setting_menu.add_command(label="Font", command=self._FontChanger)
        setting_menu.add_command(label="Corpus", command=self._CorpusForm)
        return setting_menu

    def _training_menu_options(self, training_menu):
        """Summary

        Parameters
        ----------
        training_menu : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        training_menu.add_command(label="Fantasy Check", command=self._MainGenreForm)
        training_menu.add_command(label="Discrete Sub-Genres", command=self._SubGenreForm)
        return training_menu

    def _center(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        self._root.update()

        width = 800
        height = 600

        frm_width = self._root.winfo_rootx() - self._root.winfo_x()
        win_width = width + 2 * frm_width

        titlebar_height = self._root.winfo_rooty() - self._root.winfo_y()
        win_height = height + titlebar_height + frm_width

        x = self._root.winfo_screenwidth() // 2 - win_width // 2
        y = self._root.winfo_screenheight() // 2 - win_height // 2

        self._root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        pass

    def _load_settings(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        try:
            self._default_settings = self._configUtil.get("corpuspath")

            if self._default_settings != "":
                self._corpus_dir = self._default_settings
                pass
        except Exception as e:
            LogUtils.write("error", e)
            pass
        pass

    def _refresh_plot(self, pl1=None, pl2=None):
        self._make_plot(pl1, pl2)
        self.canvas.draw_idle()
        pass

    # Forms
    def _FontChanger(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        # open the font chooser and get the font selected by the user
        font = askfont(self._root, text="Text Sample")

        if font:
            # spaces in the family name need to be escaped
            font['family'] = font['family'].replace(' ', '\ ')
            font_str = "%(family)s %(size)i %(weight)s %(slant)s" % font
            if font['underline']:
                font_str += ' underline'
                pass
            if font['overstrike']:
                font_str += ' overstrike'
                pass
            self.statebar.set("Chosen font: {}".format(font_str.replace('\ ', ' ')))
            pass
            self.raw_text_area.configure(font=font_str)
            self.find_text.configure(font=font_str)
            self.statebar.label.configure(font=font_str)
        pass

    def _CorpusForm(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if not MainForm.corpusForm:
            corpusForm = CorpusForm(self._root)
            corpusForm.grab_set()
            MainForm.corpusForm = corpusForm
            pass
        pass

    def _MainGenreForm(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if not MainForm.mainGenreForm:
            mainGenreForm = MainGenreForm(self._root)
            mainGenreForm.grab_set()
            MainForm.mainGenreForm = mainGenreForm
            pass
        pass

    def _SubGenreForm(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if not MainForm.subGenreForm:
            subGenreForm = SubGenreForm(self._root)
            subGenreForm.grab_set()
            MainForm.subGenreForm = subGenreForm
            pass
        pass

    def _About(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        messagebox.showinfo("About", "Fantasy sub genre extractor by Nathan Zwelibanzi Khupe @2017")
        pass

    def _Document(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        filename = "Documentation\_build\html\index.html"
        webbrowser.open('file://' + os.path.realpath(filename))
        pass

    # FUNCTIONS
    def extract_naive_genres(self, bag_of_words):
        """Summary

        Parameters
        ----------
        bag_of_words : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        pass

    def process_main_genre(self):
        """Conduct Fantasy filtering

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        FileNotFoundError
            Description
        FileNotFoundError
            Description
        """
        try:
            result = "Error!"
            raw_text = self.raw_text_area.get(1.0, END)
            if raw_text.strip() == "":
                messagebox.showerror("File Error", "Text not found")
                raise FileNotFoundError("Test text not found")
                pass
            else:
                self.statebar.set("Checking main genre...".format())
                prob_dist = GenreClassifier.predict_MainGenre(raw_text)

                result = max(prob_dist, key=lambda item: item[1])[0]

                if result is None:
                    stop_spinner(self.spinner)
                    messagebox.showerror("Main Genre Check", "Fantasy genre check failed")
                    self.statebar.set("Main genre checking failed".format())
                    return None

                self.msglbl['text'] = result
                self.statebar.set("Main genre is {}".format(result))

                self.display_class_probability(prob_dist)

                label1 = [(prob_dist[0])[0], (prob_dist[1])[0]]
                plot1 = [(prob_dist[0])[1], (prob_dist[1])[1]]
                self._refresh_plot((label1, plot1))

                if "Non-Fantasy" != result:
                    self.statebar.set("Extracting sub genres...".format())

                    prob_dist2 = GenreClassifier.predict_SubGenre(raw_text)

                    if prob_dist2 is None:
                        raise TypeError("prob_dist2 is a 'NoneType' object")

                    labels2, plot2 = [], []
                    for (g, p) in prob_dist2:
                        if p >= 0.5:
                           result += ", " + g
                           labels2.append(g)
                           plot2.append(p)
                           pass
                        pass

                    self.msglbl['text'] = result

                    self._refresh_plot((label1, plot1), (labels2, plot2))
                    self._ax2.set_visible(True)

                    probabity_map = []
                    probabity_map.extend(prob_dist)
                    probabity_map.extend(sorted(prob_dist2, key=itemgetter(1), reverse=True))

                    self.display_class_probability(probabity_map)
                    LogUtils.write("info", "Predicted: {}".format(result))
                    pass
                else:
                    self._ax2.set_visible(False)
                    messagebox.showinfo("Sub Genre Check", "The supplied text is not fantasy")
                    pass
                pass
        except FileNotFoundError:
            messagebox.showerror("Classifier Error", "Train classifiers first!")
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
            messagebox.showerror("Processing Error", "An unexpected error occurred.")
            pass
        finally:
            stop_spinner(self.spinner)
            self.statebar.set("Statusbar |".format())
            pass
        pass

    def display_class_probability(self, probabity_map: list):
        self.GenreTable.delete(*self.GenreTable.get_children())
        line = 0
        for (g, p) in probabity_map:
            line += 1
            self.GenreTable.insert('', 'end', text=line, values=(g, p))
            pass
        pass

    # EVENTS
    def exit_application(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self._root.destroy()
        sys.exit()
        pass

    def open_file(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.filename = filedialog.askopenfilename(parent=self._root, initialdir='/')

        if self.filename:

            self.raw_text_area.delete(1.0, END)

            try:
                # Open file and put text in the text widget
                with open(self.filename, "r", encoding="utf-8") as _file:
                    self.raw_text_area.insert(1.0, _file.read())

                    # Update the text widget
                    self._root.update_idletasks()
                    pass
            except UnicodeDecodeError:
                messagebox.showwarning("Encoding Error", "This application only accepts utf-8 encoded text files")
                pass
            except Exception as e:
                LogUtils.write("error", e.args)
                traceback.print_exc()
                pass
            pass
        pass

    def new_file(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.raw_text_area.delete(1.0, END)
        self.filename = ""
        pass

    def save_file(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        new_text = self.raw_text_area.get(1.0, END)
        if self.filename != "":
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(new_text)
                pass
            pass
        else:
            self.save_as_file()
        pass

    def save_as_file(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        file = filedialog.asksaveasfile(mode='w')
        if file is not None:
            # Get text in the text widget and delete the last newline
            data = self.raw_text_area.get('1.0', END + '-1c')

            # Write the text and close
            file.write(data)
            file.close()
            pass
        pass

    def onClearText(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.raw_text_area.tag_remove("highlight", 1.0, END)
        pass

    def onCut(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if self.raw_text_area.tag_ranges("sel"):
            text = self.raw_text_area.get(SEL_FIRST, SEL_LAST)
            text.strip()
            self.raw_text_area.delete(SEL_FIRST, SEL_LAST)
            self.raw_text_area.clipboard_clear()
            self.raw_text_area.clipboard_append(text)
        pass

    def onCopy(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.raw_text_area.clipboard_clear()
        if self.raw_text_area.tag_ranges("sel"):
            copied_text = self.raw_text_area.get(SEL_FIRST, SEL_LAST)
            copied_text.strip()
            self.raw_text_area.clipboard_append(copied_text)
        pass

    def onPaste(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        try:
            text = self.raw_text_area.selection_get(selection='CLIPBOARD')
            self.raw_text_area.insert(INSERT, text)
        except TclError:
            pass
        pass

    def onFind(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        query = self.find_text.get()
        self.raw_text_area.tag_remove("highlight", 1.0, END)
        self.raw_text_area.tag_config('highlight', background="yellow")
        count = 0
        if query:
            index = '1.0'
            while 1:
                index = self.raw_text_area.search(query, index, nocase=1, stopindex=END)
                if not index:
                    break
                lastIndex = '%s+%dc' % (index, len(query))
                count = count + 1
                self.raw_text_area.tag_add('highlight', index, lastIndex)
                index = lastIndex
                pass
            pass
        self.find_text.focus_set()
        self.statebar.set("Found {} instances of '{}'".format(count, query))
        pass

    def onEntryFocusIn(self, event):
        """Summary

        Parameters
        ----------
        event : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        entry = event.widget
        entry.delete(0, END)
        entry.insert(0, '')
        pass

    def onEntryFocusOut(self, message, event=None):
        """Summary

        Parameters
        ----------
        message : TYPE
            Description
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        entry = event.widget
        if entry.get() == "":
            entry.insert(0, message)
        pass

    def onExtractButtonClick(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        start_spinner(self.spinner)

        self.processing_thread = threading.Thread(target=self.process_main_genre, args=())
        self.processing_thread.start()
        pass

    pass  # end of MainForm
class CorpusForm(Toplevel):
    def __init__(self, root, *args, **kargs):
        Toplevel.__init__(self, master=root, *args, **kargs)

        self.testing_corpus, self.train_corpus = None, None
        self.grab_set()
        self.configUtil = ConfigUtils(configparser.ConfigParser())
        self.configUtil.load()
        self.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.bind('<Escape>', lambda e: self.destroy())
        self._make_topmost()
        self._initGUI(root)
        pass

    def _initGUI(self, parent):
        self.wm_title("Configure Corpus")
        self.geometry("{}x{}+{}+{}".format(600, 560, parent.winfo_rootx() + 96, parent.winfo_rooty() - 30))

        self._make_pre_processing()
        self._make_console()
        pass

    def _make_topmost(self):
        """Makes this window the topmost window
        """
        self.lift()
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0)
        pass

    def _make_pre_processing(self):
        frame = Frame(self)

        corpus_frame = LabelFrame(frame, text="Corpus Directory")

        self.corpus_dir = Entry(corpus_frame)

        self._load_settings()

        self.corpus_dir.bind('<FocusIn>', lambda event: self.onEntryFocusIn(event=event,  message='corpus directory'))
        self.corpus_dir.bind('<FocusOut>', lambda event:
                             self.onEntryFocusOut(event=event, message='corpus directory'))
        self.corpus_dir.pack(side=LEFT, fill=X, expand=True)

        self.browse = tk.Button(corpus_frame, text="Browse", relief=FLAT,
                                     foreground="#fff", background="#639",
                                     activebackground="#14ff33", command=self.onBrowse)
        self.browse.pack(side=LEFT, padx=3, pady=3)

        self.save_to_config = tk.Button(corpus_frame, text="Save Path", relief=FLAT,
                                             foreground="#fff", background="#639af2",
                                             activebackground="#14ff33", command=self.onSaveToConfig)
        self.save_to_config.pack(side=LEFT, fill=X, pady=3)

        corpus_frame.pack(side=TOP, fill=X, padx=2, pady=2)

        ctrl_frame = LabelFrame(frame, text="Make Training and Testing Corpus")

        split_corpus_button = tk.Button(ctrl_frame, width=14, text="Process Corpus", relief=FLAT,
                                             foreground="#fff", background="#639",
                                             activebackground="#f4424e", command=self.onSplitCorpus)
        split_corpus_button.pack(side=LEFT, fill=X, padx=2, pady=2)

        self.spinner = Progressbar(ctrl_frame, orient="horizontal", mode="indeterminate")
        self.spinner.pack(side=LEFT, fill=X, expand=TRUE, padx=2, pady=2)

        save_corpus = tk.Button(ctrl_frame, text="Save Datasets", relief=FLAT,
                                     foreground="#fff", background="#639af2",
                                     activebackground="#14ff33", command=self.onSaveCorpus)
        save_corpus.pack(side=RIGHT, padx=2, pady=2)

        ctrl_frame.pack(side=TOP, fill=X)

        frame.pack(side=TOP, expand=1, fill=BOTH)

        return frame

    def _make_console(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        console = ConsoleWidget(self)
        console.pack(side=BOTTOM, fill=BOTH, expand=True)
        return console

    def _load_settings(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        try:
            self.dir_setting = self.configUtil.get("corpuspath")

            if self.dir_setting != "":
                self.corpus_dir.insert(0, self.dir_setting)
            else:
                self.corpus_dir.insert(0, 'corpus directory')
                pass
        except Exception as e:
            LogUtils.write("error", e)
            pass
        pass

    #Functions
    def _process_corpus(self):
        """Split the main corpus into a training and testing corpus
            then preprocess the training corpus
        """
        try:
            corpus = CorpusUtils.create_genred_corpus(directory=self.corpus_dir.get(), file_pattern=r'(?!\.?!\.).*\.txt', genre_file="genre.txt")

            self.testing_corpus, self.train_corpus = CorpusUtils.get_split_corpus(corpus)

            LogUtils.write("info",  "Corpus has been processed successfully.")
            LogUtils.write("critical",  "Remember to save the datasets!")
            pass
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
            pass
        finally:
            stop_spinner(self.spinner)
            print("debug", "process_corpus: operation finished")
            pass
        pass

    #Events
    def onQuit(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.destroy()
        MainForm.corpusForm = None

        self.grab_release()
        pass

    def onBrowse(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        corpus_filepath = filedialog.askdirectory(parent=self, initialdir='/')
        if corpus_filepath != "":
            self.corpus_dir.delete(0, END)
            self.corpus_dir.insert(0, corpus_filepath)
            pass
        pass

    def onSaveToConfig(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        pathOnEntry = self.corpus_dir.get()
        if pathOnEntry != "corpus directory":
            self.configUtil.set("CorpusPath", pathOnEntry)
            self.configUtil.save()
            pass
        pass

    def onSplitCorpus(self, event=None):
        """Split the main corpus into a training and testing corpus
            then preprocess the training corpus

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        Event
            Tkinter Event object
        """
        try:
            pathOnEntry = self.corpus_dir.get()
            if pathOnEntry != self.dir_setting:
                start_spinner(self.spinner)
                self.processing_thread = threading.Thread(target=self._process_corpus, args=())
                self.processing_thread.start()
                pass
            else:
                start_spinner(self.spinner)
                self.processing_thread = threading.Thread(target=self._process_corpus, args=())
                self.processing_thread.start()
                pass
            pass
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
            pass
        pass

    def onSaveCorpus(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        ValueError
            Description
        ValueError
        Description
        """
        if self.testing_corpus is not None or self.train_corpus is not None:
            FileManager.write("TestCorpus.pickle", self.testing_corpus)
            LogUtils.write("info", "Testing corpus saved...")
            FileManager.write("TrainCorpus.pickle", self.train_corpus)
            LogUtils.write("info", "Training corpus saved...")
            LogUtils.write("info", "finished...")

        else:
            LogUtils.write("error", "Cannot save corpus because one or more corpus are empty")
            raise ValueError("pre_process_corpus: Cannot save corpus because one or more corpus are empty")
        pass

    def onEntryFocusOut(self, message, event=None):
        """Summary

        Parameters
        ----------
        message : TYPE
            Description
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        entry = event.widget
        if entry.get() == "":
            entry.insert(0, message)
        pass

    def onEntryFocusIn(self, message, event=None):
        """Summary

        Parameters
        ----------
        message : TYPE
            Description
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        entry = event.widget
        if entry.get() == message:
            entry.delete(0, END)
            entry.insert(0, '')
        pass

    pass  # end of CorpusForm
class MainGenreForm(Toplevel):
    def __init__(self, root, *args, **kargs):
        """Summary

        Parameters
        ----------
        root : TYPE
            Description
        *args
            Description
        **kargs
            Description
        """
        Toplevel.__init__(self, master=root, *args, **kargs)

        self.mainclassifier = None

        self.testing_corpus = None
        self.train_corpus = None

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.bind('<Escape>', lambda e: self.destroy())

        self._make_topmost()
        self._initGUI(root)
        pass

    def _initGUI(self, parent):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        self.wm_title("Fantasy Classifier Training")
        self.geometry("{}x{}+{}+{}".format(650, 690, parent.winfo_rootx() + 96, parent.winfo_rooty() - 113))

        self._make_main_genre()
        self._make_console()
        pass

    def _make_topmost(self):
        """Makes this window the topmost window
        """
        self.lift()
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0)
        pass

    def _make_main_genre(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        frame = Frame(self)

        visualizerFrame = self._make_tabs(frame)

        ctrl_frame = Frame(frame)

        self.train_main_button = tk.Button(ctrl_frame, width=12, text="Train Classifier", relief=FLAT,
                                                foreground="#fff", background="#639",
                                                activebackground="#f4424e", command=self.onTrainBinaryGenreCheck)
        self.train_main_button.pack(side=LEFT, fill=X, padx=2, pady=2)

        self.spinner = Progressbar(ctrl_frame, orient="horizontal", mode="indeterminate")
        self.spinner.pack(side=LEFT, fill=X, expand=TRUE, padx=2, pady=2)

        self.save_classifier = tk.Button(ctrl_frame, text="Save Classifier", relief=FLAT,
                                              foreground="#fff", background="#639af2",
                                              activebackground="#14ff33", command=self.onSaveMainClassifier)
        self.save_classifier.pack(side=RIGHT, padx=2, pady=2)

        ctrl_frame.pack(side=BOTTOM, fill=X)
        frame.pack(side=TOP, fill=BOTH, expand=TRUE)

        return frame

    def _make_tabs(self, parentFrame):
        """Summary

        Parameters
        ----------
        parentFrame : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        notebook = Notebook(parentFrame, height=430)

        page1 = self._make_confusion_matrix_tab(notebook)
        page2 = self._make_table_tab(notebook)
        page3 = self._make_plot_tab1(notebook)
        page4 = self._make_plot_tab2(notebook)

        notebook.add(page1, text='Classification Metrics')
        notebook.add(page2, text='Feature Metrics')
        notebook.add(page3, text='Classification Report')
        notebook.add(page4, text='Confusion Matrix')

        notebook.pack(side=TOP, expand=1, fill=BOTH)
        return notebook

    def _make_confusion_matrix_tab(self, parentFrame):
        """
        Parameters
        ----------
        parentFrame : TYPE
            Description
        """
        self.main_confusion = ScrolledText(parentFrame)
        self.main_confusion.pack(expand=TRUE, side=TOP, fill="both")
        cr_report = FileManager.read("clsf_main_genre_report.pickle")
        measures = FileManager.read("filtering_measures.pickle")
        class_stats = FileManager.read("clsf_main_genre_stats.pickle")
        if cr_report is not None and measures is not None:
            report = extract_report_text(cr_report, measures, class_stats)
            if report is not None:
                self.main_confusion.insert(1.0, report)
                pass
            pass
        return self.main_confusion

    def _make_table_tab(self, parentfrm):
        """Summary

        Parameters
        ----------
        parentfrm : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        tableFrame = Frame(parentfrm)
        tableScroll = Scrollbar(tableFrame)

        self.table = Treeview(tableFrame)
        self.table['columns'] = ('Term', 'Occurrence', "Weight")

        self.table.heading("#0", text='#', anchor='w')
        self.table.heading('Term', text='Term', anchor='w')
        self.table.heading('Occurrence', text='Occurrence', anchor='e')
        self.table.heading('Weight', text='Weight', anchor='e')

        self.table.column('#0', anchor='w', width=50)
        self.table.column('Term', anchor='w', width=150)
        self.table.column('Occurrence', anchor='e', width=100)
        self.table.column('Weight', anchor='e', width=100)

        tableScroll.configure(command=self.table.yview)
        tableScroll.pack(side="right", fill=Y)

        self.table.configure(yscrollcommand=tableScroll.set)
        self.table.pack(side="left", fill=BOTH, expand=TRUE)
        tableFrame.pack(side=TOP, fill=BOTH, expand=TRUE)

        self._display_feature_metrics()

        return tableFrame

    def _make_plot_tab1(self, parentFrm):
        pltframe = Frame(parentFrm)
        actualFigure = plt.figure(figsize=(8, 4))
        actualFigure.patch.set_facecolor('blue')
        actualFigure.patch.set_alpha(0.25)

        class_report = FileManager.read("clsf_main_genre_report.pickle")

        if class_report is not None:
            plot_class_report(class_report, "wide")
            plt.colorbar()
            pass
        else:
            smiley_plot()
            pass

        plt.gca().set_aspect('equal', adjustable='box')

        self.canvas = FigureCanvasTkAgg(actualFigure, master=pltframe)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=TRUE)
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=TRUE)

        toolbar = NavigationToolbar2TkAgg(self.canvas, pltframe)
        toolbar.update()
        toolbar.pack(side=TOP, fill=BOTH, expand=FALSE)

        return pltframe

    def _make_plot_tab2(self, parentFrm):
        """Summary

        Parameters
        ----------
        parentFrm : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        pltframe = Frame(parentFrm)
        actualFigure = plt.figure(figsize=(8, 4))
        actualFigure.patch.set_facecolor('blue')
        actualFigure.patch.set_alpha(0.25)

        cm = FileManager.read("clsf_main_genre_confusion.pickle")

        if cm is not None:
            plot_confusion_matrix(cm)
            plt.colorbar()
            pass
        else:
            smiley_plot()
            pass

        plt.gca().set_aspect('equal', adjustable='box')

        self.canvas2 = FigureCanvasTkAgg(actualFigure, master=pltframe)
        self.canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=TRUE)
        self.canvas2._tkcanvas.pack(side=TOP, fill=BOTH, expand=TRUE)

        toolbar = NavigationToolbar2TkAgg(self.canvas2, pltframe)
        toolbar.update()
        toolbar.pack(side=TOP, fill=BOTH, expand=FALSE)

        return pltframe

    def _make_console(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        console = ConsoleWidget(self)
        console.pack(side=BOTTOM, fill=BOTH, expand=True)
        return console

    def _refresh_plots(self, class_report, confusion_matrix):
        """Summary

        Parameters
        ----------
        class_report : TYPE
            Description
        confusion_matrix : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        plot_class_report(class_report, "wide")
        self.canvas.draw()

        plot_confusion_matrix(confusion_matrix)
        self.canvas2.draw()
        pass

    def _display_feature_metrics(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        fmetrics = FileManager.read("clsf_main_genre_features.pickle")
        if fmetrics is not None:
            self.table.delete(*self.table.get_children())
            line = 0
            for word, count, tfidf in fmetrics:
                line += 1
                self.table.insert('', 'end', text=line,
                                  values=(word, count, tfidf))
                pass
            pass
        pass

    #Functions
    def convert_to_binary_class(self, testing_corpus, train_corpus):
        """Summary

        Parameters
        ----------
        testing_corpus : TYPE
            Description
        train_corpus : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        test_output, train_output = [], []

        #print("converting training corpus to binary class")
        for fileid, genres, raw_text in train_corpus:
            if 'Non-Fantasy' in genres:
                genres = 'Non-Fantasy'
            else:
                genres = 'Fantasy'
                pass
            pass
            train_output.append((fileid, genres, raw_text))
            #print(fileid+" processed")
            pass

        #print("converting testing corpus to binary class")
        for fileid, genres, raw_text in testing_corpus:
            if 'Non-Fantasy' in genres:
                genres = 'Non-Fantasy'
            else:
                genres = 'Fantasy'
                pass
            pass
            test_output.append((fileid, genres, raw_text))
            #print(fileid+" processed")
            pass

        return test_output, train_output

    def process_main_genre(self):
        """Summary

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        ValueError
            Description
        ValueError
        Description
        """
        LogUtils.write("debug", "process_main_genre: started")
        try:
            self.train_corpus = FileManager.read("TrainCorpus.pickle")
            self.testing_corpus = FileManager.read("TestCorpus.pickle")

            if self.testing_corpus is None or self.train_corpus is None:
                raise ValueError("testing or training corpus is empty. Have you done preprocessing yet?")
                return None

            self.testing_corpus, self.train_corpus = self.convert_to_binary_class(self.testing_corpus, self.train_corpus)

            LogUtils.write("info", "split training corpus")
            train_documents = []
            train_genres = []
            for f, g, r in self.train_corpus:  # f = filename, g = genretag, r = raw text file
                LogUtils.write("debug", f + " processed")
                train_documents.append(r)
                train_genres.append(g)
                pass

            LogUtils.write("info", "training genre mappings created")
            if train_documents is not None and train_genres is not None:
                LogUtils.write("debug", "begin training classifier")
                self.mainclassifier, self.features = GenreClassifier.train_MainGenre_on(train_documents, train_genres)

                LogUtils.write("debug", "begin testing classifier")
                LogUtils.write("info", "split testing corpus")
                test_documents = []
                test_genres = []
                for f, g, r in self.testing_corpus:  # f = filename, g = genretag, r = raw text file
                    LogUtils.write("debug", f + " processed")
                    test_documents.append(r)
                    test_genres.append(g)
                    pass

                predictions = self.mainclassifier.predict(test_documents)

                LogUtils.write("debug", "begin assessing classifier")
                self.confusion_matrix = ConfusionMatrix(test_genres, predictions)
                last_class_report = "Fantasy Classifier Accuracy: {}%\n".format(round(accuracy_score(test_genres, predictions) * 100, 2))

                self.class_report = classification_report(test_genres, predictions)
                self.class_measures = GenreClassifier.binary_class_measures(test_genres, predictions)

                self.class_stats = self.confusion_matrix.stats()
                last_class_report += extract_report_text(self.class_report, self.class_measures, self.class_stats)

                self.main_confusion.delete(1.0, END)
                self.main_confusion.insert(1.0, last_class_report)

                LogUtils.write("info", "Fantasy Classifier trained, tested and assessed.")
                LogUtils.write("critical", "Remember to save the classifier!")

                pass
            else:
                raise ValueError("Failed to create one or more datasets")
            pass
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
            pass
        finally:
            stop_spinner(self.spinner)
            pass
        pass

    #Events
    def onQuit(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.destroy()
        MainForm.mainGenreForm = None

        self.grab_release()
        pass

    def onTrainBinaryGenreCheck(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        start_spinner(self.spinner)

        threading._start_new_thread(self.process_main_genre, ())

        pass

    def onSaveMainClassifier(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if self.mainclassifier:
            FileManager.write("clsf_main_genre_features.pickle", self.features)
            GenreClassifier.save(self.mainclassifier,
                                 ClassifierType.BinaryGenres)
            FileManager.write("clsf_main_genre_report.pickle", self.class_report)
            FileManager.write("clsf_main_genre_stats.pickle", self.class_stats)
            FileManager.write("clsf_main_genre_confusion.pickle", self.confusion_matrix)
            FileManager.write("filtering_measures.pickle", self.class_measures)

            self._display_feature_metrics()
            self._refresh_plots(self.class_report, self.confusion_matrix)
        else:
            LogUtils.write("error", "self.mainclassifier is None")
            messagebox.showerror("Save Main Genre Classifier",
                                 "There is currently no classifier to save.")
            pass
        pass

    pass  # end of MainGenreForm
class SubGenreForm(Toplevel):
    """Summary

    Attributes
    ----------
    actualFigure : TYPE
        Description
    canvas : TYPE
        Description
    class_measures : TYPE
        Description
    class_report : TYPE
        Description
    save_classifier : TYPE
        Description
    spinner : TYPE
        Description
    sub_confusion : TYPE
        Description
    subclassifier : TYPE
        Description
    table : TYPE
        Description
    testing_corpus : TYPE
        Description
    toolbar : TYPE
        Description
    train_corpus : TYPE
        Description
    train_sub_button : TYPE
        Description
    """
    def __init__(self, root, *args, **kargs):
        """Summary

        Parameters
        ----------
        root : TYPE
            Description
        *args
            Description
        **kargs
            Description
        """
        Toplevel.__init__(self, master=root, *args, **kargs)

        self.subclassifier = None
        self.testing_corpus = None
        self.train_corpus = None

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.onQuit)
        self.bind('<Escape>', lambda e: self.destroy())

        self._make_topmost()
        self._initGUI(root)
        pass

    def _initGUI(self, parent):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        self.wm_title("Classifier Training")
        self.geometry("{}x{}+{}+{}".format(600, 690, parent.winfo_rootx() + 96, parent.winfo_rooty() - 113))

        self._make_sub_genre_tab()
        self._make_console()
        pass

    def _make_topmost(self):
        """Makes this window the topmost window
        """
        self.lift()
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0)
        pass

    def _make_sub_genre_tab(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        frame = Frame(self)

        visualizerFrame = self._make_tabs(frame)

        ctrl_frame = Frame(frame)

        self.train_sub_button = tk.Button(ctrl_frame, width=14, text="Train Classifier", relief=FLAT,
                                               foreground="#fff", background="#639",
                                               activebackground="#f4424e", command=self.onTrainMultiGenreClassifier)
        self.train_sub_button.pack(side=LEFT, fill=X, padx=2, pady=2)

        self.spinner = Progressbar(ctrl_frame, orient="horizontal", mode="indeterminate")
        self.spinner.pack(side=LEFT, fill=X, expand=TRUE, padx=2, pady=2)

        self.save_classifier = tk.Button(ctrl_frame, width=14, text="Save Classifier", relief=FLAT,
                                              foreground="#fff", background="#639af2",
                                              activebackground="#14ff33", command=self.onSaveSubGenreClassifier)
        self.save_classifier.pack(side=RIGHT, padx=2, pady=2)

        ctrl_frame.pack(side=BOTTOM, fill=X)

        frame.pack(side=TOP, fill=BOTH, expand=TRUE)
        return frame

    def _make_console(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        console = ConsoleWidget(self, height=250)
        console.pack(side=BOTTOM, fill=BOTH, expand=True)
        return console

    def _make_tabs(self, parentFrame):
        """Summary

        Parameters
        ----------
        parentFrame : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        notebook = Notebook(parentFrame, height=430)

        page1 = self._make_classification_metric_tab(notebook)
        page2 = self._make_table_tab(notebook)
        page3 = self._make_plot_tab(notebook)

        notebook.add(page1, text='Classification Metrics')
        notebook.add(page2, text='Feature Metrics')
        notebook.add(page3, text='Classification Report')

        notebook.pack(side=TOP, expand=1, fill=BOTH)
        return notebook

    def _make_classification_metric_tab(self, parentFrame):
        """
        Parameters
        ----------
        parentFrame : TYPE
            Description
        """
        self.sub_confusion = ScrolledText(parentFrame)
        self.sub_confusion.pack(expand=TRUE, side=TOP, fill="both")

        cr_report = FileManager.read("clsf_sub_genre_report.pickle")
        cl_stat = FileManager.read("clsf_sub_genre_stats.pickle")

        if cr_report is not None:
            report = extract_report_text(cr_report, cl_stat, None)
            if report is not None:
                self.sub_confusion.insert(1.0, report)
                pass
            pass

        return self.sub_confusion

    def _make_table_tab(self, parentfrm):
        """Summary

        Parameters
        ----------
        parentfrm : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        tableFrame = Frame(parentfrm)
        tableScroll = Scrollbar(tableFrame)

        self.table = Treeview(tableFrame)
        self.table['columns'] = ('Term', 'Occurrence')

        self.table.heading("#0", text='#', anchor='w')
        self.table.heading('Term', text='Term', anchor='w')
        self.table.heading('Occurrence', text='Occurrence', anchor='e')

        self.table.column('#0', anchor='w', width=50)
        self.table.column('Term', anchor='w', width=150)
        self.table.column('Occurrence', anchor='e', width=100)

        tableScroll.configure(command=self.table.yview)
        tableScroll.pack(side="right", fill=Y)

        self.table.configure(yscrollcommand=tableScroll.set)
        self.table.pack(side="left", fill=BOTH, expand=TRUE)
        tableFrame.pack(side=TOP, fill=BOTH, expand=TRUE)

        self._display_feature_metrics()

        return tableFrame

    def _make_plot_tab(self, parentFrm):
        """Summary

        Parameters
        ----------
        parentFrm : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        pltframe = Frame(parentFrm)
        actualFigure = plt.figure(figsize=(8, 4))
        actualFigure.patch.set_facecolor('blue')
        actualFigure.patch.set_alpha(0.25)

        cr = FileManager.read("clsf_sub_genre_report.pickle")

        if cr is not None:
            plot_class_report(cr, "narrow", ['Dark Fantasy', 'Juvenile Fantasy', 'Science Fantasy',
       'Sword and Sorcery'])
            plt.colorbar()
        else:
            smiley_plot()

        plt.gca().set_aspect('equal', adjustable='box')

        self.canvas = FigureCanvasTkAgg(actualFigure, master=pltframe)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=TRUE)
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=TRUE)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, pltframe)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill=BOTH, expand=FALSE)

        return pltframe

    def _refresh_plots(self, cr, class_names):
        """Summary

        Parameters
        ----------
        cr : TYPE
            Description
        class_names : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        plot_class_report(cr, "narrow", class_names)
        self.canvas.draw()
        pass

    def _display_feature_metrics(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        fmetrics = FileManager.read("clsf_sub_genre_features.pickle")
        if fmetrics is not None:
            self.table.delete(*self.table.get_children())
            line = 0
            for word, count in fmetrics:
                line += 1
                self.table.insert('', 'end', text=line, values=(word, count))
                pass
            pass
        pass

    # Functions
    def convert_to_multi(self, test_corpus, train_corpus):
        """Summary

        Parameters
        ----------
        test_corpus : TYPE
            Description
        train_corpus : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        train_output, test_output = [], []
        for fileid, genres, raw_text in train_corpus:
            if 'Non-Fantasy' in genres:
                continue
            elif 'Fantasy' in genres:
                genres.remove('Fantasy')
                pass
            train_output.append((fileid, genres, raw_text))
            pass

        for fileid, genres, raw_text in test_corpus:
            if 'Non-Fantasy' in genres:
                continue
            elif 'Fantasy' in genres:
                genres.remove('Fantasy')
                pass

            test_output.append((fileid, genres, raw_text))
            pass

        return test_output, train_output

    def process_sub_genre(self):
        """Summary

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        ValueError
            Description
        ValueError
        Description
        """
        try:
            train_corpus, test_corpus = [], []

            test_corpus = FileManager.read("TestCorpus.pickle")
            train_corpus = FileManager.read("TrainCorpus.pickle")

            if test_corpus is None or train_corpus is None:
                raise ValueError("testing or training corpus is empty. Configure corpus first.")
                pass

            test_corpus, train_corpus = self.convert_to_multi(test_corpus, train_corpus)

            LogUtils.write("info", "split training corpus")
            train_documents = []
            train_genres = []
            for f, g, r in train_corpus:  # f = filename, g = genretag, r = raw text file
                LogUtils.write("debug", f + " processed")
                train_documents.append(r)
                train_genres.append(g)
                pass
            LogUtils.write("info", "training genre mappings created")

            if train_documents is not None and train_genres is not None:

                LogUtils.write("debug", "create label binarizer")
                lblbinarizer = preprocessing.MultiLabelBinarizer()
                lblArray = lblbinarizer.fit_transform(train_genres)
                FileManager.write("MultiLabelBinarizer.pickle", lblbinarizer)

                LogUtils.write("debug", "begin training classifier")
                self.subclassifier, self.features = GenreClassifier.train_SubGenres_on(train_documents, lblArray)

                LogUtils.write("debug", "begin testing classifier")
                test_documents, test_genres = [], []
                LogUtils.write("info", "split testing  corpus")
                for f, g, r in test_corpus:  # f = filename, g = genretag, r = raw text file
                    LogUtils.write("debug", f + " processed")
                    test_documents.append(r)
                    test_genres.append(g)
                    pass
                LogUtils.write("info", "testing genre mappings created")

                if self.subclassifier:
                    predictions = self.subclassifier.predict(test_documents)
                    predicted_genres = lblbinarizer.inverse_transform(predictions)

                    print("predicted_genres", predicted_genres)

                    LogUtils.write("debug", "begin assessing classifier")
                    LogUtils.write("warn", "\nPrediction Errors")
                    for (f, g, r), p in zip(test_corpus, predicted_genres):
                        print("g", g, "p", p)
                        if g != p:
                            LogUtils.write("warn", "Story: {} \nExpected: {}, Predicted: {}\n".format(f, g, p))
                            pass
                        pass

                    binary_test_genres = lblbinarizer.transform(test_genres)

                    last_class_report = "Sub Genre Accuracy: {}%".format(accuracy_score(binary_test_genres, predictions) * 100)

                    self.class_report = classification_report(binary_test_genres, predictions)
                    self.class_measures = GenreClassifier.multi_class_measures(binary_test_genres, predictions)

                    last_class_report += extract_report_text(self.class_report, self.class_measures, None)

                    self.sub_confusion.delete(1.0, END)
                    self.sub_confusion.insert(1.0, last_class_report)

                    self._display_feature_metrics()
                    self._refresh_plots(self.class_report, lblbinarizer.classes_)

                    LogUtils.write("info", "Fantasy Sub Classifier trained, tested and assessed.")
                    LogUtils.write("critical", "Remember to save the classifier!")
                    pass
                pass
            pass
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
        finally:
            stop_spinner(self.spinner)
            pass
        pass

    # Events
    def onQuit(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.destroy()
        MainForm.subGenreForm = None

        self.grab_release()
        pass

    def onTrainMultiGenreClassifier(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        start_spinner(self.spinner)

        t = threading._start_new_thread(self.process_sub_genre, ())
        pass

    def onSaveSubGenreClassifier(self, event=None):
        """Summary

        Parameters
        ----------
        event : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if self.subclassifier:
            FileManager.write("clsf_sub_genre_features.pickle", self.features)
            FileManager.write("clsf_sub_genre_report.pickle", self.class_report)
            FileManager.write("clsf_sub_genre_stats.pickle", self.class_measures)
            GenreClassifier.save(self.subclassifier, ClassifierType.MultiClass)

        else:
            LogUtils.write("error", "self.subclassifier is None")
            messagebox.showerror("Save Main Genre Classifier",
                                 "There is currently no classifier to save.")
            pass
        pass

    pass  # end of SubGenreForm

# Global functions
def start_spinner(spinner: Progressbar):
    """Start a given spinner control

    Notes
    --------
    While spinner is  moving other events are ignored

    Parameters
    ----------
    spinner : Progressbar
        Progressbar sscontrol set to indeterminate time
    """
    spinner.grab_set()
    spinner.start(25)
    pass


def stop_spinner(spinner: Progressbar):
    """Stop a given spinner control

    Parameters
    ----------
    spinner : Progressbar
        Progressbar control set to indeterminate time
    """
    spinner.grab_release()
    spinner.stop()
    pass


def extract_report_text(report,
                        measures,
                        confusion_matrix: OrderedDict) -> str:
    """Create a textual classification report

    Parameters
    ----------
    report : TYPE
        classification report
    measures : TYPE
        detailed statistics or measures
    confusion_matrix : OrderedDict
        confusion matrix dictionary
    Returns
    -------
    str
        Classification Report
    """
    output = "\n--------------------------------\n\nClassification Report:\n"
    output += "{}".format(report)
    output += "\n--------------------------------\n\nDetailed Measures:\n"

    if measures:
        for k, v in measures.items():
            output += "\n{}: {}\n".format(k.title(), v)
            pass
        pass

    if confusion_matrix:
        output += "\n----------------------\n\nClassification Statistics:\n"
        for k, v in confusion_matrix.items():
            output += "\n{}: {}\n".format(k.title(), v)
            pass
        pass

    return output


def plot_class_report(classification_report, orientation: str, class_names=None):
    """Plot sklearn classification report

    Parameters
    ----------
    classification_report : TYPE
        classification report text
    orientation : str
        Wide or Narrow orientation
    class_names : list, optional
        Class labels
    """
    plt.gcf().clear()
    plt.cla()
    plt.clf()

    lines = classification_report.split('\n')

    orientation = orientation.lower()

    classes, plotMat = [], []
    support = []

    if class_names is None:
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()

            if len(t) < 2:
                continue

            classes.append(t[0])

            v = [float(x) for x in t[1: len(t) - 1]]

            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)
            pass
        pass
    else:
        for line in lines[2 : (len(lines) - 2)]:
            t = line.strip().split()

            if len(t) < 2:
                continue

            classes.append(t[0])

            v = [float(x) for x in t[1: len(t) - 1]]

            support.append(int(t[-1]))
            plotMat.append(v)
            pass
        pass

    cmap = plt.cm.Purples
    plt.title("Classification Report")

    xticklabels, yticklabels = None, None

    if orientation == "narrow":
        plt.ylabel('Metrics')
        plt.xlabel('Classes')

        xticklabels = ['{0} ({1})'.format(class_names[index], _support)
                       for index, _support in enumerate(support)]
        yticklabels = ['Precision', 'Recall', 'F1-score']

        plotMat = np.array(plotMat)
        cmd = plotMat.T

        plt.imshow(plotMat.T, interpolation="nearest", cmap=cmap)
        pass
    elif orientation == "wide":
        plt.ylabel('Classes')
        plt.xlabel('Metrics')

        yticklabels = ['{0} ({1})'.format(class_names[index], _support)
                       for index, _support in enumerate(support)]
        xticklabels = ['Precision', 'Recall', 'F1-score']
        cmd = np.asarray(plotMat)
        plt.imshow(plotMat, interpolation="nearest", cmap=cmap)
        pass
    else:
        raise TypeError("invalid orientation argument")
        pass
    thresh = (cmd.max() + cmd.min()) * 0.5

    plt.gca().axes.grid(False)

    xtick_marks = np.arange(len(xticklabels))
    ytick_marks = np.arange(len(yticklabels))

    plt.xticks(xtick_marks, xticklabels, rotation=30)
    plt.yticks(ytick_marks, yticklabels, rotation=30)

    width, height = cmd.shape

    for x in range(width):
        for y in range(height):
            plt.text(y, x, cmd[x, y],
                     horizontalalignment='center',
                     color="white" if cmd[x, y] >= thresh else "black")

    plt.tight_layout()
    pass


def plot_confusion_matrix(confusion_matrix: ConfusionMatrix):
    """Plot confusion matrix

    Parameters
    ----------
    confusion_matrix : ConfusionMatrix
        sklearn ConfusionMatrix object
    """
    plt.gcf().clear()
    plt.cla()
    plt.clf()

    cmap = plt.cm.Purples
    df = confusion_matrix.to_dataframe(False)

    cmd = confusion_matrix.to_array()
    thresh = cmd.max() / 2

    plt.title("Confusion Matrix")

    plt.ylabel(df.index.name)
    plt.xlabel(df.columns.name)

    plt.imshow(df, interpolation="nearest", cmap=cmap)
    plt.gca().axes.grid(False)

    tick_marks_col = np.arange(len(df.columns))
    tick_marks_idx = tick_marks_col.copy()

    plt.yticks(tick_marks_idx, df.index, rotation=45)
    plt.xticks(tick_marks_col, df.columns, rotation=45)

    width, height = cmd.shape

    for x in range(width):
        for y in range(height):
            plt.text(y, x, cmd[x, y],
                     horizontalalignment='center',
                     color="white" if cmd[x, y] > thresh else "black")

    plt.tight_layout()
    pass


def smiley_plot():
    """Placeholder smiley face plot

    """
    y = x = np.arange(-10, 20, 1)
    x, y = np.meshgrid(x, y)

    plt.contour(x, y, (x * x * (x ** 2 + 2 * y * y - y - 40) + y * y * (y * y - y - 40) + 25 * y + 393) * ((x + 3) ** 2 + (y - 5) ** 2 - 2) * ((x - 3) ** 2 + (y - 5) ** 2 - 2) * (x * x + (y - 2) ** 2 - 64), [0])
    pass
