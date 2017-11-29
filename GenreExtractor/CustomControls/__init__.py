"""This module contains definitions of custom user interface controls using
components from the tkinter package
"""
import logging
import time
import tkinter as tk
import traceback

from Utilities import LogUtils
from builtins import Exception
from builtins import property
from builtins import zip
from threading import Thread
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import *

_author__ = "Nathan Zwelibanzi Khupe"
__copyright__ = "Copyright 2017, Nathan Zwelibanzi Khupe"
__credits__ = ["Nathan Zwelibanzi Khupe"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nathan Zwelibanzi Khupe"
__email__ = "001knatz@gmail.com"
__docformat__ = 'numpy'


class StatusBar(Frame):
    """Status bar control

    Attributes
    ----------
    label : TYPE
        Description
    label : TYPE
    Description
    """

    def __init__(self, root: Tk, **kwargs):
        """Summary

        Parameters
        ----------
        root : Tk
            Description
        **kwargs
            Description
        """
        Frame.__init__(self, root, relief=SUNKEN, borderwidth=1, **kwargs)
        self.label = Label(self, anchor=W, font=('arial', 10, 'normal'))
        self.label.pack(side=LEFT, fill=X, pady=1)
        pass

    def set(self, format, **kwargs):
        """Summary

        Parameters
        ----------
        format : TYPE
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.label.config(text=format % kwargs)
        self.label.update_idletasks()
        pass

    def clear(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        self.label.config(text='Status Bar')
        self.label.update_idletasks()
        pass

    pass  # end of StatusBar


class LoggerWidget(logging.Handler):
    """Summary

    Attributes
    ----------
    red : TYPE
        Description
    widget : TYPE
        Description
    red : TYPE
    Description
    widget : TYPE
    Description
    """

    def __init__(self, widget):
        """Summary

        Parameters
        ----------
        widget : TYPE
            Description
        """
        logging.Handler.__init__(self)
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter(
            '%(asctime)s;  %(levelname)s: %(message)s ', datefmt='%d-%m-%Y %H:%M:%S'))

        self.widget = widget
        self.widget.config(state='disabled')
        self.widget.tag_config("INFO", foreground="black")
        self.widget.tag_config("DEBUG", foreground="grey")
        self.widget.tag_config("WARNING", foreground="orange")
        self.widget.tag_config("ERROR", foreground="red")
        self.widget.tag_config("CRITICAL", foreground="red", underline=1)
        self.red = self.widget.tag_configure("red", foreground="red")
        pass

    def emit(self, record):
        """Summary

        Parameters
        ----------
        record : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        try:
            self.widget.config(state='normal')
            self.widget.insert(END, self.format(
                record) + '\n', record.levelname)
            self.widget.see(END)
            self.widget.config(state='disabled')
            self.widget.update()
        except:
            pass
        pass

    pass  # end of LoggerWidget


class ConsoleWidget(Frame):
    """Summary

    Attributes
    ----------
    container_frame : TYPE
        Description
    isNotShowing : TYPE
        Description
    show : TYPE
        Description
    title_frame : TYPE
        Description
    toggle_button : TYPE
        Description
    container_frame : TYPE
    Description
    isNotShowing : TYPE
    Description
    show : TYPE
    Description
    title_frame : TYPE
    Description
    toggle_button : TYPE
    Description
    """

    def __init__(self, parent, *args, **options):
        """Summary

        Parameters
        ----------
        parent : TYPE
            Description
        *args
            Description
        **options
            Description
        """
        Frame.__init__(self, parent, *args, **options)

        self.show = IntVar()
        self.show.set(0)

        self.isNotShowing = BooleanVar()
        self.isNotShowing.set(True)

        self._make_title_frame(self)

        self.container_frame = Frame(self, relief="sunken")
        self._make_text_area(self.container_frame)
        self.container_frame.pack(fill=BOTH, expand=True)
        self.toggle(Event)

        pass

    def _make_title_frame(self, frame):
        """Summary

        Parameters
        ----------
        frame : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.title_frame = LabelFrame(self)
        self.toggle_button = Label(self.title_frame, text="- Console -")
        self.toggle_button.bind("<Button-1>", lambda e: self.toggle(e))
        self.toggle_button.pack(side="left", fill=BOTH, expand=True)
        self.title_frame.pack(side=BOTTOM, fill=BOTH, expand=False)
        pass

    def _make_text_area(self, frame):
        """Summary

        Parameters
        ----------
        frame : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        scrolledText = ScrolledText(frame)
        scrolledText.pack(expand=TRUE, fill="both")
        widget_log = LoggerWidget(scrolledText)
        LogUtils.add_handler(widget_log)
        pass

    def toggle(self, Event: None):
        """Summary

        Parameters
        ----------
        Event : None
            Description

        Returns
        -------
        TYPE
            Description
        """
        if self.isNotShowing.get() == True:
            self.isNotShowing.set(False)
            self.container_frame.pack(fill=BOTH, expand=True)
            self.toggle_button.configure(text='- Console -')
            self.pack(fill=BOTH, expand=True)
        else:
            self.isNotShowing.set(True)
            self.container_frame.forget()
            self.toggle_button.configure(text='+ Console +')
            self.pack(fill=None, expand=False)
            pass
        pass

    pass  # end of ConsoleWidget
