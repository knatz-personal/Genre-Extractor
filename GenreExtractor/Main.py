import sys
import os
import warnings

from GUI import MainForm
from Utilities import LogUtils, FileManager
from tkinter import Tk
from Classification import GenreClassifier

_author__ = "Nathan Zwelibanzi Khupe"
__copyright__ = "Copyright 2017, Nathan Zwelibanzi Khupe"
__credits__ = ["Nathan Zwelibanzi Khupe"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nathan Zwelibanzi Khupe"
__email__ = "001knatz@gmail.com"
__docformat__ = 'numpy'


def main(argv):
    """
    MAIN program entry point
    """
    warnings.filterwarnings("ignore")
    LogUtils(False)

    LogUtils.write("info", "Application Start...")
    root = Tk()

    dir = os.path.dirname(__file__)
    MainForm(root, dir)

    root.mainloop()

    #directory = "C:\\Users\\Natha_000\\Desktop\\Test Stories"
    #for filename in os.listdir(directory):
    #    if filename.endswith(".txt"):
    #        path = os.path.join(directory, filename)
    #        if os.path.exists(path):
    #            with open(path, 'rb') as content_file:
    #                file = content_file.read()
    #                result = GenreClassifier.predict_SubGenre(file.strip())
    #                print(filename, result)
    #            pass
    #        continue
    #    else:
    #        continue
    #    pass
    #print("Et fini...")

    pass

if __name__ == '__main__':
    main(sys.argv)
    LogUtils.write("info", "...Application Exit")
    sys.exit()
    pass
