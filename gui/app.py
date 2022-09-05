import eel
import tkinter
import tkinter.filedialog as filedialog

eel.init('dist')


@eel.expose
def hello_world():
    return "Hello from python blabla"


@eel.expose
def print_string(string):
    if len(string) > 20:
        print(string)
        return "Success!"
    else:
        return "Too few characters. Please type more than 20 characters."


@eel.expose
def select_folder():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    directory_path = filedialog.askdirectory()
    print(directory_path)
    return directory_path


eel.start('index.html', size=(600, 400), port=8080)
