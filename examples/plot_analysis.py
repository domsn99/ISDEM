# Importieren Sie die benötigten Module
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from modules import corr
from tkinter import filedialog

# Erstellen Sie ein Tkinter-Fenster
window = tk.Tk()
window.title("Bereich auswählen")

hot_pixels = np.array([(116,199),(103,198),(76,166),(55,203),(121,97),(249,56),(116,199),(197,223)])

def change_directory():
    global path, data_path, name
    # Eingeben des Datenpfades
    path = filedialog.askdirectory()
    data_path = Path(path)
    name = data_path.name
    replot()

# Erstellen Sie eine Funktion, um die Koordinaten und Datenpunkte des ausgewählten Bereichs zu erhalten
def onselect(eclick, erelease):

    # eclick und erelease sind die Ereignisse an den Mausklick- und Freigabepositionen
    # Extrahieren Sie die Koordinaten des ausgewählten Bereichs
    global x1, x2, y1, y2
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Drucken Sie die Koordinaten und Datenpunkte aus
    print(f"Koordinaten: ({x1}, {y1}), ({x2}, {y2})")


# Erstellen Sie eine Funktion, um den ausgewählten Bereich zu bestätigen und eine andere Funktion auszuführen
def confirm():

    # Führen Sie eine andere Funktion aus (hier nur ein Beispiel)
    print("Bereich bestätigt")
    show_image()


# Erstellen Sie eine Funktion, um den ausgewählten Bereich zurückzusetzen und einen neuen Bereich zu wählen
def replot():

    global fig, canvas, index_coin, data_bincoin, data, hot_pixels, data_raw, binning, offset

    try:
        canvas.get_tk_widget().destroy()
    except: pass

    print(binning_box.get(), offset_box.get())
    offset = int(offset_box.get())
    binning = int(binning_box.get())

    analysis = corr.coin(path, name)

    index_coin, data_raw, data_bincoin = analysis.coincidences(offset, binning)
    data_bincoin[hot_pixels[:,1],hot_pixels[:,0]] = 0
    data_raw[hot_pixels[:,1],hot_pixels[:,0]] = 0
    # data_bincoin[56,249]=500
    data=data_bincoin

    # Erstellen Sie eine neue Abbildung
    fig = plt.figure(figsize=(8,8))
    fig.clear()
    if int(name.isdigit()): plt.title(data_path.parent.name+'\\'+name)
    else: plt.title(name)

    canvas = FigureCanvasTkAgg (fig, master=window)

    # Zeigen Sie das Bild mit plt.imshow an

    plt.imshow(data,origin='lower',cmap='hot',norm='linear')
    # plt.xlim([50,200])
    # plt.ylim([50,200])

    canvas.get_tk_widget ().pack ()

    # Aktivieren Sie den Rechteckauswähler mit der onselect-Funktion
    rs = RectangleSelector(fig.gca(), onselect, interactive=True)
   
    plt.close()

    window.mainloop()
    

# Erstellen Sie eine Funktion, um das Bild anzuzeigen und den Rechteckauswähler zu aktivieren
def show_image():

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    if name.isdigit(): fig.suptitle(data_path.parent.name+'\\'+name)
    else: fig.suptitle(name)

    # Erstellen Sie eine Leinwand, um die Figur im Fenster anzuzeigen
    canvas = FigureCanvasTkAgg (fig, window)
    data_matrix = np.load(data_path / "out_xy.npy")
    data_toa = np.load(data_path / "out_toa.npy")/10**13
    data_ph = np.load(data_path / 'out_ph.npy')

    # Creating matrix list
    xrange = np.arange(x1,x2+1)
    yrange = np.arange(y1,y2+1)
    matrix_list = np.array([i*256 + j for i in xrange for j in yrange])
    coin = index_coin[np.isin(data_matrix[index_coin],matrix_list)]
    selected_data = data[y1:y2+1, x1:x2+1]

    # Creating radial profile
    center = np.unravel_index(np.argmax(selected_data),selected_data.shape)

    # Checks data type. jpg and png have shape (y,x,misc)
    y, x = np.indices((selected_data.shape))

    # Creates an array of distances from the centre to the set radius of the profile. 
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Convert array data to integer.
    r = r.astype(int)

    # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
    tbin = np.bincount(r.ravel(), selected_data.ravel())
    # Bincounting for normalizing the profile.
    nr = np.bincount(r.ravel())

    # Normalize radial profile and return.
    circularprofile_raw = tbin / nr

    # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
    tbin = np.bincount(r.ravel(), selected_data.ravel())
    # Bincounting for normalizing the profile.
    nr = np.bincount(r.ravel())

    # Normalize radial profile and return.
    circularprofile_coin = tbin / nr

    # Customizing the plot.

    #plt.plot(circularprofile[:radi], label=name)

    # Averaging of the plot.
    avg_y=[]
    for k in range(len(circularprofile_coin)-4+1):
        avg_y.append(np.mean(circularprofile_coin[k:k+4]))

    radi = 50
    circularprofile_coin = avg_y[0:radi]
    xscale = np.array(range(0,radi,10))
    scale = 0.1615

    # Plot the histogram
    ax[0][0].set_title('{} electrons \n {} photons \n {} coincidences'.format(data_toa.shape[0], data_ph.shape[0], len(coin)))
    im = ax[0][0].imshow(selected_data,cmap='hot',origin='lower')
    fig.colorbar(im)

    ax[1][0].set_title(r'Coincidence Profile at ({},{})'.format(center[0],center[1]))
    ax[1][0].plot(circularprofile_coin)
    ax[1][0].set_yscale('log')
    ax[1][0].set_xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
    ax[1][0].set_xlabel("Radius [urad]")
    ax[1][0].set_ylabel("Counts")


    ax[1][1].set_title('ToA during Measurement')
    ax[1][1].hist(data_toa[coin],bins=200)
    ax[1][1].set_xlabel("Measurement time [s]")
    ax[1][1].set_ylabel("Counts/bin")


    ax[0][1].set_title('Raw data'.format(data_toa.shape[0]))
    im = ax[0][1].imshow(data_raw,cmap='hot',origin='lower')
    fig.colorbar(im)


    plt.show()

def dead_pixel():
    global hot_pixels
    np.concatenate(hot_pixels, (x1,y1))
    replot()

# Datei ändern
dir_button = tk.Button(window, text="Datensatz ändern", command=change_directory)
dir_button.pack()

# Erstellen von Textboxen
binning_box = tk.Entry(window)
binning_box.insert(0,"20")
label1 = tk.Label(window, text="Binning")
label1.pack()
binning_box.pack()

offset_box = tk.Entry(window)
offset_box.insert(0,"0")
label2 = tk.Label(window, text="Offset")
label2.pack()
offset_box.pack()

# Erstellen Sie einen Button, um den ausgewählten Bereich zurückzusetzen
reset_button = tk.Button(window, text="Replot", command=replot)
reset_button.pack()

# Erstellen Sie einen Button, um den ausgewählten Bereich zurückzusetzen
reset_button = tk.Button(window, text="Dead Pixel", command=dead_pixel)
reset_button.pack()


# Erstellen Sie einen Button, um den ausgewählten Bereich zu bestätigen
confirm_button = tk.Button(window, text="Bereich bestätigen", command=confirm)
confirm_button.pack()

window.mainloop()