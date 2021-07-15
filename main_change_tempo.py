# imports
import pyaudio                                                  # audio I/O functions
import librosa                                                  # audio processing package
import librosa.display                                          # audio display (chroma, spec...)
import pretty_midi                                              # midi processing package
import numpy as np                                              # array functions
import timeit
import time
import PySimpleGUI as sg                                        # GUI package
import matplotlib                                               # plotting functions
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter
matplotlib.use('TkAgg')
np.set_printoptions(threshold=np.inf)                           # Print the whole arrays


# Frequency BandPass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# AudioHandler object for audio input
class AudioHandler(object):
    global buffer_size
    global global_rate

    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = global_rate
        self.CHUNK = buffer_size
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=global_rate,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=buffer_size)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global numpy_array
        global sgi_max
        sm_array = np.frombuffer(in_data, dtype=np.float32)
        small_array = butter_bandpass_filter(sm_array, 30.0, 4000.0, global_rate, order=6) # Band Pass Filter
        small_array[abs(small_array) <= sgi_max] = 0 # Cut low values (background noise)
        print('Numpy size:', numpy_array.size)
        print('Small size:', small_array.size)
        arrays = np.array_split(numpy_array, [buffer_size])
        tup_arrays = (arrays[1], small_array)
        np.concatenate(tup_arrays, out=numpy_array)
        return None, pyaudio.paContinue

    def get_a_spec(self):
        global fig_photo
        global numpy_array
        global fig
        global tk_canvas
        global chroma
        global midi_tempo
        global tempo
        if numpy_array is not None:
            if fig_photo is not None:
                delete_figure_agg(fig_photo)
            chroma = librosa.feature.chroma_cqt(y=numpy_array, sr=self.RATE, hop_length=512)
            tempo = int(librosa.beat.tempo(y=numpy_array, sr=self.RATE, hop_length=512, start_bpm=midi_tempo[1][0])[0])
            print(tempo)
            #listen_chroma_txt = open("listen_chroma_txt.txt", 'w')
            #for element in chroma:
            #    print(element, file=listen_chroma_txt)
            #listen_chroma_txt.close()
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='s', sr=self.RATE)
            fig = plt.gcf()
            fig_photo = draw_figure(tk_canvas, fig)

# AudioHandler object for metering background noise
class AudioHandlerStart(object):
    global buffer_size
    global global_rate
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = global_rate
        self.CHUNK = buffer_size
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global sgi_max
        sm_array = np.frombuffer(in_data, dtype=np.float32)
        sgi_max2 = np.max(np.abs(sm_array))
        if sgi_max2 > sgi_max:
            sgi_max = sgi_max2
        print(sgi_max)
        return None, pyaudio.paContinue


# Draw and delete figures from canvas functions
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.clf()

# Get midi chroma and midi tempo from midi file
def get_midi_chroma_tempo(midi_url):
    # load MIDI file
    midi_file = pretty_midi.PrettyMIDI(midi_url)

    # Get Midi Tempo Changes and values
    midi_tempo = midi_file.get_tempo_changes()

    # Get Midi Chroma
    midi_chroma = midi_file.get_chroma()
    midi_chroma = np.true_divide(midi_chroma, np.max(midi_chroma))
    midi_chroma = np.hsplit(midi_chroma, 4)
    midi_chroma_txt = open("midi_chroma_txt.txt", 'w')
    for element in midi_chroma[0]:
        print(element, file=midi_chroma_txt)
    midi_chroma_txt.close()

    return midi_chroma, midi_tempo

# Get the number of line (nota) with max sum in a chroma array
def get_nota_max(chroma_array):
    return np.argmax(np.sum(chroma_array, axis=1))

# Dynamic Time Wraping algorithm, returns array D and path P
def get_dtw_D_P(ar1, ar2):
    D, P = librosa.sequence.dtw(np.array(ar1), np.array(ar2), metric='euclidean', step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]), weights_mul=np.array([1, 1, 1]), global_constraints=True, band_rad=0.20)
    return D, P

def reinitialize_buffer():
    global beat_size
    global buffer_size
    global tempo
    global global_rate
    beat_size = 60 / tempo
    buffer_size = int(global_rate * beat_size)
    numpy_array.resize([buffer_size * 16])


# load midi file properties
midi_chroma, midi_tempo = get_midi_chroma_tempo('test.mid')
nota_max = get_nota_max(midi_chroma[0])

# initialize global variables
midi_length = len(midi_chroma[0][0])
global_rate = midi_length * 32
print(global_rate)
tempo = midi_tempo[1][0]
beat_size = 60 / tempo
buffer_size = int(global_rate * beat_size)
listening = False
numpy_array = np.zeros(buffer_size * 16, dtype=np.float32)
chroma = None
#fig_photo = None  declared afterwards
#fig = None
midi_fig_photo = None
midi_fig = None
sgi_max = 0.0

# initialize audio handlers
audio = AudioHandler()
audio_start = AudioHandlerStart()

# initialize GUI
button_on = sg.Button('On', disabled=False)
button_off = sg.Button('Off', disabled=True)
DTW_distance = sg.Text('0%')
audio_canvas = sg.Canvas(key='Canvas', size=(640, 480), background_color='white')
layout = [[sg.Text('Listening :'), button_on, button_off, sg.Txt('DTW distance :'), DTW_distance], [audio_canvas]]
main_window = sg.Window('Record Test', layout, finalize=True, margins=(100, 30))
main_window.finalize()
tk_canvas = audio_canvas.TKCanvas

# Draw midi chroma
librosa.display.specshow(midi_chroma[0], y_axis='chroma', x_axis='s', sr=global_rate*2)
fig = plt.gcf()
fig_photo = draw_figure(tk_canvas, fig)

# Start background noise audio handler
audio_start.start()

old_tempo = tempo

# main loop
while True:
    event, values = main_window.read(timeout=20000/old_tempo, timeout_key='Timeout')
    if listening and ((tempo < old_tempo * 0.8) or (tempo > old_tempo * 1.8)):
            audio.stop()
            time.sleep(0.5)
            old_tempo = tempo
            print('tempo changed')
            reinitialize_buffer()
            audio.start()
    if event == 'On' and not listening:
        audio_start.stop() # Stop background noise audio handler
        DTW_distance.update(str(sgi_max)) # Show max value of background noise
        audio.start()
        listening = True
        button_on.update(disabled=True)
        button_off.update(disabled=False)
    elif event == 'Off':
        button_on.update(disabled=False)
        button_off.update(disabled=True)
        listening = False
        audio.stop()
    elif event == 'Timeout' and listening:
        audio.get_a_spec()
        tic = timeit.default_timer()
        D, P = get_dtw_D_P(midi_chroma[0], chroma)
        toc = timeit.default_timer()
        print(D[-1, -1])  #/ len(midi_chroma[0][0]))
    if event != sg.WIN_CLOSED:
        continue
    break

main_window.close()
