# imports
import pyaudio  # audio I/O functions
import librosa  # audio processing package
import librosa.display  # audio display (chroma, spec...)
import pretty_midi  # midi processing package
import numpy as np  # array functions
import PySimpleGUI as sg  # GUI package
import matplotlib  # plotting functions
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter

matplotlib.use('TkAgg')
np.set_printoptions(threshold=np.inf)  # Print the whole arrays


# Frequency BandPass Filter
def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
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
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global numpy_array
        global sgi_max
        sm_array = np.frombuffer(in_data, dtype=np.float32)  # get audio data
        small_array = butter_bandpass_filter(sm_array, 30.0, 4200.0, global_rate, order=6)  # Band Pass Filter
        # Adjust new buffer to the whole array in memory
        arrays = np.array_split(numpy_array, [buffer_size])
        tup_arrays = (arrays[1], small_array)
        np.concatenate(tup_arrays, out=numpy_array)
        return None, pyaudio.paContinue

    def get_a_spec(self):
        # GLOBAL VARIABLES
        global fig_photo
        global numpy_array
        global fig
        global tk_canvas

        fnd = False
        position = 0

        if numpy_array is not None:  # Check if buffer memory is empty
            delete_figure_agg(fig_photo, fig)  # Erase previous plot
            # Get chromagram from audio memory
            chroma = librosa.feature.chroma_cqt(y=numpy_array, threshold=sgi_max, sr=self.RATE, hop_length=512)
            d, p, fnd, position = get_dtw_results(chroma, midi_chroma)  # get dtw results

            # Display similarity plot
            fig, ax = plt.subplots(nrows=2, sharex='none')
            p_s = librosa.frames_to_time(p, sr=global_rate, hop_length=512)
            img = librosa.display.specshow(d, vmin=0, vmax=100, y_axis='time', x_axis='time', sr=global_rate, ax=ax[0])
            ax[0].plot(p_s[:, 1], p_s[:, 0], color='y')
            fig.colorbar(img, ax=ax[0])
            img2 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=global_rate, ax=ax[1])
            fig.colorbar(img2, ax=ax[1])
            fig_photo = draw_figure(tk_canvas, fig)

        return fnd, position


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
        sm_array = np.frombuffer(in_data, dtype=np.float32)  # get audio data from buffer
        sgi_max2 = np.max(np.abs(sm_array))  # get max absolute value
        if sgi_max2 > sgi_max:
            sgi_max = sgi_max2   # from all max values keep the greatest
        return None, pyaudio.paContinue


# Draw and delete figures from canvas functions
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas_agg.flush_events()
    return figure_canvas_agg


def delete_figure_agg(figure_agg, figure):
    figure_agg.get_tk_widget().forget()
    plt.close(figure)
    plt.clf()
    return figure_agg


# Get midi tempo and number of beats from midi file
def get_midi_tempo_beats(midi_url):
    # load MIDI file
    midi_file = pretty_midi.PrettyMIDI(midi_url)

    # Get Midi Tempo Changes and values (accept that there are no tempo changes)
    midi_tempo = midi_file.get_tempo_changes()
    beats = midi_file.get_beats()
    time_sign = midi_file.time_signature_changes[0].numerator
    return midi_tempo[1][0], len(beats), time_sign


# Dynamic Time Warping algorithm, returns array D and path P
def get_dtw_results(ar1, ar2):
    d, p = librosa.sequence.dtw(X=ar1, Y=ar2, subseq=True, metric='euclidean',
                                step_sizes_sigma=[[2, 1], [1, 2], [2, 2], [3, 2], [2, 3], [3, 3]])
    # if current distance is smaller than 80% of max dist, we found position
    current_dist = d[p[0, 0], p[0, 1]]
    max_dist = d[-1, -1]
    fnd = current_dist <= 0.8 * max_dist
    # current position is where the path ends
    position = p[0, 1]
    return d, p, fnd, position


# load midi file properties and set global rate and buffer size
# # # get midi properties from midi file
tempo, midi_beats, time_signature = get_midi_tempo_beats('piano_sonata_330_1_(c)oguri.mid')
num_of_measures = midi_beats//time_signature
global_rate = 11025  # global sample rate is 11025Hz (only interested in frequencies bellow 5512Hz
beat_size = 60 / tempo  # calculate beat_size in seconds
buffer_size = int(global_rate * beat_size)  # calculate buffer_size in frames
midi_wav = librosa.load('mozart.wav', sr=global_rate, dtype=np.float32)  # load synthesized audio from midi
# get "midi" chromagram from synthesized audio
midi_chroma = librosa.feature.chroma_cqt(y=midi_wav[0], sr=global_rate, hop_length=512)
midi_length = len(midi_chroma[0])  # count midi chromagram frames
print(midi_length)

# initialize global variables
# # # make audio memory array equal to 2 measures
numpy_array = np.zeros(buffer_size * time_signature * 2, dtype=np.float32)
listening = False  # listening status
sgi_max = 0.0  # maximum value of noise set to zero

# initialize audio handlers
audio = AudioHandler()
audio_start = AudioHandlerStart()

# initialize GUI
button_on = sg.Button('On', disabled=False)
button_off = sg.Button('Off', disabled=True)
noise_volume = sg.Text('       ')
error_text = sg.Text('                               ')
error_text.TextColor = 'red'
measures_text = sg.Text(str(num_of_measures))
current_text = sg.Text('        ')
audio_canvas = sg.Canvas(key='Canvas', size=(640, 480), background_color='white')
layout = [[sg.Text('Listening :'), button_on, button_off, sg.Txt('Noise Volume:'), noise_volume],
          [sg.Txt('Total measures:'), measures_text, sg.Txt('Current measure:'), current_text, error_text],
          [audio_canvas]]
main_window = sg.Window('Page Turner', layout, finalize=True, margins=(100, 30))
main_window.finalize()
tk_canvas = audio_canvas.TKCanvas

# Draw midi chroma
librosa.display.specshow(midi_chroma, y_axis='chroma', x_axis='s', sr=global_rate)
fig = plt.gcf()
fig_photo = draw_figure(tk_canvas, fig)

# Start background noise audio handler
audio_start.start()

# main loop
while True:
    event, values = main_window.read(timeout=beat_size, timeout_key='Timeout')
    if event == 'On' and not listening:
        audio_start.stop()  # Stop background noise audio handler
        noise_volume.update(str(sgi_max))  # Show max value of background noise
        audio.start()  # start audio handler
        listening = True
        button_on.update(disabled=True)
        button_off.update(disabled=False)
    elif event == 'Off':
        button_on.update(disabled=False)
        button_off.update(disabled=True)
        listening = False
        audio.stop()  # stop audio handler
    elif event == 'Timeout' and listening:
        found, current_frame = audio.get_a_spec()  # get audio chromagram and use DTW to get position
        if not found:
            error_text.update('Error finding position')  # Show error message
            current_text.update('     ')
        else:
            error_text.update('')
            current_measure = int(round(current_frame*num_of_measures/midi_length, 0))
            current_text.update(str(current_measure))
    elif event == 'Timeout' and not listening:
        noise_volume.update(str(sgi_max))
    if event != sg.WIN_CLOSED:
        continue
    break

main_window.close()
