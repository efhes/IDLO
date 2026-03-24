# Exercises in order to perform laboratory work

# Import of modules
import os
import time
import subprocess
import numpy as np
import scipy.signal
from scipy.signal import convolve
from common.dataprep import md5
from skimage.morphology import opening, closing

import requests
import os
import hashlib
from tqdm.notebook import tqdm  # Importamos tqdm.notebook
import time

def download_dataset_UPM(url, save_path, reload=False):
    """
    Download the test part of the VoxCeleb dataset with requests and show progress in Colab.
    :url: url of the dataset to download
    :param save_path: path to folder
    :param reload: rewrite if file exists
    """

    outfile = 'vox1_test_wav.zip'
    filepath = os.path.join(save_path, outfile)
    md5gt = '185fdc63c3c739954633d50379a3d102'

    if not os.path.exists(filepath) or reload:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with open(filepath, 'wb') as file, tqdm(
                desc=outfile,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

        except requests.exceptions.RequestException as e:
            print(f"Download failed for {outfile}: {e}")
            time.sleep(120)
            return

    # Check MD5
    md5ck = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
    if md5ck == md5gt:
        print(f'Checksum successful {outfile}.')
    else:
        raise Warning(f'Checksum failed {outfile}.')

def load_vad_markup(path_to_rttm, signal, fs):
    ''' 
    Function to read rttm files and generate VAD's markup in samples
    
    The function takes three arguments:
    path_to_rttm: a string representing the path to the RTTM file containing the VAD information.
    signal: a 1-dimensional numpy array representing the audio signal.
    fs: a scalar representing the sampling frequency of the audio signal.
    The function reads the RTTM file and extracts the VAD information for each segment of the audio signal.
    It then generates a VAD markup array with the same length as the audio signal, 
    where each sample is either 0.0 (not voiced) or 1.0 (voiced).
    '''

    vad_markup = np.zeros(len(signal)).astype('float32')

    # Read rttm file
    with open(path_to_rttm, 'r') as f:
        for line in f:
            line_parts = line.strip().split()
            if line_parts[0] == 'SPEAKER': #and line_parts[7] == 'vad':
                start_time = float(line_parts[3])
                end_time = start_time + float(line_parts[4])
                start_sample = int(start_time * fs)
                end_sample = int(end_time * fs)
                vad_markup[start_sample:end_sample] = 1.0
    
    print(f"DEBUG: {os.path.basename(path_to_rttm)} | Voz en etiquetas: {np.mean(vad_markup)*100:.1f}%")
    return vad_markup

def framing(signal, window=320, shift=160):
    ''' 
    Function to create frames from signal
    
    Se utiliza un bucle for para aplica el enventanado a la señal de entrada. 
    El parámetro "window" especifica el tamaño de cada ventana, 
    mientras que el parámetro "shift" especifica el desplazamiento entre ventanas adyacentes.
    Dentro del bucle for, se utiliza el índice "i" para calcular la posición de inicio y finalización de cada ventana en la señal de entrada, 
    utilizando la fórmula i*shift e i*shift+window, respectivamente. 
    Estos índices se utilizan para extraer la porción correspondiente de la señal de entrada y almacenar la ventana correspondiente en la matriz "frames".
    Finalmente, la matriz "frames" se devuelve como resultado de la función.
    Nótese que se usa la variable "shape" para crear una matriz de ceros con la forma correcta antes de llenarla con las diferentes ventanas.
    ''' 
    
    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    for i in range(shape[0]):
        frames[i] = signal[i*shift:i*shift+window]
        
    return frames

def frame_energy(frames):
    '''
    Function to compute frame energies
     
    Se utiliza un bucle for para calcular la energía de cada ventana. 
    El parámetro "frames" es una matriz de dos dimensiones que contiene las ventanas de la señal de entrada, 
    donde cada fila representa un ventana y cada columna representa una muestra dentro de la ventana.
    Dentro del bucle for, se utiliza el índice "i" para acceder a cada ventana en la matriz de entrada. 
    Luego, se utiliza la función np.square para elevar al cuadrado cada muestra dentro de la ventana, 
    y la función np.sum para sumar los resultados de estas operaciones para todas las muestras dentro de la ventana. 
    El resultado se almacena en el elemento correspondiente de la matriz "E", que contiene las energías de cada ventana.
    Finalmente, la matriz "E" se devuelve como resultado de la función. 
    Nótese que la variable "E" se inicializa con una matriz de ceros con la misma forma que el número de ventanas en la matriz "frames" antes de llenarla con los valores de energía.
    '''
    
    E = np.zeros(frames.shape[0]).astype('float32')

    for i in range(frames.shape[0]):
        E[i] = np.sum(np.square(frames[i]))

    return E

def norm_energy(E):
    '''
    Function to normalize energy by mean energy and energy standard deviation
    
    Se calcula primero la media y la desviación estándar de las energías de las ventanas.
    Esto se realiza mediante las funciones np.mean y np.std, respectivamente.
    Luego, se normalizan las energías de las ventanas mediante la siguiente fórmula: 
    E_norm = (E - E_mean) / E_std, 
    donde "E" es la matriz de energías de entrada, 
    "E_mean" es la media de las energías y "E_std" es la desviación estándar de las energías.
    Finalmente, la matriz "E_norm" se devuelve como resultado de la función.
    '''
    
    E_mean = np.mean(E)
    E_std = np.std(E)
    E_norm = (E - E_mean) / E_std

    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    ''' 
    Function to train parameters of gaussian mixture model
    
    Se inicializan los parámetros del modelo de mezcla de gaussianas con valores predeterminados.
    En la variable "w" se almacenan las ponderaciones de cada componente de la mezcla, 
    en "m" se almacenan las medias de cada componente y en "sigma" se almacenan las desviaciones estándar de cada componente.
    Luego, se utiliza un bucle for para iterar sobre el número de realineamientos especificado en el parámetro "n_realignment".
    Dentro de este bucle, se realiza el E-step y el M-step para actualizar los parámetros del modelo.
    En el E-step, se calcula la probabilidad de que cada ventana pertenezca a cada componente de la mezcla de gaussianas.
    Para hacer esto, se utiliza una función "gauss_pdf" que devuelve la densidad de probabilidad de una distribución normal para un valor dado de x, una media mu y una desviación estándar sigma.
    La matriz "g" se utiliza para almacenar estas probabilidades para cada ventana y componente.
    En el M-step, se actualizan los parámetros del modelo utilizando las probabilidades calculadas en el E-step.
    Se actualizan las ponderaciones de cada componente de la mezcla, las medias de cada componente y las desviaciones estándar de cada componente. 
    Esto se realiza utilizando fórmulas basadas en la regla de máxima verosimilitud.
    Finalmente, se devuelven los parámetros actualizados del modelo como resultado de la función.
    '''

    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):

        # E-step
        for j in range(len(w)):
            g[:,j] = gauss_pdf(E, m[j], sigma[j]) * w[j]
        g_sum = np.sum(g, axis=1)
        for j in range(len(w)):
            g[:,j] = g[:,j] / g_sum

        # M-step
        for j in range(len(w)):
            w[j] = np.mean(g[:,j])
            m[j] = np.sum(g[:,j] * E) / np.sum(g[:,j])
            sigma[j] = np.sqrt(np.sum(g[:,j] * np.square(E - m[j])) / np.sum(g[:,j]))

    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    ''' 
    Function to estimate a posterior probability that frame IS speech
    
    Se calcula la probabilidad de que cada ventana no contenga voz.
    Para hacer esto, se utiliza una función "gauss_pdf" que devuelve la densidad de probabilidad de una distribución normal para un valor dado de x, una media mu y una desviación estándar sigma.
    Los parámetros del modelo de mezcla de gaussianas (ponderaciones, medias y desviaciones estándar) se pasan como argumentos a la función.
    Dentro del bucle for, se calcula la probabilidad de que cada ventana pertenezca a cada componente de la mezcla de gaussianas utilizando la función "gauss_pdf", 
    y se multiplican estas probabilidades por las ponderaciones correspondientes.
    El resultado se almacena en la variable "g0".
    Finalmente, se calcula la probabilidad de que cada marco contenga voz restando "g0" de 1.
    El resultado se almacena en la variable "g1" y se devuelve como resultado de la función.
    '''
    
    g0 = np.zeros(len(E))

    for j in range(len(w)):
        g0 += gauss_pdf(E, m[j], sigma[j]) * w[j]

    g1 = np.ones(len(E)) - g0

    return g1

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Calculamos la probabilidad a posteriori de que SEA voz (0.0 a 1.0)
    # Ahora g1 = 1 significa "Voz con total seguridad"
    g1 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g1 > vad_thr).astype('float32')  # frame VAD's markup

    # Repite cada decisión del frame 'shift' veces y recorta al largo de la señal original
    vad_markup_real = np.repeat(vad_frame_markup_real, shift)[:len(signal)]

    # Si la señal es un poco más larga que los frames procesados, rellena con la última decisión
    if len(vad_markup_real) < len(signal):
        padding = np.full(len(signal) - len(vad_markup_real), vad_frame_markup_real[-1])
        vad_markup_real = np.concatenate([vad_markup_real, padding])
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    ''' 
    Function to create reverberation effect
    
    La función utiliza la función "convolve" de SciPy para convolucionar la señal de entrada con la respuesta al impulso.
    Esto simula el efecto de la reverberación.
    La variable "signal_reverb" almacena el resultado de la convolución, 
    pero se acorta a la longitud de la señal de entrada para que tengan la misma duración.
    El resultado se devuelve como salida de la función.
    Nótese que la respuesta al impulso debe estar en la misma escala de amplitud que la señal de entrada.
    Si la respuesta al impulso tiene una amplitud máxima diferente de 1, 
    es posible que deba normalizarse antes de aplicarla a la señal de entrada.
    '''
    
    signal_reverb = convolve(signal, impulse_response)
    
    return signal_reverb[:len(signal)]

def awgn(signal, sigma_noise):
    '''
    Function to add white gaussian noise to signal
    
    La función utiliza la función "normal" de NumPy para generar una señal de ruido blanco gaussiano con una desviación estándar de "sigma_noise".
    La señal de ruido se agrega a la señal de entrada para crear la señal ruidosa resultante.
    El resultado se devuelve como salida de la función.
    Nótese que la señal de ruido debe ser de la misma longitud que la señal de entrada y tener la misma escala de amplitud.
    Si la señal de entrada está normalizada a un rango de [-1, 1], la señal de ruido debe ser generada en el mismo rango.
    '''
    
    noise = np.random.normal(0, sigma_noise, size=len(signal))
    signal_noise = signal + noise
    
    return signal_noise