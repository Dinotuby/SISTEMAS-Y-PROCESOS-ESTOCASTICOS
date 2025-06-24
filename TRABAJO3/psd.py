# ===========================================
# MODELO PCM - UNIPOLAR NRZ CON ANÁLISIS DE ERGODICIDAD Y PSD
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate
from scipy.fft import fft, fftfreq, fftshift
# -------------------------------
# 1. PARÁMETROS GENERALES
# -------------------------------
A = 1                               # Amplitud del pulso rectangular
Tb = 1e-3                           # Tiempo de bit (1 ms)
fs = 1e5                            # Frecuencia de muestreo (100 kHz)
N = 1000                            # Número total de bits a simular
# -------------------------------
# 2. DERIVADOS Y TIEMPO
# -------------------------------
Ts = 1 / fs                         # Periodo de muestreo
Ns = int(Tb * fs)                   # Número de muestras por bit
T_total = N * Tb                    # Duración total de la señal
t = np.arange(0, T_total, Ts)       # Vector de tiempo total para la señal
# -------------------------------
# 3. GENERACIÓN DE BITS Y SEÑAL PCM
# -------------------------------
bk = np.random.randint(0, 2, N)     # Generar bits aleatorios (0 o 1)
ak = A * bk                         # Codificación Unipolar NRZ: 0 → 0, 1 → A
s = np.repeat(ak, Ns)               # Expandir cada bit durante Ns muestras
# -------------------------------
# 4. PULSO g(t) Y SU ESPECTRO
# -------------------------------
t_g = np.linspace(-1.5*Tb, 1.5*Tb, 3*Ns)          # Tiempo para graficar g(t)
g_t = np.where((t_g >= 0) & (t_g < Tb), A, 0)     # Pulso rectangular entre 0 y Tb

f_g = np.linspace(-5 / Tb, 5 / Tb, 1000)          # Frecuencias para espectro de g(t)
G_f_mag_squared = (A * Tb)**2 * (np.sinc(f_g * Tb))**2  # Espectro teórico |G(f)|^2
f_g_norm = f_g * Tb                               # Frecuencia normalizada
G_f_mag_squared_norm = G_f_mag_squared / np.max(G_f_mag_squared)  #Magnitud Normalizada
# Visualizar primeros 50 bits
t_vis = t[:50 * Ns]
s_vis = s[:50 * Ns]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
#Realización de s(t)
axs[0].plot(t_vis * 1e3, s_vis, drawstyle='steps-post')  # Señal PCM
axs[0].set_title("Realización del proceso estocástico $s(t)$")
axs[0].set_ylabel("Amplitud")
axs[0].set_xlabel("Tiempo [ms]")
axs[0].grid()
# Pulso base g(t)
axs[1].plot(t_g * 1e3, g_t, drawstyle='steps-post', color='darkgreen')  # Pulso base
axs[1].set_title("Pulso base $g(t)$ (respuesta al impulso)")
axs[1].set_ylabel("Amplitud")
axs[1].set_xlabel("Tiempo [ms]")
axs[1].grid()
#Espectro |G(f)|² (normalizado)
axs[2].plot(f_g_norm, G_f_mag_squared_norm, color='blue')  # Espectro normalizado |G(f)|²
axs[2].set_title("Espectro del pulso base $|G(f)|^2$ (normalizado)")
axs[2].set_ylabel("PSD")
axs[2].set_xlabel("Frecuencia normalizada [f·Tb]")
axs[2].grid()
axs[2].set_xlim(-5, 5)
axs[2].set_ylim(0, 1.1)
plt.tight_layout()
plt.show()
# -------------------------------
# 5. MEDIAS: TEÓRICA Y TEMPORAL
# -------------------------------
media_teorica = A / 2              # Media teórica (esperanza de ak)
media_tiempo = np.mean(s)          # Media temporal sobre s(t)
print(f"Media teórica: {media_teorica:.3f} ") #Mostrar en terminal
print(f"Media temporal {media_tiempo:.3f} ")
# -------------------------------
# 6. AUTOCORRELACIONES
# -------------------------------
# Teórica: triangular, centrada en tau=0
tau_teo = np.linspace(-5*Tb, 5*Tb, 1000)    #vector de retardos (τ) entre -5·Tb y +5·Tb, con 1000 puntos
R_teo = np.where(np.abs(tau_teo) <= Tb,0.5 * A**2 * (Tb - np.abs(tau_teo)),0)   # R_ss(τ) = 0.5 * ∫ g(u)·g(u+τ) du,triang decre  τ = 0
# Temporal: promedio de productos desplazados
max_tau = int(5 * Tb * fs)  # máximo desfase en muestras, convertimos 5·Tb a número de muestras (τ en pasos de muestreo)
taus = np.arange(-max_tau, max_tau + 1) #vector de desfases enteros (tau) desde -max_tau hasta +max_tau
R_temporal_avg = []                     # Lista pa almacenar la Rss.temp para c/tau

for tau in taus:                # Calculamos Rss.temp promediada para c/tau
    if tau >= 0:                # Desplaza -->
        s1 = s[:len(s) - tau]   # segmento sin los últimos 'tau' valores
        s2 = s[tau:]            # segmento sin los primeros 'tau' valores
    else:                       # Desplaza <--
        s1 = s[-tau:]           # eliminamos primeros |tau| valores
        s2 = s[:len(s) + tau]   # eliminamos últimos |tau| valores
    R_temporal_avg.append(np.mean(s1 * s2)) # Prom del producto punto entre los segmentos
R_temporal_avg = np.array(R_temporal_avg)       #Lista --> Arreglo Numpy
lags_tau = taus * Ts          # c/desfase  * Ts (Vector Tiempos retardo [s])
lags_tau_ms = lags_tau * 1e3  #[s] --> [ms] para eje τ 
tau_teo_ms = tau_teo * 1e3    #Retardos téorics --> [ms]

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
# Comparación de medias 
axs[0].axhline(media_teorica, color='r', linestyle='-', label='Media teórica')
axs[0].axhline(media_tiempo, color='g', linestyle='-', label='Media temporal')
axs[0].set_title("Comparación de medias (amplitud promedio)")
axs[0].set_ylim([0.48, 0.53])
axs[0].set_xlim([-15, 15])
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud promedio")
axs[0].legend()
axs[0].grid()
# Autocorrelación teórica vs temporal
axs[1].plot(lags_tau_ms, R_temporal_avg / np.max(R_temporal_avg), label='Autocorrelación Temporal')
axs[1].plot(tau_teo_ms, R_teo / np.max(R_teo), 'r--', label='Autocorrelación Teórica')
axs[1].set_title('Comparación: Autocorrelación Temporal vs Teórica (\u00b15Tb)')
axs[1].set_xlabel('Retardo [ms]')
axs[1].set_ylabel('Autocorrelación Normalizada')
axs[1].set_xlim(-5, 5)
axs[1].legend()
axs[1].grid()
plt.tight_layout()
plt.show()
# -------------------------------
# 7. PSD: INDIRECTA DESDE R_teo
# -------------------------------
S_ind = np.abs(fftshift(fft(R_teo)))          # FFT de la autocorrelación
f_ind = fftshift(fftfreq(len(R_teo), Ts))     # Frecuencias correspondientes
f_ind_norm = f_ind * Tb                       # Frecuencia normalizada
S_ind_norm = S_ind / np.max(S_ind)            # PSD normalizada
# -------------------------------
# 8. PSD: DIRECTA (PERIODOGRAMA)
# -------------------------------
S_f = fft(s)                                   # Transformada de la señal/señal s
S_dir = np.abs(S_f)**2 / (len(s) * fs)         # PSD directa normalizada(periodograma)
f_dir = fftshift(fftfreq(len(S_f), Ts))        # Frecuencias centras
f_dir_norm = f_dir * Tb                        # y Normalizadas
S_dir_norm = fftshift(S_dir) / np.max(S_dir)   # PSD normalizada pa comparar
# -------------------------------
# 9. PSD: ANALÍTICA
# -------------------------------
f_analitico = np.linspace(-5 / Tb, 5 / Tb, 1000)            #Vect freq Hz (-5Tb,5Tb)
f_norm = f_analitico * Tb                                   #Normalizamos f·Tb (eje normalizado)
S_continua = (A**2 / Tb) * (np.sinc(f_analitico * Tb))**2   #Componente continua de :(A² / Tb) · sinc²(f·Tb)
S_continua_norm = S_continua / np.max(S_continua)           #Normalizamos PSD para la gráfica
fig, axs = plt.subplots(3, 1, figsize=(10, 9))
# PSD Indirecta
axs[0].plot(f_ind_norm, S_ind_norm, label='PSD Indirecta')
axs[0].axvline(-1, color='orange', linestyle='--', label=r'Límites $\pm 1/T_b$')
axs[0].axvline(1, color='orange', linestyle='--')
axs[0].set_title("PSD Indirecta (normalizada)")
axs[0].set_xlim(-5, 5)
axs[0].set_ylabel("PSD")
axs[0].grid()
axs[0].legend()
#PSD Directa
axs[1].plot(f_dir_norm, S_dir_norm, label='PSD Directa')
axs[1].axvline(-1, color='orange', linestyle='--', label=r'Límites $\pm 1/T_b$')
axs[1].axvline(1, color='orange', linestyle='--')
axs[1].set_title("PSD Directa (normalizada)")
axs[1].set_xlim(-5, 5)
axs[1].set_ylabel("PSD")
axs[1].grid()
axs[1].legend()
#PSD Analítica
axs[2].plot(f_norm, S_continua_norm, label=r'$|G(f)|^2/T_b$', color='blue')
axs[2].axvline(0, color='red', linestyle='--', ymax=0.9, label=r'Impulso en $f=0$')
axs[2].axvline(-1, color='orange', linestyle='--', label=r'Límites $\pm 1/T_b$')
axs[2].axvline(1, color='orange', linestyle='--')
axs[2].set_title("PSD Analítica: sinc$^2$ + impulso en $f=0$")
axs[2].set_xlabel("Frecuencia normalizada [f·Tb]")
axs[2].set_ylabel("PSD")
axs[2].set_xlim(-5, 5)
axs[2].set_ylim(0, 1.1)
axs[2].grid()
axs[2].legend()
plt.tight_layout()
plt.show()
# -------------------------------
# 10. ANCHO DE BANDA APROXIMADO
# -------------------------------
BW = (2 / Tb)/2  # BW ≈ 1/Tb  para dos dado que bandwith solo positivo
print(f"Ancho de banda efectivo aproximado: {BW:.2f} Hz (desde -1/Tb hasta 1/Tb)")
