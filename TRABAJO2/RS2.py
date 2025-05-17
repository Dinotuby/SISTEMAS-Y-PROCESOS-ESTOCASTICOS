import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import norm

# ----------------------------
# 1. Generador LFSR
# ----------------------------
def generar_lfsr(semilla, realimentaciones, n_bits):
    sr = semilla.copy()
    salida = []
    for _ in range(n_bits):
        salida.append(sr[0])  # <-- salida correcta
        xor = 0
        for t in realimentaciones:
            xor ^= sr[-t]  # se mantiene el tap en base a posición desde el final
        sr = sr[1:] + [xor]
    return np.array(salida)


# ----------------------------
# 2. Modulador (AM)
# ----------------------------
def modular(bits, A=1):
    return np.array([A if bit == 1 else -A for bit in bits])

# ----------------------------
# 3. Canal con ruido blanco gaussiano (AWGN)
# ----------------------------
def agregar_ruido_awgn(senal, snr_db):
    snr_lineal = 10**(snr_db / 10)
    potencia_senal = np.mean(senal**2)
    potencia_ruido = potencia_senal / snr_lineal
    ruido = np.random.normal(0, np.sqrt(potencia_ruido), len(senal))
    return senal + ruido, ruido

# ----------------------------
# 4. Demodulador
# ----------------------------
def demodular(senal_recibida):
    return np.array([1 if x > 0 else 0 for x in senal_recibida])

# ----------------------------
# 5. Evaluación del sistema
# ----------------------------
def simular_sistema(m=7, realimentaciones=[7,6,4,2], semilla=[1,0,0,0,0,0,0], A=1, n_bits=10000):
    bits = generar_lfsr(semilla, realimentaciones, n_bits)
    
    print(f"Cantidad de unos: {np.sum(bits)}")
    print(f"Cantidad de ceros: {n_bits - np.sum(bits)}")
    print(f"Media de la secuencia: {np.mean(bits):.2f}")
    
    senal = modular(bits, A)

    valores_snr = list(range(0, 15, 2))
    ber_resultados = []
    errores_totales = []

    ejemplo_ruido = None
    ejemplo_decodificado = None

    for i, snr in enumerate(valores_snr):
        senal_con_ruido, ruido = agregar_ruido_awgn(senal, snr)
        decodificado = demodular(senal_con_ruido)
        errores = np.sum(bits != decodificado)
        pe = errores / n_bits

        errores_totales.append(errores)
        ber_resultados.append(pe)

        print(f"SNR: {snr} dB - Errores: {errores} - BER: {pe:.5f}")

        if i == 0:
            ejemplo_ruido = ruido
            ejemplo_decodificado = decodificado

    print("\nTabla de BER para cada SNR:")
    print("SNR (dB)\tBER")
    for snr, ber in zip(valores_snr, ber_resultados):
        print(f"{snr}\t\t{ber:.2e}")
    
    print(f"\nTotal de bits erróneos encontrados: {np.sum(errores_totales)}")

    return valores_snr, ber_resultados, bits, ejemplo_decodificado, ejemplo_ruido


# ----------------------------
# BER teórica
# ----------------------------
def ber_teorica_bpsk(snr_db):
    snr_lineal = 10**(snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_lineal))

# ----------------------------
# Gráfica de BER
# ----------------------------
def graficar_ber(valores_snr, ber_simulada):
    plt.figure(figsize=(8,5))
    plt.semilogy(valores_snr, ber_simulada, 'o-', color='blue', label="P(error) simulada")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilidad de Error (log)")
    plt.title("Probabilidad de error vs SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Gráfica de bits originales vs demodulados
# ----------------------------
def graficar_bits(bits, decodificados):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.step(range(100), bits[:100], where='mid', label="Bits originales")
    plt.title("Bits originales (100 muestras)")
    plt.xlabel("Índice")
    plt.ylabel("Valor bit")
    plt.ylim(-0.2, 1.2)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(range(100), decodificados[:100], where='mid', label="Bits demodulados", color='orange')
    plt.title("Bits demodulados (100 muestras)")
    plt.xlabel("Índice")
    plt.ylabel("Valor bit")
    plt.ylim(-0.2, 1.2)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Gráficas del ruido y errores por bit
# ----------------------------
def graficar_ruido_y_errores(ruido, bits, decodificados):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Histograma del ruido
    ax = axs[0]
    n, bins, patches = ax.hist(ruido[:1000], bins=50, density=True, alpha=0.6, color='purple', edgecolor='black')
    mu, std = norm.fit(ruido[:1000])
    x = np.linspace(min(bins), max(bins), 200)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label="PDF teórica")
    ax.set_title("Histograma del ruido AWGN + PDF teórica")
    ax.set_xlabel("Amplitud ruido")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True)

    # Señal de ruido
    ax = axs[1]
    ax.plot(ruido[:100], color='red')
    ax.set_title("Ruido Gaussiano generado (100 muestras)")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Amplitud ruido")
    ax.grid(True)

    # Errores por bit
    ax = axs[2]
    errores = (bits != decodificados).astype(int)
    ax.stem(range(100), errores[:100], linefmt='r-', markerfmt='ro', basefmt='k-')
    ax.set_title("Error por bit (primeras 100 muestras)")
    ax.set_xlabel("Índice bit")
    ax.set_ylabel("Error (1=error, 0=correcto)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Comparación BER simulada vs teórica
# ----------------------------
def comparar_ber(valores_snr, ber_simulada):
    ber_teorica = ber_teorica_bpsk(np.array(valores_snr))
    plt.figure(figsize=(8, 5))
    plt.semilogy(valores_snr, ber_simulada, 'o-', label='BER simulada')
    plt.semilogy(valores_snr, ber_teorica, 's--', label='BER teórica BPSK')
    plt.title("Comparación BER simulada vs teórica")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilidad de error (log)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Programa principal
# ----------------------------
if __name__ == "__main__":
    m = 7
    realimentaciones = [7, 6, 4, 2]
    semilla = [1, 0, 0, 0, 0, 0, 0]
    A = 1
    n_bits = 10000

    snrs, bers, bits, decodificados, ruido = simular_sistema(m, realimentaciones, semilla, A, n_bits)
    
    graficar_ber(snrs, bers)
    graficar_bits(bits, decodificados)
    graficar_ruido_y_errores(ruido, bits, decodificados)
    comparar_ber(snrs, bers)
