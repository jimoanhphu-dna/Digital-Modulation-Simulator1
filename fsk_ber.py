
import numpy as np
import matplotlib.pyplot as plt

# Buoc 1: Tao chuoi bit
np.random.seed(42)
bits = np.array([0, 1, 0, 1, 1, 0, 0, 1])

# Buoc 2: Cau hinh BFSK
f0 = 2
f1 = 5
Tb = 1.0
fs = 100

t = np.arange(0, Tb, 1 / fs)

# Hai tin hieu tham chieu
ref0 = np.sin(2 * np.pi * f0 * t)
ref1 = np.sin(2 * np.pi * f1 * t)

# Buoc 3: Dieu che BFSK
tx = []

for bit in bits:
    frequency = f1 if bit == 1 else f0
    tx.append(np.sin(2 * np.pi * frequency * t))

tx = np.array(tx)

# Buoc 4: Them nhieu AWGN
noise_std = 0.5
noise = np.random.normal(0, noise_std, tx.shape)
rx = tx + noise

# Buoc 5: Giai dieu che bang tuong quan
detected_bits = []

for received_symbol in rx:
    corr0 = np.sum(received_symbol * ref0)
    corr1 = np.sum(received_symbol * ref1)

    if corr1 > corr0:
        detected_bits.append(1)
    else:
        detected_bits.append(0)

detected_bits = np.array(detected_bits)

# Buoc 6: Tinh BER
errors = np.sum(bits != detected_bits)
ber = errors / len(bits)

# Buoc 7: Ve tin hieu phat va thu cua bit dau tien
time = np.arange(len(bits) * len(t)) / fs
tx_wave = tx.flatten()
rx_wave = rx.flatten()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, tx_wave)
plt.title("BFSK Transmitted Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, rx_wave)
plt.title("BFSK Received Signal with AWGN")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# Buoc 8: In ket qua
print("Bits goc:")
print(bits)

print("\nBits thu duoc:")
print(detected_bits)

print("\nSo bit loi:", errors, "/", len(bits))
print("BER =", ber)
