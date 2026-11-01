
import numpy as np
import matplotlib.pyplot as plt

# Buoc 1: Tao chuoi bit
np.random.seed(42)
N = 20
bits = np.random.randint(0, 2, N)

# Buoc 2: Dieu che BASK
# Bit 0 -> bien do 0
# Bit 1 -> bien do 1
symbols = bits.astype(float)

# Buoc 3: Them nhieu AWGN
noise_std = 0.3
noise = np.random.normal(0, noise_std, N)
received = symbols + noise

# Buoc 4: Giai dieu che BASK
# Nguong quyet dinh = 0.5
detected_bits = (received > 0.5).astype(int)

# Buoc 5: Tinh so bit loi va BER
errors = np.sum(bits != detected_bits)
ber = errors / N

# Buoc 6: Ve do thi
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(symbols)
plt.title("BASK Transmitted Symbols")
plt.ylabel("Amplitude")
plt.yticks([0, 1])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(received)
plt.axhline(0.5, color="red", linestyle="--",
            label="Decision threshold")
plt.title("BASK Received Symbols with AWGN")
plt.xlabel("Symbol Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Buoc 7: In ket qua
print("Bits ban dau:")
print(bits)

print("\nBASK symbols:")
print(symbols.astype(int))

print("\nTin hieu thu:")
print(np.round(received, 3))

print("\nBits thu duoc:")
print(detected_bits)

print("\nSo bit loi:", errors, "/", N)
print("BER =", ber)
