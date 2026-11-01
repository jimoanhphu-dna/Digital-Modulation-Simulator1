
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc

# ==========================================
# 1. CAU HINH MO PHONG
# ==========================================
np.random.seed(42)

N_BITS = 100_000
EBN0_DB = np.arange(0, 13, 2)
FS = 1000             # Tan so lay mau (Hz)
SYMBOL_RATE = 100     # Toc do symbol (symbols/s)
SPS = FS // SYMBOL_RATE
s = input("Nhap day bit (chi gom 0 va 1): ").strip()

if s and all(c in "01" for c in s):
    bits_demo = np.array([int(c) for c in s])
else:
    raise ValueError("Day bit khong hop le!")

print("DAY BIT DANG SU DUNG:", bits_demo)
# ==========================================
# 2. DIEU CHE / GIAI DIEU CHE
# ==========================================

def modulate(name, bits):
    """Dieu che va tra ve symbol phuc co nang luong trung binh Eb = 1."""
    if name in ("BASK", "BFSK", "BPSK"):
        b = bits
        if name == "BASK":
            return b.astype(float).astype(complex)
        if name == "BFSK":
            return b.astype(complex)
        return (2 * b - 1).astype(complex)

    if name == "QPSK":
        b = bits[:len(bits) // 2 * 2].reshape(-1, 2)
        return ((2*b[:, 0]-1) + 1j*(2*b[:, 1]-1)) / np.sqrt(2)

    if name == "16-QAM":
        b = bits[:len(bits) // 4 * 4].reshape(-1, 4)
        levels = np.array([-3, -1, 3, 1])
        I = levels[2*b[:, 0] + b[:, 1]]
        Q = levels[2*b[:, 2] + b[:, 3]]
        return (I + 1j*Q) / np.sqrt(10)

    raise ValueError("Phuong phap dieu che khong hop le")


def demodulate(name, received):
    """Giai dieu che symbol va tra ve chuoi bit."""
    if name == "BASK":
       return (received.real > (np.sqrt(2) / 2)).astype(int)

    if name == "BFSK":
        return (received.real > 0).astype(int)

    if name == "BPSK":
        return (received.real > 0).astype(int)

    if name == "QPSK":
        out = np.column_stack((
            (received.real > 0).astype(int),
            (received.imag > 0).astype(int)
        ))
        return out.reshape(-1)

    if name == "16-QAM":
        x = received.real * np.sqrt(10)
        y = received.imag * np.sqrt(10)

        def pam_bits(v):
            idx = np.zeros(len(v), dtype=int)
            idx[(v >= -2) & (v < 0)] = 1
            idx[(v >= 0) & (v < 2)] = 3
            idx[v >= 2] = 2
            return np.column_stack((idx // 2, idx % 2))

        return np.column_stack((
            pam_bits(x), pam_bits(y)
        )).reshape(-1)

    raise ValueError("Phuong phap giai dieu che khong hop le")

# KIEM TRA DIEU CHE / GIAI DIEU CHE 16-QAM
test_bits = np.array([
    0,0, 0,1, 1,0, 1,1,
    0,0, 0,1, 1,0, 1,1,
    0,0, 0,1, 1,0, 1,1,
    0,0, 0,1, 1,0, 1,1
])

test_symbols = modulate("16-QAM", test_bits)
recovered_bits = demodulate("16-QAM", test_symbols)

print("Bit ban dau:       ", test_bits)
print("Bit giai dieu che: ", recovered_bits)
print("So bit loi:        ", np.sum(test_bits != recovered_bits))
# ==========================================
# 3. TAO SONG MANG VA DANG SONG
# ==========================================

def carrier_wave(symbols, name):
    """Tao dang song mang minh hoa cho mot chuoi symbol."""
    t_symbol = np.arange(SPS) / FS
    waves = []

    for s in symbols[:8]:
        if name == "BFSK":
            f = 100 if s.real < 0.5 else 200
            wave = np.cos(2*np.pi*f*t_symbol)
        else:
            phase = np.angle(s)
            amplitude = abs(s)
            wave = amplitude * np.cos(
                2*np.pi*100*t_symbol + phase
            )
        waves.extend(wave)

    return np.asarray(waves)
def demodulate_bfsk_wave(rx_wave, n_symbols):
    """Giai dieu che BFSK bang hai bo tuong quan."""
    t = np.arange(SPS) / FS

    f0 = 100
    f1 = 200

    ref0 = np.cos(2 * np.pi * f0 * t)
    ref1 = np.cos(2 * np.pi * f1 * t)

    detected = []

    for i in range(n_symbols):
        start = i * SPS
        end = start + SPS
        segment = rx_wave[start:end]
        corr0 = np.sum(segment * ref0)
        corr1 = np.sum(segment * ref1)

        detected.append(1 if corr1 > corr0 else 0)

    return np.array(detected, dtype=int)
def modulate_bfsk_wave(bits):
    """Tao song BFSK: bit 0 dung 100 Hz, bit 1 dung 200 Hz."""
    t = np.arange(SPS) / FS

    f0 = 100
    f1 = 200

    waves = []
    for bit in bits:
        f = f1 if bit == 1 else f0
        wave = np.cos(2 * np.pi * f * t)
        wave = wave / np.sqrt(np.sum(wave**2))
        waves.extend(wave)

    return np.asarray(waves)
# ==========================================
# 4. BER LY THUYET
# ==========================================

def theoretical_ber(name, ebn0):
    if name in ("BPSK", "QPSK"):
        return 0.5 * erfc(np.sqrt(ebn0))

    if name == "BFSK":
        return 0.5 * erfc(np.sqrt(ebn0 / 2))

    if name == "BASK":
        # OOK coherent detection, voi nang luong bit trung binh Eb
        return 0.5 * erfc(np.sqrt(ebn0 / 2))

    if name == "16-QAM":
        M = 16
        k = np.log2(M)
        return (
            (4/k) * (1 - 1/np.sqrt(M))
            * 0.5
            * erfc(np.sqrt(3*k*ebn0/(2*(M-1))))
        )

    raise ValueError("Khong co cong thuc ly thuyet")


# ==========================================
# 5. MO PHONG BER
# ==========================================

methods = ["BASK", "BFSK", "BPSK", "QPSK", "16-QAM"]
results = {}

for name in methods:
    # Tao bit theo tung phuong phap
    bits = np.random.randint(0, 2, N_BITS)
    symbols = modulate(name, bits)

    # So bit thuc su duoc su dung
    if name == "QPSK":
        bits = bits[:len(symbols)*2]
    elif name == "16-QAM":
        bits = bits[:len(symbols)*4]

    ber_sim = []
    ber_theory = []

    for db in EBN0_DB:
        ebn0 = 10**(db/10)
        
        # BFSK: tao song, them AWGN, giai dieu che bang tuong quan 
        if name == "BFSK":
            tx_wave = modulate_bfsk_wave(bits)

            # Nang luong trung binh moi bit cua song cos = 1/2
            # Chuan hoa nang luong bit ve Eb = 1
            

            noise_std = np.sqrt(1 / (2 * ebn0))
            noise = noise_std * np.random.randn(len(tx_wave))
            rx_wave = tx_wave + noise

            detected = demodulate_bfsk_wave(rx_wave, len(bits))
            errors = np.sum(bits != detected)
            ber_sim.append(errors / len(bits))
            ber_theory.append(theoretical_ber(name, ebn0))
            continue

        # Chuoi symbol da duoc chuan hoa Eb = 1
        # AWGN phuc: phuong sai moi truc = N0/2
        bits_per_symbol = 1 if name in (
            "BASK", "BFSK", "BPSK"
        ) else (2 if name == "QPSK" else 4)

        if name == "BASK":
            # OOK: bit 0 -> 0, bit 1 -> sqrt(2)
            tx = symbols * np.sqrt(2)
            k = 1

        elif name == "QPSK":
            # 2 bit / symbol
            tx = symbols
            k = 2

        elif name == "16-QAM":
            # 4 bit / symbol
            tx = symbols
            k = 4

        else:
            # BFSK va BPSK trong mo hinh hien tai
            tx = symbols
            k = 1

        noise_std = np.sqrt(1 / (2 * k * ebn0))

        noise = noise_std * (
            np.random.randn(len(tx))
            + 1j * np.random.randn(len(tx))
        )

        rx = tx + noise

        detected = demodulate(name, rx)
        errors = np.sum(bits != detected[:len(bits)])
        ber_sim.append(errors/len(bits))
        ber_theory.append(theoretical_ber(name, ebn0))

    results[name] = {
        "symbols": symbols,
        "ber_sim": np.array(ber_sim),
        "ber_theory": np.array(ber_theory)
    }


# ==========================================
# 6. VE DO THI BER - PHIEN BAN HOAN THIEN
# ==========================================

fig, ax = plt.subplots(figsize=(10, 7))

# Mau sac nhat quan cho tung phuong phap
colors = {
    "BASK": "#0072B2",
    "BFSK": "#D55E00",
    "BPSK": "#009E73",
    "QPSK": "#CC79A7",
    "16-QAM": "#E69F00"
}

for name in methods:
    color = colors[name]

    # BER mo phong: danh dau bang hinh tron
    ber_sim_plot = results[name]["ber_sim"].astype(float).copy()
    ber_sim_plot[ber_sim_plot <= 0] = np.nan

    ax.semilogy(
        EBN0_DB,
        ber_sim_plot,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        color=color,
        label=f"{name} Simulation"
    )

    # BER ly thuyet: duong dut net
    ax.semilogy(
        EBN0_DB,
        results[name]["ber_theory"],
        linestyle="--",
        linewidth=2,
        color=color,
        alpha=0.85,
        label=f"{name} Theory"
    )

# Tieu de va nhan truc
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
ax.set_title(
    "BER Performance of Digital Modulation over AWGN",
    fontsize=14,
    fontweight="bold",
    pad=12
)

# Gioi han va vach chia
ax.set_xticks(EBN0_DB)
ax.set_xlim(EBN0_DB[0] - 0.5, EBN0_DB[-1] + 0.5)
ax.set_ylim(1e-6, 3e-1)

# Luoi chinh va luoi phu
ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.55)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

# Chu thich chia 2 cot
ax.legend(
    loc="lower left",
    ncol=2,
    fontsize=9,
    frameon=True,
    framealpha=0.95
)

fig.tight_layout()

# Luu anh do phan giai cao
fig.savefig(
    "BER_Comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)



# ==========================================
# 7. VE CHOM SAO I/Q - PHIEN BAN HOAN THIEN
# ==========================================

for name in ("QPSK", "16-QAM"):
    symbols = results[name]["symbols"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Ve cac diem chom sao ly tuong
    ax.scatter(
        symbols.real[:3000],
        symbols.imag[:3000],
        s=12 if name == "QPSK" else 8,
        alpha=0.55,
        color="#0072B2",
        edgecolors="none",
        label="Transmitted symbols"
    )

    # Truc I va Q tai goc toa do
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    # Nhan truc va tieu de
    ax.set_xlabel("In-phase (I)", fontsize=12)
    ax.set_ylabel("Quadrature (Q)", fontsize=12)
    ax.set_title(
        f"{name} Constellation Diagram",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    # Luoi va ti le truc
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_aspect("equal", adjustable="box")

    # Gioi han truc phu hop voi tung dieu che
    if name == "QPSK":
        limit = 1.0
    else:
        limit = 1.2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    # Luu anh chat luong cao
    fig.savefig(
        f"{name}_Constellation.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

# ==========================================
# 8. IN KET QUA
# ==========================================


# ==========================================
# 8. VE DANG SONG DIEU CHE
# ==========================================


for name in ("BASK", "BFSK", "BPSK"):

    if name == "BFSK":
        wave = modulate_bfsk_wave(bits_demo)
    else:
        symbols_demo = modulate(name, bits_demo)
        wave = carrier_wave(symbols_demo, name)

    t = np.arange(len(wave)) / FS

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Duong song chinh
    ax.plot(t, wave, color="#2878B5", linewidth=1.6)

    # Ranh gioi cac bit: nhe va khong gay roi mat
    for i in range(1, len(bits_demo)):
        ax.axvline(
            i / SYMBOL_RATE,
            color="#B0B0B0",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5
        )

    # Tieu de va nhan truc
    ax.set_title(f"{name} Modulated Signal", fontsize=13, pad=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)

    ax.set_xlim(0, len(bits_demo) / SYMBOL_RATE)
    ax.set_xticks(np.arange(0, (len(bits_demo) + 1) / SYMBOL_RATE,
                            1 / SYMBOL_RATE))

    # Nen trang, luoi nhe
    ax.set_facecolor("white")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{name}_Waveform.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

print("\nDa ve xong cac dang song BASK, BFSK va BPSK.")
