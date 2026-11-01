import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from modulation import (
    modulate,
    demodulate,
    add_awgn,
    theoretical_ber,
    modulate_bfsk_wave,
    demodulate_bfsk_wave,
    add_awgn_bfsk
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Digital Modulation Simulator",
    page_icon="📡",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #e2e8f0;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #26344d;
        padding: 15px;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 45px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
# ==========================================
# BER SIMULATION FUNCTION
# ==========================================

def simulate_ber(name, ebn0_db, num_bits):
    bits = np.random.randint(0, 2, num_bits)

    if name == "BFSK":
        tx_wave = modulate_bfsk_wave(bits)
        rx_wave = add_awgn_bfsk(tx_wave, ebn0_db)
        detected = demodulate_bfsk_wave(rx_wave, len(bits))

    else:
        symbols = modulate(name, bits)
        received = add_awgn(name, symbols, ebn0_db)
        detected = demodulate(name, received)

    errors = np.sum(bits != detected)
    ber_simulated = errors / num_bits

    ebn0_linear = 10 ** (ebn0_db / 10)
    ber_theoretical = theoretical_ber(name, ebn0_linear)

    return ber_simulated, ber_theoretical
# ==========================================
# BER CURVE SIMULATION
# ==========================================

def simulate_ber_curve(name, num_bits):
    ebn0_range = np.arange(0, 13, 2)

    ber_simulated = []
    ber_theoretical = []

    for ebn0_db in ebn0_range:
        ber_sim, ber_theory = simulate_ber(
            name,
            ebn0_db,
            num_bits
        )

        ber_simulated.append(ber_sim)
        ber_theoretical.append(ber_theory)

    return (
        ebn0_range,
        np.array(ber_simulated),
        np.array(ber_theoretical)
    )
def simulate_all_ber_curves(num_bits):
    modulation_list = ["BASK", "BFSK", "BPSK", "QPSK", "16-QAM"]

    all_results = {}

    for name in modulation_list:
        ebn0_range, ber_simulated, ber_theoretical = simulate_ber_curve(
            name, num_bits
        )

        all_results[name] = {
            "ebn0": ebn0_range,
            "simulation": ber_simulated,
            "theory": ber_theoretical
        }

    return all_results
# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">📡 Digital Modulation Simulator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Digital Communication System | Simulation Laboratory</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Simulation Settings")

modulation = st.sidebar.selectbox(
    "Modulation Technique",
    ["BASK", "BFSK", "BPSK", "QPSK", "16-QAM"]
)

bit_sequence = st.sidebar.text_input(
    "Input Bit Sequence",
    value="10101010"
)

ebn0 = st.sidebar.slider(
    "Eb/N0 (dB)",
    0, 20, 10
)

num_bits = st.sidebar.selectbox(
    "Number of bits for BER",
    [1000, 10000, 100000, 1000000],
    index=2
)

run_simulation = st.sidebar.button(
    "▶ RUN SIMULATION",
    type="primary"
)

st.sidebar.divider()
st.sidebar.caption("Digital Modulation Project")
st.sidebar.caption("Python | NumPy | SciPy | Streamlit")

# ==========================================
# STATUS
# ==========================================

if run_simulation:
    valid = (
        len(bit_sequence) > 0
        and all(c in "01" for c in bit_sequence)
)

    if not valid:
        st.sidebar.error("Invalid bit sequence!")

    elif modulation == "16-QAM" and len(bit_sequence) % 4 != 0:
        st.sidebar.error("16-QAM requires bits divisible by 4.")

    elif modulation == "QPSK" and len(bit_sequence) % 2 != 0:
        st.sidebar.error("QPSK requires an even number of bits.")

    else:
        with st.spinner("Running simulation..."):
            try:
                ber_sim, ber_theory = simulate_ber(
                    modulation,
                    ebn0,
                    num_bits
                )

                st.session_state["simulation_ready"] = True
                st.session_state["input_bits"] = bit_sequence
                st.session_state["modulation"] = modulation
                st.session_state["ebn0"] = ebn0
                st.session_state["num_bits"] = num_bits
                st.session_state["ber_sim"] = ber_sim
                st.session_state["ber_theory"] = ber_theory

            except Exception as e:
                st.session_state["simulation_ready"] = False
                st.error(f"Simulation error: {e}")

# ==========================================
# METRICS
# ==========================================
st.subheader("📊 Simulation Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Modulation", modulation)
c2.metric("Input Bits", len(bit_sequence))
c3.metric("Eb/N0", f"{ebn0} dB")
c4.metric("BER Data", f"{num_bits:,} bits")

st.divider()

# ==========================================
# SIGNAL ANALYSIS
# ==========================================
st.subheader("📈 Signal Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Waveform",
    "Constellation",
    "BER Performance",
    "Comparison",
    "Simulation Gallery"
])


with tab1:
    st.markdown("#### Time Domain Signal")

    if st.session_state.get("simulation_ready", False):

        # Lay chuoi bit va thong so mo phong
        bits = np.array(
            [int(bit) for bit in bit_sequence],
            dtype=int
        )

        # Tao tin hieu dieu che va tin hieu co nhieu
        if modulation == "BFSK":
            tx_signal = modulate_bfsk_wave(bits)
            rx_signal = add_awgn_bfsk(tx_signal, ebn0)

            detected_bits = demodulate_bfsk_wave(
                rx_signal, len(bits)
            )

            fs = 1000
            time = np.arange(len(tx_signal)) / fs

            tx_plot = tx_signal
            rx_plot = rx_signal

        else:
            symbols = modulate(modulation, bits)
            rx_symbols = add_awgn(
                modulation, symbols, ebn0
            )

            detected_bits = demodulate(
                modulation, rx_symbols
            )

            # BASK duoc chuan hoa nang luong trong add_awgn()
            tx_symbols = (
                symbols * np.sqrt(2)
                if modulation == "BASK"
                else symbols
            )

            samples_per_symbol = 20

            tx_plot = np.repeat(
                tx_symbols, samples_per_symbol
            )
            rx_plot = np.repeat(
                rx_symbols, samples_per_symbol
            )

            time = (
                np.arange(len(tx_plot))
                / samples_per_symbol
            )

        # 1. Input bit sequence
        st.markdown("##### 1. Input Bit Sequence")

        fig1, ax1 = plt.subplots(figsize=(11, 2.5))

        ax1.step(
            np.arange(len(bits)),
            bits,
            where="post"
        )

        ax1.set_xlabel("Bit Index")
        ax1.set_ylabel("Bit Value")
        ax1.set_title("Input Digital Signal")
        ax1.set_yticks([0, 1])
        ax1.set_ylim(-0.3, 1.3)
        ax1.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig1)
        plt.close(fig1)

        # 2. Transmitted and Received Signal
        st.markdown("##### 2. Transmitted and Received Signal")

        fig2, ax2 = plt.subplots(figsize=(11, 4))

        if modulation == "BFSK":
            ax2.plot(
                time,
                tx_plot,
                label="Transmitted Signal"
            )
            ax2.plot(
                time,
                rx_plot,
                label="Received Signal",
                alpha=0.7
            )

            ax2.set_xlabel("Time (seconds)")

        else:
            ax2.plot(
                time,
                tx_plot.real,
                label="Transmitted I"
            )
            ax2.plot(
                time,
                rx_plot.real,
                label="Received I",
                alpha=0.7
            )

            if modulation in ["QPSK", "16-QAM"]:
                ax2.plot(
                    time,
                    tx_plot.imag,
                    "--",
                    label="Transmitted Q"
                )
                ax2.plot(
                    time,
                    rx_plot.imag,
                    "--",
                    label="Received Q",
                    alpha=0.7
                )

            ax2.set_xlabel("Time (Symbol Intervals)")

        ax2.set_ylabel("Amplitude")
        ax2.set_title(
            f"{modulation} Signal over AWGN Channel"
        )
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        st.pyplot(fig2)
        plt.close(fig2)

        # 3. Demodulated bits
        st.markdown("##### 3. Demodulated Bit Sequence")

        fig3, ax3 = plt.subplots(figsize=(11, 2.5))

        ax3.step(
            np.arange(len(detected_bits)),
            detected_bits,
            where="post",
            label="Detected Bits"
        )

        ax3.set_xlabel("Bit Index")
        ax3.set_ylabel("Bit Value")
        ax3.set_title("Recovered Digital Signal")
        ax3.set_yticks([0, 1])
        ax3.set_ylim(-0.3, 1.3)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend()

        st.pyplot(fig3)
        plt.close(fig3)

        # Calculate bit errors
        errors = np.sum(bits != detected_bits)
        ber = errors / len(bits)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Bits", len(bits))
        col2.metric("Bit Errors", int(errors))
        col3.metric("BER", f"{ber:.6f}")

    else:
        st.info("Press RUN SIMULATION to generate waveform.")




with tab2:
    st.markdown("#### Constellation Diagram")

    if st.session_state.get("simulation_ready", False):

        if modulation == "BFSK":
            st.info(
                "BFSK uses two carrier frequencies. "
                "Its signal is better analyzed in the time domain."
            )

        else:
            # Number of symbols for constellation visualization
            num_constellation_symbols = 2000

            bits_per_symbol = {
                "BASK": 1,
                "BPSK": 1,
                "QPSK": 2,
                "16-QAM": 4
            }

            num_bits_constellation = (
                num_constellation_symbols *
                bits_per_symbol[modulation]
            )

            # Generate random bits for constellation
            constellation_bits = np.random.randint(
                0, 2, num_bits_constellation
            )

            # Modulation
            symbols = modulate(
                modulation,
                constellation_bits
            )

            # Normalize BASK
            if modulation == "BASK":
                tx_symbols = symbols * np.sqrt(2)
            else:
                tx_symbols = symbols

            # Add AWGN noise
            rx_symbols = add_awgn(
                modulation,
                symbols,
                ebn0
            )

             # Ideal constellation points
            if modulation == "QPSK":

                # Generate all possible QPSK bit combinations
                all_qpsk_bits = np.array([
                    0, 0,
                    0, 1,
                    1, 0,
                    1, 1
                ])

                ideal_points = np.unique(
                    modulate("QPSK", all_qpsk_bits)
                )

            else:
                ideal_points = np.unique(tx_symbols)

            # Plot constellation
            fig, ax = plt.subplots(figsize=(9, 7))

           # Received symbols
            ax.scatter(
                rx_symbols.real,
                rx_symbols.imag,
                color="tab:blue",
                alpha=0.25,
                s=9,
                label="Received Symbols"
            )
            # Ideal constellation
            ax.scatter(
                ideal_points.real,
                ideal_points.imag,
                color="red",
                marker="x",
                s=150,
                linewidths=3,
                label="Ideal Constellation"
            )

            # Reference axes
            ax.axhline(
                0,
                color="gray",
                linewidth=0.8
            )

            ax.axvline(
                0,
                color="gray",
                linewidth=0.8
            )

            ax.set_xlabel("In-phase (I)")
            ax.set_ylabel("Quadrature (Q)")

            ax.set_title(
                f"{modulation} Constellation Diagram "
                f"(Eb/N0 = {ebn0} dB)"
            )

            ax.grid(
                True,
                linestyle="--",
                alpha=0.5
            )

            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Constellation information
            st.markdown("#### Constellation Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Modulation",
                modulation
            )

            col2.metric(
                "Number of Symbols",
                len(symbols)
            )

            col3.metric(
                "Eb/N0",
                f"{ebn0} dB"
            )

            st.caption(
                "2,000 randomly generated symbols are used "
                "for constellation visualization. "
                "Blue dots represent received symbols after AWGN, "
                "while red crosses represent ideal constellation points."
            )

    else:
        st.info(
            "Press RUN SIMULATION to generate constellation diagram."
        )

with tab3:
    st.markdown("#### Bit Error Rate")

    if st.session_state.get("simulation_ready", False):

        with st.spinner("Calculating BER curve..."):
            ebn0_range, ber_simulated, ber_theoretical = (
                simulate_ber_curve(modulation, num_bits)
            )

        # Thay BER = 0 bằng mức hiển thị nhỏ nhất trên trục log
        ber_plot = np.where(
            ber_simulated == 0,
            0.5 / num_bits,
            ber_simulated
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.semilogy(
            ebn0_range,
            ber_plot,
            "o-",
            label="Simulation"
        )

        ax.semilogy(
            ebn0_range,
            ber_theoretical,
            "s--",
            label="Theory"
        )

        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(f"{modulation} BER Performance over AWGN")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Note: Zero simulated BER values are displayed at "
            "0.5/N bits for visibility on the logarithmic axis."
        )

        # Bảng kết quả
        st.markdown("#### BER Data")

        results = {
            "Eb/N0 (dB)": ebn0_range,
            "Simulated BER": ber_simulated,
            "Theoretical BER": ber_theoretical
        }

        st.dataframe(results, use_container_width=True)
        
        # Performance Summary
        st.markdown("#### Performance Summary")

        best_idx = np.argmin(ber_simulated)

        best_ebn0 = ebn0_range[best_idx]
        best_sim_ber = ber_simulated[best_idx]
        best_theory_ber = ber_theoretical[best_idx]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Simulated BER",
            f"{best_sim_ber:.6f}"
        )

        col2.metric(
            "Corresponding Eb/N0",
            f"{best_ebn0} dB"
        )

        col3.metric(
            "Theoretical BER",
            f"{best_theory_ber:.6f}"
        )

        st.markdown("#### Performance Analysis")

        st.write(
            f"The selected modulation technique is **{modulation}**."
        )

        st.write(
            f"The lowest simulated BER is **{best_sim_ber:.6f}**, "
            f"observed at Eb/N0 = **{best_ebn0} dB**."
        )

        st.write(
            "As Eb/N0 increases, the noise power decreases "
            "relative to the signal energy. Therefore, "
            "the BER is generally expected to decrease."
        )

        st.write(
            "The simulated BER may differ from the theoretical "
            "BER because the simulation uses a finite number "
            "of randomly generated bits."
        )

          
    else:
        st.info("Press RUN SIMULATION to generate BER performance.")
with tab4:
    st.markdown("#### BER Comparison - All Modulation Techniques")

    if st.session_state.get("simulation_ready", False):

        comparison_bits = 100000

        with st.spinner("Simulating all modulation techniques..."):
            all_results = simulate_all_ber_curves(comparison_bits)

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, result in all_results.items():

            ebn0 = result["ebn0"]
            ber_sim = result["simulation"]
            ber_theory = result["theory"]

            # Thay BER = 0 bằng giá trị hiển thị trên trục log
            ber_plot = np.where(
                ber_sim == 0,
                0.5 / comparison_bits,
                ber_sim
            )

            ax.semilogy(
                ebn0,
                ber_plot,
                "o-",
                label=f"{name} Simulation"
            )

            ax.semilogy(
                ebn0,
                ber_theory,
                "--",
                label=f"{name} Theory"
            )

        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title("BER Comparison of Digital Modulation Techniques")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        ax.legend(fontsize=8, ncol=2)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
         # Bang so sanh BER
        st.markdown("#### BER Comparison Data")

        comparison_data = []

        for name, result in all_results.items():
           for i, ebn0 in enumerate(result["ebn0"]):
              comparison_data.append({
                 "Modulation": name,
                "Eb/N0 (dB)": int(ebn0),
                "Simulated BER": result["simulation"][i],
                "Theoretical BER": result["theory"][i]
                })

        st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
        )
        st.markdown("#### Modulation Characteristics")

        modulation_info = {
            "Modulation": ["BASK", "BFSK", "BPSK", "QPSK", "16-QAM"],
            "Bits per Symbol": [1, 1, 1, 2, 4],
            "Modulation Order": [2, 2, 2, 4, 16],
            "Detection": [
                "Coherent",
                "Coherent",
                "Coherent",
                "Coherent",
                "Coherent"
            ]
        }

        st.dataframe(
            modulation_info,
            use_container_width=True,
            hide_index=True
        )
        st.markdown("#### Performance Analysis")

        st.markdown("""
        **1. BASK**
        - Simple modulation scheme.
        - Sensitive to noise and amplitude variations.

        **2. BFSK**
        - Uses two different carrier frequencies.
        - Provides good noise performance under coherent detection.

        **3. BPSK**
        - Transmits one bit per symbol.
        - Offers good noise immunity and simple implementation.

        **4. QPSK**
        - Transmits two bits per symbol.
        - Has the same theoretical BER as BPSK under AWGN.

        **5. 16-QAM**
        - Transmits four bits per symbol.
        - Provides higher data rate potential but requires better signal quality.
        """)

        st.caption(
            "Comparison uses 100,000 bits per modulation technique. "
            "Zero simulated BER values are displayed at 0.5/N."
        )

    else:
        st.info("Press RUN SIMULATION to enable BER comparison.")
st.divider()

# ==========================================
# TAB 5: SIMULATION GALLERY
# ==========================================

with tab5:
    from pathlib import Path

    st.markdown("# Digital Modulation Simulation Gallery")

    st.markdown("""
    ### Tổng quan kết quả mô phỏng

    Tab này trình bày các kết quả được tạo từ chương trình
    `modulation.py`, bao gồm dạng sóng điều chế, biểu đồ
    chòm sao và hiệu suất BER.

    Các hình ảnh giúp quan sát trực quan đặc điểm của từng
    phương pháp điều chế và đánh giá ảnh hưởng của nhiễu AWGN.
    """)

    st.divider()

    # --------------------------------------
    # 1. BER COMPARISON
    # --------------------------------------

    st.markdown("## 1. BER Performance Comparison")

    st.markdown("""
    Biểu đồ so sánh tỷ lệ lỗi bit (BER) giữa kết quả mô phỏng
    và kết quả lý thuyết của năm phương pháp điều chế.

    - Đường liền: BER mô phỏng.
    - Đường nét đứt: BER lý thuyết.
    - BER càng thấp: chất lượng truyền dẫn càng tốt.
    """)

    ber_image = Path("BER_Comparison.png")

    if ber_image.exists():
        st.image(
            str(ber_image),
            caption="Figure 1. BER Performance Comparison",
            use_container_width=True
        )
    else:
        st.warning("Chưa tìm thấy BER_Comparison.png. Hãy chạy modulation.py trước.")

    st.info("""
    **Nhận xét:** Khi Eb/N0 tăng, BER nhìn chung giảm.
    Kết quả mô phỏng được so sánh với đường BER lý thuyết
    để đánh giá tính phù hợp của chương trình.
    """)

    st.divider()

    # --------------------------------------
    # 2. CONSTELLATION DIAGRAM
    # --------------------------------------

    st.markdown("## 2. Constellation Diagrams")

    st.markdown("""
    Biểu đồ chòm sao biểu diễn các symbol trên mặt phẳng I/Q
    (In-phase và Quadrature). Mỗi điểm đại diện cho một symbol
    được truyền đi.
    """)

    col1, col2 = st.columns(2)

    with col1:
        qpsk_image = Path("QPSK_Constellation.png")

        st.markdown("### 2.1 QPSK")

        if qpsk_image.exists():
            st.image(
                str(qpsk_image),
                caption="Figure 2. QPSK Constellation",
                use_container_width=True
            )
        else:
            st.warning("Chưa tìm thấy QPSK_Constellation.png.")

        st.caption("""
        QPSK sử dụng 4 điểm chòm sao, tương ứng với 4 tổ hợp
        hai bit khác nhau. Mỗi symbol mang 2 bit thông tin.
        """)

    with col2:
        qam_image = Path("16-QAM_Constellation.png")

        st.markdown("### 2.2 16-QAM")

        if qam_image.exists():
            st.image(
                str(qam_image),
                caption="Figure 3. 16-QAM Constellation",
                use_container_width=True
            )
        else:
            st.warning("Chưa tìm thấy 16-QAM_Constellation.png.")

        st.caption("""
        16-QAM sử dụng 16 điểm chòm sao khác nhau.
        Mỗi symbol mang 4 bit thông tin.
        """)

    st.divider()

    # --------------------------------------
    # 3. MODULATED WAVEFORMS
    # --------------------------------------

    st.markdown("## 3. Digital Modulation Waveforms")

    st.markdown("""
    Các đồ thị dưới đây minh họa dạng sóng tín hiệu sau khi
    thực hiện điều chế. Trục ngang biểu diễn thời gian,
    trục dọc biểu diễn biên độ tín hiệu.
    """)

    waveform_data = [
        (
            "BASK",
            "BASK_Waveform.png",
            "Biên độ sóng mang thay đổi theo bit đầu vào."
        ),
        (
            "BFSK",
            "BFSK_Waveform.png",
            "Bit 0 và bit 1 được biểu diễn bằng hai tần số khác nhau."
        ),
        (
            "BPSK",
            "BPSK_Waveform.png",
            "Pha sóng mang thay đổi giữa hai trạng thái tương ứng với bit 0 và 1."
        )
    ]

    cols = st.columns(3)

    for col, (name, filename, description) in zip(
        cols, waveform_data
    ):
        with col:
            st.markdown(f"### {name}")

            image_path = Path(filename)

            if image_path.exists():
                st.image(
                    str(image_path),
                    caption=f"{name} Modulated Signal",
                    use_container_width=True
                )
            else:
                st.warning(f"Chưa tìm thấy {filename}.")

            st.caption(description)

    st.divider()

    # --------------------------------------
    # 4. CONCLUSION
    # --------------------------------------

    st.markdown("## 4. Overall Observations")

    st.markdown("""
    Qua các kết quả mô phỏng có thể quan sát:

    1. Mỗi phương pháp điều chế có đặc điểm dạng sóng riêng.
    2. QPSK và 16-QAM biểu diễn thông tin bằng các điểm trên
       mặt phẳng I/Q.
    3. Biểu đồ BER cho phép đánh giá khả năng chống nhiễu
       của các phương pháp điều chế.
    4. Việc so sánh kết quả mô phỏng với lý thuyết giúp
       kiểm tra hoạt động của hệ thống.

    **Kết luận:** Các hình ảnh cung cấp góc nhìn trực quan
    về quá trình điều chế số và hiệu suất truyền dẫn
    trong môi trường nhiễu AWGN.
    """)

# ==========================================
# RESULTS
# ==========================================

st.subheader("🔬 Simulation Results")

if st.session_state.get("simulation_ready", False):

    st.success("Simulation completed successfully!")

    st.write("**Input bit sequence:**",
             st.session_state["input_bits"])

    st.write("**Selected modulation:**",
             st.session_state["modulation"])

    st.write("**Eb/N0:**",
             f'{st.session_state["ebn0"]} dB')

    st.divider()

    # BER Results
    st.markdown("### Bit Error Rate")

    col1, col2 = st.columns(2)

    col1.metric(
        "Simulated BER",
        f'{st.session_state["ber_sim"]:.6f}'
    )

    col2.metric(
        "Theoretical BER",
        f'{st.session_state["ber_theory"]:.6f}'
    )

    st.write(
        f'**Number of bits:** {st.session_state["num_bits"]:,}'
    )

else:
    st.caption(
        "Configure the simulation in the sidebar "
        "and press RUN SIMULATION."
    )

st.divider()

st.caption(
    "Digital Modulation Simulator | Academic Project"
)
