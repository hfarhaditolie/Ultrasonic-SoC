<div align="center">

# Large format battery SoC estimation: An ultrasonic sensing and deep transfer learning predictions for heterogeneity

[**Hamidreza Farhadi Tolie**](https://scholar.google.com/citations?user=nzCbjWIAAAAJ&hl=en&authuser=1)<sup>a, b</sup> · **Benjamin Reichmann**<sup>c</sup> · [**James Marco**](https://scholar.google.com/citations?user=icR08CQAAAAJ&hl=en&oi=ao)<sup>a, b</sup> · [**Zahra Sharif Khodaei**](https://scholar.google.com/citations?user=iy8X1bUAAAAJ&hl=en&authuser=1)<sup>c</sup> ·
[**Mona Faraji Niri**](https://scholar.google.com/citations?user=1PK7IocAAAAJ&hl=en&oi=ao)<sup>a, b</sup>
<br>

<sup>a</sup> Warwick Manufacturing Group, University of Warwick, Coventry, UK  
<sup>b</sup> The Faraday Institution, Harwell Science & Innovation Campus, Didcot, UK  
<sup>c</sup> Department of Aeronautics, Imperial College London, London, UK

<hr>

<a href="https://github.com/hfarhaditolie/Ultrasonic-SoC">
<img src="https://img.shields.io/badge/Ultrasonic%20SoC-Dataset%20%26%20Code-blue?style=for-the-badge" alt="Project Badge">
</a>
<br>
</div>

This repository provides the full implementation of a **deep learning–enhanced ultrasonic sensing framework** developed for accurate and real-time **State of Charge (SoC) estimation** in large-format lithium-ion pouch cells.

---

## 🔍 Abstract

> Accurate State of Charge (SoC) estimation is vital for the safe and efficient operation of lithium-ion batteries. Classical approaches such as Coulomb counting and open-circuit voltage measurements face well-known limitations including drift, cumulative error and sensitivity to spatial inhomogeneities—issues that are amplified in large-format cells used in electric vehicles and grid storage.  
>
> This study investigates **ultrasonic sensing** as a non-invasive and real-time alternative for SoC estimation. Leveraging experimentally collected ultrasonic signals transmitted between **four sensors**, a **customised deep learning framework** is developed that transforms raw waveforms into images and applies transfer learning using strong pre-trained convolutional models.  
>
> We demonstrate that using **bidirectional ultrasonic transmission** and **dynamic machine learning–based actuator–receiver selection** significantly improves SoC estimation accuracy compared with traditional data-driven analysis. Furthermore, initial investigations into **self-supervision** reveal promising potential for reducing reliance on conventional ground-truth measurements.  

---

## 📁 Repository Structure

```
Ultrasonic-SoC/
│
├── data/
│   ├── charging/
│   │   ├── Signal1_2_SoC_raw.csv
│   │   ├── Signal1_3_SoC_raw.csv
│   │   └── ...
│   ├── discharging/
│   │   ├── Signal1_2_SoCD_raw.csv
│   │   ├── Signal1_3_SoCD_raw.csv
│   │   └── ...
│   └── README_data.md
│
├── src/
│   ├── preprocessing/
│   ├── waveform_to_image/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── notebooks/
├── results/
└── README.md
```

---

## 📡 Data Description

Raw ultrasonic waveforms are stored under `data/` and grouped by experiment phase:

- **Charging:**  
  `SignalA_B_SoC_raw.csv`

- **Discharging:**  
  `SignalA_B_SoCD_raw.csv`  
  where **D** = discharging.

Each filename represents a **sensor pair**:

- `A` → actuator sensor ID  
- `B` → receiver sensor ID  

Example:  
- `Signal1_2_SoC_raw.csv` → Sensor 1 actuating → Sensor 2 receiving (charging)  
- `Signal1_3_SoCD_raw.csv` → Sensor 1 → Sensor 3 (discharging)

Each CSV file contains the full ultrasonic waveform acquired at a specific SoC stage.

---

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the waveforms
```bash
python src/preprocessing/preprocess_signals.py
```

### 3. Convert waveforms to images
```bash
python src/waveform_to_image/convert.py
```

### 4. Train the deep learning model
```bash
python src/training/train_model.py
```

### 5. Evaluate performance
```bash
python src/evaluation/evaluate.py
```

---

## 🧠 Key Features

- **Non-invasive ultrasonic sensing** for real-time SoC estimation  
- **Waveform-to-image transformation pipeline**  
- **Deep CNNs + transfer learning** for improved representation learning  
- **Bidirectional actuator–receiver signal fusion**  
- **Dynamic path selection** for optimal sensor pair identification  
- Early exploration of **self-supervised SoC learning**  

---

## 📊 Visual Examples

<p align="center">
<img src="https://dummyimage.com/800x350/cccccc/000000&text=Ultrasonic+Pipeline+Diagram+(placeholder)" width="80%">
<br>
<i>Full figures will be added after dataset release</i>
</p>

---

## 📌 Citation

If you find this repository useful, please cite our work (BibTeX available upon publication):

```
@article{FarhadiTolie2025UltrasonicSoC,
  title={Ultrasonic Sensing and Deep Learning for Accurate State of Charge Estimation in Large-Format Lithium-Ion Batteries},
  author={Farhadi Tolie, Hamidreza and Guk, Erdogan and Marco, James and Faraji Niri, Mona},
  journal={To appear},
  year={2025}
}
```

---

## 💬 Feedback & Contact

For questions, collaborations or feedback:

📧 **hamidreza.farhadi-tolie@warwick.ac.uk**  
📧 **h.farhaditolie@gmail.com**

---

## 🙏 Acknowledgements

We thank the Warwick Manufacturing Group and The Faraday Institution for supporting this research, and acknowledge the emerging body of work on ultrasonic diagnostics in electrochemical systems that inspires continued innovation.

---

## 📄 Licence

This project is licensed under the **MIT Licence**.

---

<div align="center">
Made with ❤️ and ultrasonic waves  
<br>
<b>© 2025 Hamidreza Farhadi Tolie</b>
</div>
