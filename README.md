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
|   ├── waveform_images/
│   │   ├── waveform_000.png
│   │   ├── waveform_000.png
│   │   └── ...
│   └── SoCs.npy
│   └── signal_data.npy
│
├── src/
│   ├── convert.py
│   ├── convert.py
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

File contents (CSV):
- Each CSV contains the full ultrasonic waveform for a single acquisition (at one SoC stage).
- The last column of each CSV file contains the corresponding SoC value for that waveform.
- All other columns contain the time-series samples (waveform amplitudes).

Precomputed / derived data:

- Waveform images (spectrograms/waveform plots) are stored in the waveform_images/ folder (one image per acquisition;).
- Signal arrays for the sensor pair (3, 4) have been pre-saved as:
  - signal_data.npy — NumPy array of shape (N, L) containing the raw signal vectors
  - SoCs.npy — NumPy array of shape (N,) containing the matching SoC values

Examples for clarity:
- Signal1_2_SoC_raw.csv → charging experiment; sensor 1 actuates, sensor 2 receives.
- Signal1_3_SoCD_raw.csv → discharging experiment; sensor 1 actuates, sensor 3 receives.
- waveform_images/waveform_000.png (or similar) → generated image for the first acquisition.
---

## 🚀 Usage


### 1. Convert waveforms to images
```bash
python src/convert.py
```
use the code above to regenerate the waveform images for the sensor pair you desire, the signal_data.npy needs to be updated accordingly as well.
### 2. Train/Evaluate the deep learning model
```bash
python src/main.py
```
to train and evaluate the model on both K-Fold and holdout training modes run the code above and then specify the mode of training when asked.

---

## 🧠 Key Features

- **Non-invasive ultrasonic sensing** for real-time SoC estimation  
- **Waveform-to-image transformation pipeline**  
- **Deep CNNs + transfer learning** for improved representation learning  
- **Bidirectional actuator–receiver signal fusion**  
- **Dynamic path selection** for optimal sensor pair identification  
---

## 📊 Visual Examples

<p align="center">
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S2666546825001946-gr1_lrg.jpg" width="80%">
<br>
</p>

---

## 📌 Citation

If you find this repository useful, please cite our work (BibTeX available upon publication):

```
@article{FARHADITOLIE2025100662,
title = {Large format battery SoC estimation: An ultrasonic sensing and deep transfer learning predictions for heterogeneity},
journal = {Energy and AI},
pages = {100662},
year = {2025},
issn = {2666-5468},
doi = {https://doi.org/10.1016/j.egyai.2025.100662},
url = {https://www.sciencedirect.com/science/article/pii/S2666546825001946},
author = {Hamidreza {Farhadi Tolie} and Benjamin Reichmann and James Marco and Zahra {Sharif Khodaei} and Mona {Faraji Niri}},
keywords = {Ultrasonic sensing, State of Charge estimation, Deep neural networks, Directional signal analysis, Ultrasonic sensor placement},
abstract = {Accurate state of charge (SoC) estimation is vital for safe and efficient operation of lithium-ion batteries. Methods such as Coulomb counting and open-circuit voltage measurements face challenges related to drift and accuracy, especially in large-format cells with spatial gradients in electric vehicles and grid storage usage. This study investigates ultrasonic sensing as a non-invasive and real-time technique for SoC estimation. It explores the opportunity of sensor placement using machine learning models to identify optimal actuator–receiver paths based on signal quality and pinpoints the maximum accuracy that can be achieved for SoC estimation. Based on experimentally collected ultrasound signals transmitted between four sensors installed on a large format pouch cell, a novel and customised deep learning framework enhanced by convolutional neural networks is developed to process ultrasonic signals through transformation to waveform images and leverage transfer learning from strong pre-trained models. The results demonstrate that combining bidirectional signal transmission with a dynamic deep learning-based strategy for actuator and receiver selection significantly enhances the effectiveness of ultrasonic sensing compared to traditional data analysis and pave the way for a robust and scalable SoC monitoring in large-format battery cells. Furthermore, preliminary pathways towards self-supervision are explored by examining the differentiability of ultrasonic signals with respect to SoC, offering a promising route to reduce reliance on conventional ground truths and enhance the scalability of ultrasound-based SoC estimation. The data and source code will be made available at https://github.com/hfarhaditolie/Ultrasonic-SoC.}
}
```

---

## 💬 Feedback & Contact

For questions, collaborations or feedback:

📧 **hamidreza.farhadi-tolie@warwick.ac.uk**  
📧 **h.farhaditolie@gmail.com**

---

## 🙏 Acknowledgements

We acknowledge that the ultrasonic experimental data were originally collected at Imperial College London. We thank the Warwick Manufacturing Group (WMG) and The Faraday Institution for supporting this collaborative research. The deep neural network development and data analysis were carried out at WMG, building upon the foundation established by the experimental work performed at Imperial College London.

---

## 📄 Licence

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ and ultrasonic waves  
<br>
<b>© 2025 Hamidreza Farhadi Tolie</b>
</div>
