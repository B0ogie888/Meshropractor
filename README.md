# 🛠️ Meshropractor

[[русский язык](README_RU.md)]

**Meshropractor** — This is a professional tool for design engineers and technologists, designed for pre-deformation of 3D models and compensation of production errors.

The program allows you to upload a CAD model, overlay an optical scan of the resulting part, and calculate the inverted RBF deformation matrix. This allows you to introduce distortions into the original geometry to compensate for shrinkage, warpage, and metal bulges during subsequent manufacturing.

![PySide6](https://img.shields.io/badge/GUI-PySide6-green?style=flat-square&logo=qt)
![Open3D](https://img.shields.io/badge/Engine-Open3D%20C%2B%2B-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)

## Main use cases

* **Metal 3D printing (SLM/DMLS):** Compensation for temperature fluctuations and internal stresses when printing complex geometries and heat-resistant alloys.
* **Casting and milling:** Taking into account material shrinkage and adjusting molds.
* **Quality control (QC):** Visual comparison (Heatmap) of a real part with the original drawing.

## 🚀 Key functionality

### Step 1: Intelligent Combination (Alignment)

* **Algorithm ICP (Iterative Closest Point):** Highly accurate scan overlay on the CAD model (up to 2000 iterations).
* **Autopilot:** Automatic primary alignment of models by centers of mass.
* **Manual marking:** The ability to install local markers directly on the 3D scene for rough positioning of complex parts.

### Step 2: Pre-deformation calculation (RBF Compensation)

* **Algorithm Raycasting:** Using the Intel/Open3D Tensor Engine for instant laser ray tracing and anomaly detection.
* **Smart Remeshing:** Automatic fragmentation of large CAD mesh polygons to ensure smooth deformation.
* **Mathematic RBF:** Using `SciPy` -based radial basis functions to calculate the counter-distortion matrix. Support for both local and resource-intensive global anti-aliasing.

### Analysis and Visualization

* **Heatmap (Color deviation map):** Calculation of the (Signed Distance) and construction of a gradient heat map of deviations in real time.
* **Modern GUI:** Dark theme, interactive 3D window based on `PyVista/VTK`, flexible layer and transparency control.

---

## ⚙️ Installation and launch

The program requires Python version 3.10 or higher to run.

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/NIKNAME/Meshropractor.git](https://github.com/NIKNAME/Meshropractor.git)
   cd Meshropractor
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # For Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the program:**

   ```bash
   python Meshropractor.py
   ```

*(Note: It is also possible to build a project into a standalone `.exe` file using PyInstaller or auto-py-to-exe).*

---

## 👨‍💻 License and Legal Information

The program uses the official **PySide6** port (LGPL license), making it possible to use this software in closed commercial processes without the need to purchase commercial keys. The mathematical core is built on open-source libraries (MIT/BSD).

## 💰 Support the author

It's tough being a design engineer these days... Sleepless nights before deadlines, liters of energy drinks, the constant struggle with tolerances, and attempts to fit a crooked optical scan onto a perfect CAD model.

If this program has saved you a couple of hours of sleep, a lot of nerves, or prevented an entire batch of parts from being defective at the factory, I'd be grateful for your support! Every penny will go toward the development of useful software.

* **Bank card details:** 2200 1509 5905 0136

📬 Suggestions for improvement and bug reports: <theboogie888@gmail.com>