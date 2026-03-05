# EMG-Controlled Hidden Blade – V1

This is a functional prototype of a wearable, spring-loaded, servo-triggered arm blade inspired by the Assassin's Creed hidden blade — controlled using EMG (electromyography) signals from forearm muscle flexion.

## 🧠 Concept

- Blade is mounted on a sled inside a forearm-mounted housing
- A compression spring pushes the sled/blade forward
- A servo-controlled latch holds the sled in place until triggered
- EMG signal (via MyoWare 2.0 sensor) detects muscle flex
- When threshold is passed, servo releases latch and blade extends

## ⚙️ Current Version

- ✅ Spring-based extension only
- ❌ No automatic retraction (planned for V2)
- ✅ EMG-to-servo prototype logic
- ✅ Breadboarded servo response with simulated input
- ✅ **BiLSTM-CNN gesture classification system**
- ✅ **Real-time web dashboard for EMG visualization**
- 🛠️ In-progress Onshape design for housing and blade sled

## 📐 Hardware Plan

| Component        | Status      |
|------------------|-------------|
| MyoWare 2.0 EMG Sensor | Ordered |
| Servo Motor       | Ready       |
| Spring System     | Designing   |
| Blade Sled & Housing | CAD WIP   |
| Microcontroller   | Arduino Uno    |

## 💻 Software Architecture

### Arduino Firmware (`Release.ino`)
Simple EMG threshold detection for servo trigger control.

### Python Gesture Recognition (`emg_gesture/`)
Advanced BiLSTM-CNN neural network for multi-gesture classification:

```
emg_gesture/
├── data/           # EMG collection & preprocessing
├── model/          # BiLSTM-CNN architecture & training
├── inference/      # Real-time classification & actuator control
├── frontend/       # Web dashboard with live visualization
├── train.py        # Train the model
├── run.py          # Run real-time inference
└── dashboard.py    # Launch web dashboard
```

**Features:**
- 8-channel EMG signal processing (bandpass filter, 60Hz notch)
- Time-domain feature extraction (MAV, RMS, WL, ZC, SSC)
- Hybrid CNN (spatial) + BiLSTM (temporal) architecture
- Real-time gesture classification: closed_hand, open_hand, pointing
- Serial actuator control with gesture-to-command mapping
- Cyberpunk-themed web dashboard

### Quick Start
```bash
cd emg_gesture
pip install -r requirements.txt

# Train model (generates mock data for testing)
python train.py --generate-mock --epochs 20

# Launch dashboard
python dashboard.py

# Run CLI inference
python run.py --mock
```

## 🛠️ Build Goals

- [x] CAD housing + sled
- [x] Design spring + latch system
- [x] Wire EMG sensor to controller
- [x] Test trigger + safety threshold logic
- [x] Build gesture classification model
- [x] Create real-time dashboard
- [ ] Final assembly and arm mounting
- [ ] Integrate classifier with Arduino

## 📸 Progress Log

### Week 1 – Pre-Build Prep
- Created repo, sketched sled/latch system
- Simulated EMG input using potentiometer
- Breadboarded servo control logic
- Created Onshape base housing model

### Week 2 – Software Development
- Built complete EMG gesture recognition pipeline
- Implemented BiLSTM-CNN hybrid model
- Created real-time web dashboard
- Added mock mode for hardware-free testing

## 📦 Parts List

- [MyoWare 2.0 Muscle Sensor Basic Kit](https://www.sparkfun.com/products/19166)
- 3M Red Dot Electrodes
- SG90 or MG90S Servo
- Springs (compression, medium stiffness)
- Arduino Uno

---

## 🤖 Future Plans

- [ ] Add automatic retraction (winch or linear actuator)
- [ ] Integrate battery + wireless control
- [ ] Create 3D printed wearable housing
- [x] ~~Explore gesture-based EMG classification~~ ✅ Done!
- [ ] Train on real EMG data from MyoWare sensor

---

## ⚠️ Disclaimer

**This project is for educational and prototyping purposes only.** It uses a spring-loaded mechanism and should be handled with extreme care. Do not use it in public or without proper safety measures.
