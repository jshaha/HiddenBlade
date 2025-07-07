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
- 🛠️ In-progress Onshape design for housing and blade sled

## 📐 Hardware Plan

| Component        | Status      |
|------------------|-------------|
| MyoWare 2.0 EMG Sensor | Ordered |
| Servo Motor       | Ready       |
| Spring System     | Designing   |
| Blade Sled & Housing | CAD WIP   |
| Microcontroller   | Arduino Uno    |

## 💻 Software Flow

1. Read analog EMG signal
2. Apply threshold filtering
3. If flex > threshold, trigger servo release
4. Reset by pushing blade back manually

## 🛠️ Build Goals

- [ ] CAD housing + sled
- [ ] Design spring + latch system
- [ ] Wire EMG sensor to controller
- [ ] Test trigger + safety threshold logic
- [ ] Final assembly and arm mounting

## 📸 Progress Log

### Week 1 – Pre-Build Prep
- Created repo, sketched sled/latch system
- Simulated EMG input using potentiometer
- Breadboarded servo control logic
- Created Onshape base housing model


## 📦 Parts List

- [MyoWare 2.0 Muscle Sensor Basic Kit](https://www.sparkfun.com/products/19166)
- 3M Red Dot Electrodes
- SG90 or MG90S Servo
- Springs (compression, medium stiffness)
- Aruduino Uno

---

## 🤖 Future Plans

- [ ] Add automatic retraction (winch or linear actuator)
- [ ] Integrate battery + wireless control
- [ ] Create 3D printed wearable housing
- [ ] Explore gesture-based EMG classification

---

## ⚠️ Disclaimer

**This project is for educational and prototyping purposes only.** It uses a spring-loaded mechanism and should be handled with extreme care. Do not use it in public or without proper safety measures.
