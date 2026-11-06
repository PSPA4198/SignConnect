# SignConnect — ISL Alphabet HCI Prototype

A React + MediaPipe browser prototype for experimenting with static handshape recognition for the Indian Sign Language (ISL) alphabet.

## Important scope note

This project is a working **prototype**, not a validated ISL translator. It uses MediaPipe Hand Landmarker to detect 21 hand landmarks, then a simple K-nearest-neighbour classifier over examples collected by the user. It does not ship with an authoritative pretrained ISL alphabet model or labelled ISL dataset. The classifier must be trained with representative, correctly labelled examples before recognition is meaningful.

Some letters may involve movement, two hands, facial expressions, or context. A single-hand static landmark classifier cannot reliably recognize all ISL letters. Treat this as an HCI proof of concept and document the limitation in your report.

## Requirements

- Node.js 18+ (20+ recommended)
- Modern browser (Chrome or Edge recommended)
- Webcam
- Internet connection for the first run (MediaPipe WASM and hand landmark model are downloaded from Google's CDN)

## Run locally

```bash
npm install
npm run dev
```

Open the localhost URL printed by Vite and allow camera access. Camera access generally requires `localhost` or HTTPS.

## How to use

1. Click **Start camera** and allow access.
2. Choose a letter in **Teach the alphabet**.
3. Hold the corresponding ISL handshape in view.
4. Click **Collect [letter] samples**. The app saves one normalized landmark sample per video frame while collection is enabled. Keep the handshape steady; stop collection after a few seconds.
5. Repeat from multiple slightly varied positions and lighting conditions. Aim for 10–20 examples per letter as a starting point.
6. Once samples exist, show a handshape. The app displays the closest match.
7. Click **Add detected letter** to append it to the message, or enable automatic addition.
8. Use **Speak** to read the message aloud.
9. Export/import your locally collected training samples as JSON.

## Data and privacy

- Webcam frames are processed in the browser and are not uploaded by this app.
- Training samples (63 normalized numeric landmark values plus label) are stored in browser localStorage.
- Use Export data to back up samples. Reset data deletes local training samples.
- The MediaPipe model and WASM runtime are fetched from external Google/jsDelivr CDNs on first use.

## Model details

- Detector: MediaPipe Tasks Vision `HandLandmarker`.
- Features: 21 landmarks × (x, y, z), centered on wrist and normalized by maximum landmark distance.
- Classifier: 5-nearest-neighbour vote using Euclidean distance in normalized feature space.
- The displayed vote confidence is a relative vote share, not a calibrated probability.

## Suggested HCI evaluation

Test with 5–8 participants. Give them the same short task (e.g. collect samples for a few letters, recognize a test sequence, clear it, and use speech). Record:
- task completion rate
- time to complete
- recognition corrections
- usability feedback (SUS questionnaire or short interview)
- accessibility feedback (contrast, text size, feedback clarity, control discoverability)

Do not claim sign-language translation accuracy without a separate labelled test set and a documented evaluation protocol.
