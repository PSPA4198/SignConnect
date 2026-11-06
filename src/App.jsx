import React, { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import {
  Camera, CameraOff, Volume2, Trash2, RotateCcw, Hand, BookOpen,
  Download, Upload, Play, Pause, CheckCircle2, AlertCircle, Sparkles
} from "lucide-react";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const STORAGE_KEY = "signconnect-isl-training-v1";

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const centered = landmarks.map(p => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);
  const scale = Math.max(...centered.map(p => Math.hypot(p[0], p[1], p[2])), 1e-6);
  return centered.flatMap(p => p.map(v => v / scale));
}
function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum / a.length);
}
function classify(features, samples) {
  if (!samples.length) return null;
  const nearest = samples.map(s => ({ label: s.label, d: distance(features, s.features) }))
    .sort((a,b) => a.d - b.d).slice(0, 5);
  const votes = {};
  nearest.forEach((n, i) => {
    const weight = 1 / (n.d + 0.025);
    votes[n.label] = (votes[n.label] || 0) + weight;
  });
  const ranked = Object.entries(votes).sort((a,b) => b[1]-a[1]);
  const total = ranked.reduce((s, x) => s+x[1], 0);
  return { label: ranked[0][0], confidence: ranked[0][1] / total, distance: nearest[0].d };
}
function loadSamples() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const lastAcceptedRef = useRef({ label: "", at: 0 });
  const lastCollectedRef = useRef(0);
  const [samples, setSamples] = useState(loadSamples);
  const [cameraOn, setCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Camera is off");
  const [prediction, setPrediction] = useState(null);
  const [selectedLetter, setSelectedLetter] = useState("A");
  const [sentence, setSentence] = useState("");
  const [collecting, setCollecting] = useState(false);
  const [autoAdd, setAutoAdd] = useState(false);
  const [threshold, setThreshold] = useState(0.55);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  }, [samples]);

  const startCamera = useCallback(async () => {
    setError(""); setLoading(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires localhost or HTTPS.");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO", numHands: 1
      });
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 960, height: 720 }, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraOn(true); setStatus("Camera ready — show one hand");
    } catch (e) {
      setError(e.message || "Unable to start camera. Check browser permissions and internet connection.");
      stopCamera();
    } finally { setLoading(false); }
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false); setPrediction(null); setStatus("Camera is off");
  }, []);

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    landmarkerRef.current?.close();
  }, []);

  const addSample = useCallback((label, features) => {
    setSamples(prev => [...prev, { label, features, id: `${Date.now()}-${Math.random()}` }]);
  }, []);

  const loop = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !landmarkerRef.current) return;
    const video = videoRef.current, canvas = canvasRef.current;
    if (video.readyState >= 2 && !paused) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const result = landmarkerRef.current.detectForVideo(video, performance.now());
      const hand = result.landmarks?.[0];
      if (hand) {
        const points = hand.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
        ctx.lineWidth = 3; ctx.strokeStyle = "#7c6cff"; ctx.fillStyle = "#b8adff";
        const links = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
        links.forEach(([a,b]) => { ctx.beginPath(); ctx.moveTo(points[a].x, points[a].y); ctx.lineTo(points[b].x, points[b].y); ctx.stroke(); });
        points.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,5,0,Math.PI*2); ctx.fill(); });
        const features = normalizeLandmarks(hand);
        const resultGuess = classify(features, samples);
        if (resultGuess) {
          setPrediction(resultGuess);
          setStatus(`Hand detected · nearest match ${resultGuess.label}`);
          if (collecting && Date.now() - lastCollectedRef.current > 350) { addSample(selectedLetter, features); lastCollectedRef.current = Date.now(); }
          if (autoAdd && resultGuess.confidence >= threshold) {
            const now = Date.now();
            if (lastAcceptedRef.current.label !== resultGuess.label || now - lastAcceptedRef.current.at > 1800) {
              setSentence(s => s + resultGuess.label);
              lastAcceptedRef.current = { label: resultGuess.label, at: now };
            }
          }
        } else {
          setPrediction(null);
          setStatus("Hand detected · add training samples to recognize letters");
        }
      } else {
        setPrediction(null); setStatus("No hand detected — move your hand into view");
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [samples, collecting, selectedLetter, autoAdd, threshold, paused, addSample]);

  useEffect(() => {
    if (cameraOn) rafRef.current = requestAnimationFrame(loop);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [cameraOn, loop]);

  const speak = () => {
    if (!sentence.trim()) return;
    if (!("speechSynthesis" in window)) { setError("Speech synthesis is not supported in this browser."); return; }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sentence.split("").join(" "));
    utterance.lang = "en-IN"; window.speechSynthesis.speak(utterance);
  };
  const clearSentence = () => { setSentence(""); lastAcceptedRef.current = { label: "", at: 0 }; };
  const exportSamples = () => {
    const blob = new Blob([JSON.stringify({ language: "ISL", scope: "alphabet", samples }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = "signconnect-isl-training.json"; a.click(); URL.revokeObjectURL(url);
  };
  const importSamples = async e => {
    const file = e.target.files?.[0]; if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data.samples) || data.samples.some(s => !LETTERS.includes(s.label) || !Array.isArray(s.features) || s.features.length !== 63)) throw new Error("Invalid SignConnect training file.");
      setSamples(data.samples); setError("");
    } catch (err) { setError(err.message); }
    e.target.value = "";
  };
  const countFor = label => samples.filter(s => s.label === label).length;
  const totalTrained = LETTERS.filter(l => countFor(l) > 0).length;

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-icon"><Hand size={22}/></span><span>Sign<span className="accent">Connect</span><small>ISL ALPHABET RECOGNITION</small></span></a>
      <div className="top-actions"><span className="privacy-pill"><span className="green-dot"/> On-device processing</span><span className="version">HCI PROJECT · MVP</span></div>
    </header>
    <main>
      <section className="intro">
        <div><div className="eyebrow"><Sparkles size={14}/> BRIDGING COMMUNICATION</div><h1>Every gesture has<br/><em>a voice.</em></h1><p>An experimental Indian Sign Language alphabet tool. Show a hand sign, see the closest trained letter, and build a message.</p></div>
        <div className="intro-stat"><strong>{samples.length}</strong><span>training examples</span><div className="stat-line"><span style={{width:`${totalTrained/26*100}%`}}/></div><small>{totalTrained}/26 letters trained</small></div>
      </section>

      {error && <div className="error-banner"><AlertCircle size={18}/>{error}<button onClick={()=>setError("")}>Dismiss</button></div>}

      <div className="workspace">
        <section className="panel camera-panel">
          <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Gesture camera</h2></div><span className={`live-status ${cameraOn ? "active":""}`}><i/>{cameraOn ? "LIVE" : "OFFLINE"}</span></div>
          <div className="camera-stage">
            <video ref={videoRef} playsInline muted className={cameraOn ? "visible" : ""}/>
            <canvas ref={canvasRef}/>
            {!cameraOn && <div className="camera-placeholder"><div className="camera-symbol"><Camera size={34}/></div><strong>Your camera preview</strong><span>Start your camera to begin hand tracking</span></div>}
            {cameraOn && <div className="camera-corner">ONE HAND · PALM IN VIEW</div>}
          </div>
          <div className="camera-footer"><div className="status-text"><span className={cameraOn ? "pulse-dot":""}/>{status}</div>
            {!cameraOn ? <button className="primary-btn" onClick={startCamera} disabled={loading}>{loading ? "Starting…" : <><Camera size={17}/> Start camera</>}</button> : <div className="button-row"><button className="secondary-btn" onClick={()=>setPaused(p=>!p)}>{paused ? <Play size={16}/> : <Pause size={16}/>} {paused ? "Resume":"Pause"}</button><button className="danger-btn" onClick={stopCamera}><CameraOff size={16}/> Stop</button></div>}
          </div>
        </section>

        <section className="panel result-panel">
          <div className="panel-heading"><div><span className="section-kicker">02 / RECOGNITION</span><h2>Detected letter</h2></div><span className="tiny-label">KNN CLASSIFIER</span></div>
          <div className="prediction-box"><div className="prediction-letter">{prediction && prediction.confidence >= threshold ? prediction.label : "—"}</div><div className="prediction-meta"><strong>{prediction ? (prediction.confidence >= threshold ? "Possible match" : "Low confidence") : "Waiting for gesture"}</strong><span>{prediction ? `${Math.round(prediction.confidence*100)}% vote confidence` : "Place your hand in the camera frame"}</span></div></div>
          <div className="confidence-track"><span style={{width:`${Math.min(100,(prediction?.confidence || 0)*100)}%`}}/></div>
          <div className="threshold-row"><label htmlFor="threshold">Match threshold</label><strong>{Math.round(threshold*100)}%</strong></div>
          <input id="threshold" className="range" type="range" min=".35" max=".95" step=".05" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/>
          <p className="hint">A higher threshold is stricter. Confidence is a nearest-neighbour vote, not a calibrated probability.</p>
          <button className={`add-letter-btn ${prediction && prediction.confidence >= threshold ? "ready":""}`} disabled={!prediction || prediction.confidence < threshold} onClick={()=>setSentence(s=>s+prediction.label)}><CheckCircle2 size={17}/> Add detected letter to message</button>
          <div className="divider-line"/>
          <div className="panel-heading compact"><div><span className="section-kicker">03 / YOUR MESSAGE</span><h2>Build a message</h2></div></div>
          <div className="message-output">{sentence || <span>Your recognized letters will appear here…</span>}</div>
          <div className="button-row message-actions"><button className="primary-btn" onClick={speak} disabled={!sentence}><Volume2 size={17}/> Speak</button><button className="secondary-btn" onClick={clearSentence}><Trash2 size={16}/> Clear</button><button className="secondary-btn" onClick={()=>setSentence(s=>s+" ")}><span>␣</span> Space</button></div>
          <label className="auto-add"><input type="checkbox" checked={autoAdd} onChange={e=>setAutoAdd(e.target.checked)}/> Automatically add stable predictions <small>(may repeat letters)</small></label>
        </section>
      </div>

      <section className="panel training-panel">
        <div className="panel-heading"><div><span className="section-kicker">04 / PERSONAL TRAINING</span><h2>Teach the alphabet</h2><p className="panel-subtitle">Collect examples of each ISL letter using your camera. Aim for 10–20 varied samples per letter.</p></div><span className="training-count">{samples.length} samples</span></div>
        <div className="training-content">
          <div className="letter-picker">{LETTERS.map(letter=><button key={letter} className={selectedLetter===letter?"selected":""} onClick={()=>{setSelectedLetter(letter);setCollecting(false);}}><strong>{letter}</strong><small>{countFor(letter)}</small></button>)}</div>
          <div className="training-controls"><div><strong>Collect samples for <b>{selectedLetter}</b></strong><span>{countFor(selectedLetter)} examples saved for this letter</span></div>
            <button className={collecting?"recording-btn":"primary-btn"} disabled={!cameraOn} onClick={()=>setCollecting(c=>!c)}>{collecting ? <><Pause size={16}/> Stop collecting</> : <><Sparkles size={16}/> {`Collect ${selectedLetter} samples`}</>}</button>
          </div>
          {collecting && <div className="collect-notice"><span className="pulse-dot"/> Collecting a sample on each camera frame. Hold the selected handshape steady, then stop after several examples.</div>}
          <div className="training-bottom"><div className="progress-wrap"><div><span>Alphabet coverage</span><strong>{totalTrained}/26</strong></div><div className="progress-bar"><span style={{width:`${totalTrained/26*100}%`}}/></div></div><div className="button-row"><button className="secondary-btn" onClick={exportSamples} disabled={!samples.length}><Download size={16}/> Export data</button><label className="upload-btn"><Upload size={16}/> Import data<input type="file" accept="application/json" onChange={importSamples}/></label><button className="danger-outline-btn" onClick={()=>{if(window.confirm("Delete all locally stored training samples?"))setSamples([]);}} disabled={!samples.length}><RotateCcw size={16}/> Reset data</button></div></div>
        </div>
      </section>

      <section className="info-strip"><div className="info-icon"><BookOpen size={20}/></div><div><strong>Prototype limitation</strong><p>This MVP detects 21 hand landmarks and classifies them using examples you collect. It is not yet a validated ISL translator. Some ISL letters involve movement, two hands, facial expressions, or context and cannot be reliably identified from a single static hand pose.</p></div></section>
    </main>
    <footer><span>SignConnect · HCI Project Prototype</span><span>Built for learning, accessibility & inclusive design</span></footer>
  </div>;
}