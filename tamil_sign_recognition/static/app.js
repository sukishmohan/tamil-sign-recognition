/* Tamil Sign Recognition — Web UI JS
   Polls /state every 150 ms and updates the UI accordingly.
*/

const fpsDisplay    = document.getElementById("fps-display");
const statusDot     = document.getElementById("status-dot");
const statusLabel   = document.getElementById("status-label");
const letterDisplay = document.getElementById("letter-display");
const ringFill      = document.getElementById("ring-fill");
const ringText      = document.getElementById("ring-text");
const collectStatus = document.getElementById("collect-status");
const confBar       = document.getElementById("conf-bar");
const confPct       = document.getElementById("conf-pct");
const sequenceGrid  = document.getElementById("sequence-grid");
const clearBtn      = document.getElementById("clear-btn");

// Ring circumference = 2π × r = 2π × 24 ≈ 150.796
const CIRCUMFERENCE = 150.796;

let prevSequenceLen = 0;

async function speakTamil(letter) {
  // Use Google Translate TTS via the Flask backend.
  if (!letter || letter === "—") return;

  console.log(`[SPEAK] Requesting speech for: ${letter}`);
  try {
    const url = `/speak/${encodeURIComponent(letter)}`;
    console.log(`[SPEAK] Fetch URL: ${url}`);
    const resp = await fetch(url);
    console.log(`[SPEAK] Response status: ${resp.status}`);
    if (!resp.ok) {
      const errorText = await resp.text();
      console.error("Speech fetch failed:", resp.status, errorText);
      return;
    }
    const audioBlob = await resp.blob();
    console.log(`[SPEAK] Audio blob size: ${audioBlob.size} bytes`);
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    console.log(`[SPEAK] Playing audio...`);
    audio.play().catch((e) => console.error("Audio play failed:", e));
  } catch (e) {
    console.error("Speech error:", e);
  }
}

function updateRing(count, target) {
  const fraction = Math.min(count / target, 1.0);
  const offset   = CIRCUMFERENCE * (1 - fraction);
  ringFill.style.strokeDashoffset = offset.toFixed(2);
  ringText.textContent = `${count}/${target}`;
}

function updateSequence(seq) {
  if (seq.length === prevSequenceLen) return;

  // Add new items without rebuilding the whole grid
  if (seq.length > prevSequenceLen) {
    for (let i = prevSequenceLen; i < seq.length; i++) {
      // Remove 'latest' from previous last item
      const prev = sequenceGrid.querySelector(".seq-char.latest");
      if (prev) prev.classList.remove("latest");

      const chip = document.createElement("div");
      chip.className = "seq-char latest";
      chip.textContent = seq[i];
      sequenceGrid.appendChild(chip);

      // Speak the Tamil letter using Google Translate TTS
      speakTamil(seq[i]);
    }
    // Scroll to newest
    sequenceGrid.scrollTop = sequenceGrid.scrollHeight;
  } else {
    // Sequence was cleared
    sequenceGrid.innerHTML = "";
  }
  prevSequenceLen = seq.length;
}

async function pollState() {
  try {
    const resp = await fetch("/state");
    if (!resp.ok) throw new Error("bad response");
    const s = await resp.json();

    // FPS
    fpsDisplay.textContent = `${s.fps} fps`;

    // Status dot + label
    if (s.is_background) {
      statusDot.classList.add("offline");
      statusLabel.textContent = "நிறுத்து · Waiting";
    } else {
      statusDot.classList.remove("offline");
      statusLabel.textContent = "வாழும் · Camera Live";
    }

    // Predicted letter
    letterDisplay.textContent = s.letter || "—";

    // Collecting ring
    updateRing(s.collect_step, s.collect_target);

    // Collect status text
    collectStatus.textContent = s.status;

    // Confidence bar
    const pct = Math.round(s.confidence * 100);
    confBar.style.width = pct + "%";
    confPct.textContent  = pct + "%";

    // Sequence
    updateSequence(s.sequence);

  } catch (e) {
    statusDot.classList.add("offline");
    statusLabel.textContent = "இணைப்பு துண்டிக்கப்பட்டது · Disconnected";
  }
}

// Clear button
clearBtn.addEventListener("click", async () => {
  await fetch("/clear", { method: "POST" });
  sequenceGrid.innerHTML = "";
  prevSequenceLen = 0;
});

// Start polling
setInterval(pollState, 150);
pollState();
