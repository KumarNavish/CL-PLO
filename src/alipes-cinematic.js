const FRAMES = [
  {
    src: "./assets/alipes-cinematic/frame-01.png",
    title: "Portfolio optimization",
    subtitle: "signals -> constrained weights -> executable trades",
  },
  {
    src: "./assets/alipes-cinematic/frame-02.png",
    title: "Portfolio optimization",
    subtitle: "same contract in research -> production",
  },
  {
    src: "./assets/alipes-cinematic/frame-03.png",
    title: "Optimization is constraint geometry",
    subtitle: "the portfolio is the closest feasible answer to the alpha push",
  },
  {
    src: "./assets/alipes-cinematic/frame-04.png",
    title: "Optimization is constraint geometry",
    subtitle: "alphas create the desired push; constraints shape feasibility",
  },
  {
    src: "./assets/alipes-cinematic/frame-05.png",
    title: "Optimization is constraint geometry",
    subtitle: "the optimizer projects to the nearest feasible portfolio",
  },
  {
    src: "./assets/alipes-cinematic/frame-06.png",
    title: "Markets drift. Stress returns.",
    subtitle: "signals need updates while constraints still hold",
  },
  {
    src: "./assets/alipes-cinematic/frame-07.png",
    title: "Markets drift. Stress returns.",
    subtitle: "naive adaptation can increase drawdown without stress memory",
  },
  {
    src: "./assets/alipes-cinematic/frame-08.png",
    title: "Decision Spine (original intent)",
    subtitle: "same model, same stream - only the update rule changes",
  },
  {
    src: "./assets/alipes-cinematic/frame-09.png",
    title: "Decision Spine (original intent)",
    subtitle: "replay adds stress fixtures to the same live flow",
  },
  {
    src: "./assets/alipes-cinematic/frame-10.png",
    title: "Decision Spine (original intent)",
    subtitle: "hybrid adds projection to keep updates risk-safe",
  },
  {
    src: "./assets/alipes-cinematic/frame-11.png",
    title: "Update = constrained move",
    subtitle: "project the learning step into a risk-safe set",
  },
  {
    src: "./assets/alipes-cinematic/frame-12.png",
    title: "Update = constrained move",
    subtitle: "combine live and anchor gradients, then project",
  },
  {
    src: "./assets/alipes-cinematic/frame-13.png",
    title: "Anchors = stress fixtures",
    subtitle: "replay keeps the policy de-risking in stress",
  },
  {
    src: "./assets/alipes-cinematic/frame-14.png",
    title: "Release gate for model updates",
    subtitle: "accept only if anchors + risk + cost checks pass",
  },
  {
    src: "./assets/alipes-cinematic/frame-15.png",
    title: "Fits a quant portfolio stack",
    subtitle: "feature store -> sim -> risk+cost -> optimizer -> execution",
  },
  {
    src: "./assets/alipes-cinematic/frame-16.png",
    title: "How I would ship this (humbly)",
    subtitle: "instrument, anchor, hybrid gate, then staged release",
  },
];

const FRAME_MS = 1350;

const imgA = document.getElementById("frame-a");
const imgB = document.getElementById("frame-b");
const titleEl = document.getElementById("frame-title");
const subtitleEl = document.getElementById("frame-subtitle");
const stepEl = document.getElementById("frame-step");
const rangeEl = document.getElementById("frame-range");
const playBtn = document.getElementById("play-btn");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const speedEl = document.getElementById("speed-select");

let index = 0;
let playing = true;
let timer = null;
let frontIsA = true;

function preloadFrames() {
  for (const frame of FRAMES) {
    const img = new Image();
    img.src = frame.src;
  }
}

function getBackImage() {
  return frontIsA ? imgB : imgA;
}

function getFrontImage() {
  return frontIsA ? imgA : imgB;
}

function setCaption(i) {
  const frame = FRAMES[i];
  titleEl.textContent = frame.title;
  subtitleEl.textContent = frame.subtitle;
  stepEl.textContent = `Frame ${i + 1} / ${FRAMES.length}`;
  rangeEl.value = String(i);
}

function showFrame(i, immediate = false) {
  const safe = ((i % FRAMES.length) + FRAMES.length) % FRAMES.length;
  index = safe;

  const src = FRAMES[safe].src;
  const back = getBackImage();
  const front = getFrontImage();

  back.src = src;

  if (immediate) {
    front.classList.remove("frame-show");
    front.classList.add("frame-hide");
    back.classList.remove("frame-hide");
    back.classList.add("frame-show");
    frontIsA = !frontIsA;
    setCaption(safe);
    return;
  }

  requestAnimationFrame(() => {
    front.classList.remove("frame-show");
    front.classList.add("frame-hide");
    back.classList.remove("frame-hide");
    back.classList.add("frame-show");
    frontIsA = !frontIsA;
    setCaption(safe);
  });
}

function stopLoop() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

function startLoop() {
  stopLoop();
  if (!playing) {
    return;
  }
  const speed = Number(speedEl.value || "1");
  const delay = Math.max(350, Math.floor(FRAME_MS / speed));
  timer = setTimeout(() => {
    showFrame(index + 1);
    startLoop();
  }, delay);
}

function setPlaying(next) {
  playing = Boolean(next);
  playBtn.textContent = playing ? "Pause" : "Play";
  startLoop();
}

function bind() {
  playBtn.addEventListener("click", () => {
    setPlaying(!playing);
  });

  prevBtn.addEventListener("click", () => {
    showFrame(index - 1);
    if (playing) {
      startLoop();
    }
  });

  nextBtn.addEventListener("click", () => {
    showFrame(index + 1);
    if (playing) {
      startLoop();
    }
  });

  rangeEl.addEventListener("input", () => {
    showFrame(Number(rangeEl.value), true);
    if (playing) {
      startLoop();
    }
  });

  speedEl.addEventListener("change", () => {
    if (playing) {
      startLoop();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      setPlaying(!playing);
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      showFrame(index + 1);
      if (playing) {
        startLoop();
      }
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      showFrame(index - 1);
      if (playing) {
        startLoop();
      }
    }
  });
}

function init() {
  rangeEl.max = String(FRAMES.length - 1);
  imgA.src = FRAMES[0].src;
  imgB.src = FRAMES[0].src;
  setCaption(0);
  preloadFrames();
  bind();
  startLoop();
}

init();
