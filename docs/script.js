const progress = document.querySelector(".scroll-progress");
const milestoneToast = document.querySelector(".milestone-toast");
const finishCelebration = document.querySelector(".finish-celebration");
const finishMessage = document.querySelector(".finish-message");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let confettiDropped = false;
let finishCelebrated = false;
let milestoneToastTimeout;
let finishCleanupTimeout;

function updateProgress() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
  progress.style.width = `${pct}%`;

  if (!prefersReducedMotion && !confettiDropped && pct >= 50) {
    confettiDropped = true;
    showMilestoneToast();
    dropConfetti();
  }

  if (!prefersReducedMotion && !finishCelebrated && pct >= 98) {
    finishCelebrated = true;
    showFinishCelebration();
  }
}

window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("resize", updateProgress);
updateProgress();

const animatedItems = document.querySelectorAll("[data-animate]");

if (prefersReducedMotion) {
  animatedItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  animatedItems.forEach((item) => observer.observe(item));
}

function showMilestoneToast() {
  if (!milestoneToast) return;

  milestoneToast.hidden = false;
  milestoneToast.textContent = "50% through the page!";
  milestoneToast.classList.add("is-visible");
  window.clearTimeout(milestoneToastTimeout);
  milestoneToastTimeout = window.setTimeout(() => {
    milestoneToast.classList.remove("is-visible");
    milestoneToast.textContent = "";
    milestoneToast.hidden = true;
  }, 2000);
}

function showFinishCelebration() {
  if (!finishCelebration) return;

  window.clearTimeout(milestoneToastTimeout);
  window.clearTimeout(finishCleanupTimeout);
  if (milestoneToast) {
    milestoneToast.textContent = "";
    milestoneToast.hidden = true;
    milestoneToast.classList.remove("is-visible");
  }
  if (finishMessage) {
    finishMessage.textContent = "You made it!";
  }
  finishCelebration.hidden = false;
  finishCelebration.classList.remove("is-visible");
  void finishCelebration.offsetWidth;
  finishCelebration.classList.add("is-visible");
  finishCleanupTimeout = window.setTimeout(() => {
    finishCelebration.classList.remove("is-visible");
    finishCelebration.hidden = true;
    if (finishMessage) {
      finishMessage.textContent = "";
    }
  }, 5600);
}

function dropConfetti() {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  if (!context) return;

  canvas.className = "confetti-canvas";
  canvas.setAttribute("aria-hidden", "true");
  document.body.appendChild(canvas);

  const colors = ["#0e7c78", "#245bb3", "#c85f13", "#2e7d52", "#b98705", "#17212b"];
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pieceCount = window.innerWidth < 620 ? 72 : 132;
  const pieces = Array.from({ length: pieceCount }, () => ({
    x: Math.random() * window.innerWidth,
    y: -Math.random() * window.innerHeight * 0.45,
    width: 5 + Math.random() * 7,
    height: 8 + Math.random() * 10,
    color: colors[Math.floor(Math.random() * colors.length)],
    velocityX: -2.2 + Math.random() * 4.4,
    velocityY: 2.5 + Math.random() * 4.2,
    rotation: Math.random() * Math.PI,
    spin: -0.18 + Math.random() * 0.36,
    wobble: Math.random() * Math.PI * 2,
  }));

  function resizeCanvas() {
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function drawPiece(piece, opacity) {
    context.save();
    context.globalAlpha = opacity;
    context.translate(piece.x, piece.y);
    context.rotate(piece.rotation);
    context.fillStyle = piece.color;
    context.fillRect(-piece.width / 2, -piece.height / 2, piece.width, piece.height);
    context.restore();
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  const duration = 3600;
  const startedAt = performance.now();

  function animate(now) {
    const elapsed = now - startedAt;
    const fadeStart = duration * 0.72;
    const opacity = elapsed > fadeStart ? Math.max(0, 1 - (elapsed - fadeStart) / (duration - fadeStart)) : 1;

    context.clearRect(0, 0, window.innerWidth, window.innerHeight);

    pieces.forEach((piece) => {
      piece.x += piece.velocityX + Math.sin(elapsed / 180 + piece.wobble) * 0.45;
      piece.y += piece.velocityY;
      piece.velocityY += 0.018;
      piece.rotation += piece.spin;
      drawPiece(piece, opacity);
    });

    if (elapsed < duration) {
      window.requestAnimationFrame(animate);
      return;
    }

    window.removeEventListener("resize", resizeCanvas);
    canvas.remove();
  }

  window.requestAnimationFrame(animate);
}

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    const originalText = button.textContent;

    try {
      await navigator.clipboard.writeText(target.innerText.trim());
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = originalText;
      }, 1400);
    } catch {
      button.textContent = "Select BibTeX";
      window.setTimeout(() => {
        button.textContent = originalText;
      }, 1400);
    }
  });
});
