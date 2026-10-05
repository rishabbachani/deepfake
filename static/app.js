(() => {
  const $ = (id) => document.getElementById(id);
  const input = $("file-input");
  const dropzone = $("dropzone");
  const analyzeBtn = $("analyze");
  let currentFile = null;

  const show = (el, visible) => { el.hidden = !visible; };
  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  // ---------------------------------------------------------------- health
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      const s = $("status");
      if (h.model_loaded) {
        s.textContent = h.calibrated ? "Model ready · calibrated" : "Model ready";
        s.className = "status status--ok";
      } else {
        s.textContent = "Model not loaded";
        s.className = "status status--bad";
        s.title = h.error || "";
      }
    })
    .catch(() => { $("status").textContent = "Server unreachable"; });

  // ---------------------------------------------------------------- file picking
  function setFile(file) {
    if (!file) return;
    currentFile = file;
    const preview = $("preview");
    preview.innerHTML = "";
    const url = URL.createObjectURL(file);
    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name);
    const el = document.createElement(isVideo ? "video" : "img");
    el.src = url;
    if (isVideo) { el.controls = true; el.muted = true; }
    preview.appendChild(el);

    show($("drop-empty"), false);
    show(preview, true);
    show($("file-row"), true);
    $("file-name").textContent = file.name;
    $("file-meta").textContent = `${isVideo ? "Video" : "Image"} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    resetResult();
  }

  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove("drag");
  }));
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // ---------------------------------------------------------------- analysis
  function resetResult() {
    show($("placeholder"), true);
    show($("spinner"), false);
    show($("error"), false);
    show($("result"), false);
  }

  analyzeBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    if (!currentFile) return;
    show($("placeholder"), false);
    show($("result"), false);
    show($("error"), false);
    show($("spinner"), true);
    analyzeBtn.disabled = true;

    const body = new FormData();
    body.append("file", currentFile);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
      renderResult(data);
    } catch (err) {
      $("error").textContent = err.message;
      show($("error"), true);
    } finally {
      show($("spinner"), false);
      analyzeBtn.disabled = false;
    }
  });

  function renderResult(r) {
    const fake = r.label === "fake";
    const card = $("result");
    card.classList.toggle("is-fake", fake);
    card.classList.toggle("is-real", !fake);

    $("verdict-label").textContent = fake ? "Manipulated" : "Authentic";
    $("verdict-text").textContent = fake
      ? "The classifier found synthesis artefacts in the facial region."
      : "No synthesis artefacts were found in the facial region.";
    $("confidence").textContent = pct(r.confidence);
    $("bar-fill").style.left = pct(r.fake_probability);
    $("pfake").textContent = `P(fake) = ${r.fake_probability.toFixed(3)}`;
    $("faces").textContent = r.faces_scored;
    $("frames").textContent = r.frames_sampled;
    $("detector").textContent = r.detector;
    $("calibrated").textContent = r.calibrated ? "Calibrated" : "Raw sigmoid";
    show($("no-face"), !r.face_found);

    const grid = $("faces-grid");
    grid.innerHTML = "";
    r.face_previews.forEach((src, i) => {
      const img = document.createElement("img");
      img.src = src;
      img.alt = `Face crop ${i + 1}`;
      if (r.frame_scores[i] !== undefined) img.title = `P(fake) = ${r.frame_scores[i].toFixed(3)}`;
      grid.appendChild(img);
    });

    const timeline = $("timeline");
    timeline.innerHTML = "";
    if (r.frame_scores.length > 1) {
      r.frame_scores.forEach((s, i) => {
        const bar = document.createElement("div");
        bar.style.height = `${Math.max(s * 100, 2)}%`;
        bar.style.background = s >= r.threshold ? "var(--fake)" : "var(--real)";
        bar.title = `Face ${i + 1}: P(fake) = ${s.toFixed(3)}`;
        timeline.appendChild(bar);
      });
    }
    show($("timeline-wrap"), r.frame_scores.length > 1);
    show(card, true);
  }
})();
