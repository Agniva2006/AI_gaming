// ===== 2D FOOTBALL DASHBOARD CLIENT =====
const API_BASE = window.location.origin.includes("http") ? window.location.origin : "http://127.0.0.1:8000";
let ws = null;
let rewardChart = null;
let lossChart = null;

// Formation Editor State
const canvas = document.getElementById("pitchCanvas");
const ctx = canvas ? canvas.getContext("2d") : null;

let currentFormationType = "4-3-3";
let players = [];
let draggingPlayer = null;

const DEFAULT_PRESETS = {
  "4-3-3": [
    { role: "GK", x: 0.05, y: 0.50 },
    { role: "LB", x: 0.20, y: 0.15 },
    { role: "CB", x: 0.15, y: 0.35 },
    { role: "CB", x: 0.15, y: 0.65 },
    { role: "RB", x: 0.20, y: 0.85 },
    { role: "CDM", x: 0.32, y: 0.50 },
    { role: "LCM", x: 0.45, y: 0.30 },
    { role: "RCM", x: 0.45, y: 0.70 },
    { role: "LW", x: 0.70, y: 0.15 },
    { role: "ST", x: 0.75, y: 0.50 },
    { role: "RW", x: 0.70, y: 0.85 }
  ],
  "4-4-2": [
    { role: "GK", x: 0.05, y: 0.50 },
    { role: "LB", x: 0.20, y: 0.15 },
    { role: "CB", x: 0.15, y: 0.35 },
    { role: "CB", x: 0.15, y: 0.65 },
    { role: "RB", x: 0.20, y: 0.85 },
    { role: "LM", x: 0.40, y: 0.15 },
    { role: "CM", x: 0.38, y: 0.38 },
    { role: "CM", x: 0.38, y: 0.62 },
    { role: "RM", x: 0.40, y: 0.85 },
    { role: "ST", x: 0.72, y: 0.40 },
    { role: "ST", x: 0.72, y: 0.60 }
  ],
  "3-5-2": [
    { role: "GK", x: 0.05, y: 0.50 },
    { role: "LCB", x: 0.18, y: 0.25 },
    { role: "CB", x: 0.14, y: 0.50 },
    { role: "RCB", x: 0.18, y: 0.75 },
    { role: "LWB", x: 0.38, y: 0.12 },
    { role: "CM", x: 0.35, y: 0.36 },
    { role: "CDM", x: 0.30, y: 0.50 },
    { role: "CM", x: 0.35, y: 0.64 },
    { role: "RWB", x: 0.38, y: 0.88 },
    { role: "ST", x: 0.73, y: 0.40 },
    { role: "ST", x: 0.73, y: 0.60 }
  ],
  "5-3-2": [
    { role: "GK", x: 0.05, y: 0.50 },
    { role: "LWB", x: 0.20, y: 0.12 },
    { role: "CB", x: 0.16, y: 0.30 },
    { role: "CB", x: 0.14, y: 0.50 },
    { role: "CB", x: 0.16, y: 0.70 },
    { role: "RWB", x: 0.20, y: 0.88 },
    { role: "LCM", x: 0.42, y: 0.30 },
    { role: "CM", x: 0.38, y: 0.50 },
    { role: "RCM", x: 0.42, y: 0.70 },
    { role: "ST", x: 0.72, y: 0.42 },
    { role: "ST", x: 0.72, y: 0.58 }
  ],
  "4-2-3-1": [
    { role: "GK", x: 0.05, y: 0.50 },
    { role: "LB", x: 0.20, y: 0.15 },
    { role: "CB", x: 0.15, y: 0.35 },
    { role: "CB", x: 0.15, y: 0.65 },
    { role: "RB", x: 0.20, y: 0.85 },
    { role: "LDM", x: 0.32, y: 0.38 },
    { role: "RDM", x: 0.32, y: 0.62 },
    { role: "LAM", x: 0.55, y: 0.20 },
    { role: "CAM", x: 0.52, y: 0.50 },
    { role: "RAM", x: 0.55, y: 0.80 },
    { role: "ST", x: 0.75, y: 0.50 }
  ]
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initWebSocket();
  initFormationCanvas();
  initCharts();
  loadMatchSummary();
  loadMatchHistory();
  loadAIStats();
  loadFormations();
  checkAuth();
});

// Toast Notifications
function showToast(msg) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.innerText = msg;
  toast.style.display = "block";
  setTimeout(() => { toast.style.display = "none"; }, 3500);
}

// Tab Navigation
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Redraw canvas or resize charts if active
      if (targetId === "tab-formation") {
        resizeCanvas();
        drawPitch();
      } else if (targetId === "tab-ai") {
        loadAIStats();
      }
    });
  });
}

// WebSocket connection for real-time live events
function initWebSocket() {
  const wsUrl = API_BASE.replace(/^http/, "ws") + "/ws/live";
  try {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      document.getElementById("statusDot").style.background = "#10b981";
      document.getElementById("statusText").innerText = "Online";
    };
    ws.onclose = () => {
      document.getElementById("statusDot").style.background = "#ef4444";
      document.getElementById("statusText").innerText = "Reconnecting...";
      setTimeout(initWebSocket, 3000);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "match_recorded") {
          showToast(`⚽ Match Finished! Score: YOU ${data.human_score} - ${data.ai_score} RL AI (${data.result})`);
          loadMatchSummary();
          loadMatchHistory();
          loadAIStats();
        } else if (data.type === "telemetry") {
          updateChartData(data.reward, data.loss);
        }
      } catch (e) {}
    };
  } catch (err) {}
}

// ===== MATCH CENTER =====
async function loadMatchSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/matches/summary`);
    const json = await res.json();
    if (json.success && json.summary) {
      const s = json.summary;
      document.getElementById("totalMatches").innerText = s.total_matches || 0;
      document.getElementById("humanWins").innerText = s.human_wins || 0;
      document.getElementById("aiWins").innerText = s.ai_wins || 0;
      document.getElementById("matchDraws").innerText = s.draws || 0;

      const total = s.total_matches || 0;
      const rate = total > 0 ? Math.round((s.human_wins / total) * 100) : 0;
      document.getElementById("humanWinRate").innerText = rate + "%";
    }
  } catch (e) {}
}

async function loadMatchHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/matches/history?limit=25`);
    const json = await res.json();
    
    const summaryBody = document.getElementById("matchesTableBody");
    const fullBody = document.getElementById("fullMatchesTableBody");

    if (json.matches && json.matches.length > 0) {
      const latest = json.matches[0];
      renderShotMap(latest.shots_data || []);
      renderAIAssessment(latest.ai_adaptation || {});
    } else {
      renderShotMap([]);
    }

    if (summaryBody) {
      summaryBody.innerHTML = "";
      if (!json.matches || json.matches.length === 0) {
        summaryBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:24px;">No match records found yet. Launch Pygame to play against the RL AI!</td></tr>`;
      } else {
        json.matches.slice(0, 8).forEach(m => {
          const badgeClass = m.result === "WIN" ? "badge-win" : (m.result === "LOSS" ? "badge-loss" : "badge-draw");
          const row = document.createElement("tr");
          row.style.cursor = "pointer";
          row.onclick = () => {
            renderShotMap(m.shots_data || []);
            renderAIAssessment(m.ai_adaptation || {});
            showToast(`Loaded analytics for Match #${m.id}`);
          };
          const xgHuman = (m.xg_human != null ? m.xg_human : 0).toFixed(2);
          const xgAi = (m.xg_ai != null ? m.xg_ai : 0).toFixed(2);
          const tactic = m.tactical_style || "Balanced";
          row.innerHTML = `
            <td>#${m.id}</td>
            <td><strong>YOU ${m.human_score} - ${m.ai_score} AI</strong></td>
            <td><span class="xg-badge">${xgHuman} - ${xgAi}</span></td>
            <td><span class="badge-result ${badgeClass}">${m.result}</span></td>
            <td><span class="badge-tactic">${tactic}</span></td>
            <td>${m.possession_human}% - ${m.possession_ai}%</td>
            <td>${m.shots_human} / ${m.shots_ai}</td>
            <td style="color:var(--text-muted)">${new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
          `;
          summaryBody.appendChild(row);
        });
      }
    }

    if (fullBody) {
      fullBody.innerHTML = "";
      if (!json.matches || json.matches.length === 0) {
        fullBody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--text-muted);padding:24px;">No match records found yet. Launch Pygame to play against the RL AI!</td></tr>`;
      } else {
        json.matches.forEach(m => {
          const badgeClass = m.result === "WIN" ? "badge-win" : (m.result === "LOSS" ? "badge-loss" : "badge-draw");
          const row = document.createElement("tr");
          row.style.cursor = "pointer";
          row.onclick = () => {
            renderShotMap(m.shots_data || []);
            renderAIAssessment(m.ai_adaptation || {});
            showToast(`Loaded analytics for Match #${m.id}`);
          };
          const xgHuman = (m.xg_human != null ? m.xg_human : 0).toFixed(2);
          const xgAi = (m.xg_ai != null ? m.xg_ai : 0).toFixed(2);
          const tactic = m.tactical_style || "Balanced";
          row.innerHTML = `
            <td>#${m.id}</td>
            <td><strong>YOU ${m.human_score} - ${m.ai_score} AI</strong></td>
            <td><span class="xg-badge">${xgHuman} - ${xgAi}</span></td>
            <td><span class="badge-result ${badgeClass}">${m.result}</span></td>
            <td><span class="badge-tactic">${tactic}</span></td>
            <td>${m.human_formation} vs ${m.ai_formation}</td>
            <td>${m.possession_human}% - ${m.possession_ai}%</td>
            <td>${m.shots_human} / ${m.shots_ai}</td>
            <td style="color:var(--text-muted)">${new Date(m.created_at).toLocaleString()}</td>
          `;
          fullBody.appendChild(row);
        });
      }
    }
  } catch (e) {
    console.error("Failed to load match history:", e);
  }
}

// ===== SHOT MAP & AI ADAPTATION RENDERING =====
function renderShotMap(shots) {
  const canvas = document.getElementById("shotMapCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * (window.devicePixelRatio || 1);
  canvas.height = rect.height * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  const w = rect.width;
  const h = rect.height;

  // Background turf
  ctx.fillStyle = "#0c1712";
  ctx.fillRect(0, 0, w, h);

  // Pitch outline
  ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(8, 8, w - 16, h - 16);

  // Halfway line & Center Circle
  ctx.beginPath();
  ctx.moveTo(w / 2, 8);
  ctx.lineTo(w / 2, h - 8);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(w / 2, h / 2, h * 0.22, 0, Math.PI * 2);
  ctx.stroke();

  // Penalty areas
  const boxW = w * 0.16;
  const boxH = h * 0.54;
  ctx.strokeRect(8, (h - boxH) / 2, boxW, boxH);
  ctx.strokeRect(w - 8 - boxW, (h - boxH) / 2, boxW, boxH);

  // Draw Shots
  if (!shots || shots.length === 0) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No shot attempts recorded for this match.", w / 2, h / 2 + 4);
    return;
  }

  // World pitch is 1280 x 720
  const scaleX = (w - 16) / 1280.0;
  const scaleY = (h - 16) / 720.0;

  shots.forEach((s) => {
    const px = 8 + (s.x || 0) * scaleX;
    const py = 8 + (s.y || 0) * scaleY;
    const xgVal = s.xg || 0.1;
    const r = Math.max(5, Math.min(22, 6 + xgVal * 20));

    const isGoal = s.result === "GOAL";
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = isGoal ? "rgba(34, 197, 94, 0.85)" : "rgba(239, 68, 68, 0.75)";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Shot xG label
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(xgVal.toFixed(2), px, py);
  });
}

function renderAIAssessment(adaptation) {
  if (!adaptation) return;
  const tendency = adaptation.tendency_summary || {};
  const counter = adaptation.counter_strategy || {};
  const flanks = tendency.flank_percentages || { left: 33.3, center: 33.4, right: 33.3 };

  // Flank Bar
  const barL = document.getElementById("flankBarLeft");
  const barC = document.getElementById("flankBarCenter");
  const barR = document.getElementById("flankBarRight");
  const biasText = document.getElementById("flankBiasText");

  if (barL && barC && barR) {
    barL.style.width = (flanks.left || 33.3) + "%";
    barC.style.width = (flanks.center || 33.4) + "%";
    barR.style.width = (flanks.right || 33.3) + "%";
  }
  if (biasText) {
    biasText.innerText = `Left ${flanks.left || 0}% | Center ${flanks.center || 0}% | Right ${flanks.right || 0}%`;
  }

  // Pass style
  const passStyleEl = document.getElementById("passStyleDetected");
  const throughEl = document.getElementById("throughBallShare");
  if (passStyleEl) passStyleEl.innerText = tendency.pass_style || "Balanced Buildup";
  if (throughEl) throughEl.innerText = (tendency.through_ball_pct || 0) + "%";

  // Counter badge & debrief
  const badgeEl = document.getElementById("aiCounterBadge");
  const debriefEl = document.getElementById("aiDebriefText");
  if (badgeEl) badgeEl.innerText = counter.strategy_name || "Standard Scout";
  if (debriefEl) debriefEl.innerText = counter.tactical_debrief || "AI observing human movement patterns.";
}

// ===== TACTICAL PLAYBOOK =====
let currentTacticalStyle = "Balanced";

function selectTacticalStyle(styleName) {
  currentTacticalStyle = styleName;
  document.querySelectorAll("#tab-playbook .stat-item").forEach(el => {
    el.style.borderColor = "var(--glass-border)";
    el.style.boxShadow = "none";
  });
  if (event && event.currentTarget) {
    event.currentTarget.style.borderColor = "var(--accent-cyan)";
    event.currentTarget.style.boxShadow = "0 0 16px rgba(56, 189, 248, 0.25)";
  }
  showToast(`🎯 Tactical style set to '${styleName}'. In Pygame, press keys [1-5] during a match to switch real-time!`);
}

// ===== AI LEARNING CENTER =====
function initCharts() {
  const rewardCtx = document.getElementById("rewardChart");
  const lossCtx = document.getElementById("lossChart");
  if (!rewardCtx || !lossCtx) return;

  rewardChart = new Chart(rewardCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "AI Match Reward (Goal, Progress, Shots)",
        data: [],
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56, 189, 248, 0.1)",
        borderWidth: 2,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
      }
    }
  });

  lossChart = new Chart(lossCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "PPO Policy & Value Loss",
        data: [],
        borderColor: "#ef4444",
        backgroundColor: "rgba(239, 68, 68, 0.1)",
        borderWidth: 2,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } }
      }
    }
  });
}

async function loadAIStats() {
  try {
    const res = await fetch(`${API_BASE}/api/ai/stats`);
    const json = await res.json();
    if (json.success && json.stats) {
      const s = json.stats;
      document.getElementById("aiEpisodes").innerText = s.episodes_trained || 0;
      document.getElementById("aiRewardAvg").innerText = (s.current_reward || 0).toFixed(3);
      document.getElementById("aiLossAvg").innerText = (s.average_loss || 0).toFixed(4);
      document.getElementById("aiWinRate").innerText = (s.ai_win_rate || 0) + "%";
    }

    // Load history for charts
    const hRes = await fetch(`${API_BASE}/api/ai/history?limit=30`);
    const hJson = await hRes.json();
    if (hJson.success && hJson.history) {
      const labels = hJson.history.map(h => `Ep ${h.episode}`);
      const rewards = hJson.history.map(h => h.reward);
      const losses = hJson.history.map(h => h.actor_loss + h.critic_loss);

      if (rewardChart) {
        rewardChart.data.labels = labels;
        rewardChart.data.datasets[0].data = rewards;
        rewardChart.update();
      }
      if (lossChart) {
        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = losses;
        lossChart.update();
      }
    }
  } catch (e) {}
}

function updateChartData(reward, loss) {
  if (!rewardChart) return;
  const label = `Ep ${rewardChart.data.labels.length + 1}`;
  rewardChart.data.labels.push(label);
  rewardChart.data.datasets[0].data.push(reward);
  if (rewardChart.data.labels.length > 30) {
    rewardChart.data.labels.shift();
    rewardChart.data.datasets[0].data.shift();
  }
  rewardChart.update();

  if (!lossChart) return;
  lossChart.data.labels.push(label);
  lossChart.data.datasets[0].data.push(loss);
  if (lossChart.data.labels.length > 30) {
    lossChart.data.labels.shift();
    lossChart.data.datasets[0].data.shift();
  }
  lossChart.update();
}

async function triggerTrainStep() {
  showToast("⚙️ Running PPO optimization step in background...");
  try {
    const res = await fetch(`${API_BASE}/api/ai/train-step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: 1 })
    });
    const json = await res.json();
    if (json.success) {
      showToast(`✅ Training step complete! Reward: ${json.result.reward}, Loss: ${json.result.loss}`);
      loadAIStats();
    }
  } catch (e) {
    showToast("Training request failed.");
  }
}

async function resetAIModel() {
  if (!confirm("Are you sure you want to reset the AI policy to untrained rookie weights?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/ai/reset`, { method: "POST" });
    const json = await res.json();
    if (json.success) {
      showToast("AI Model has been reset to rookie baseline.");
      loadAIStats();
    }
  } catch (e) {}
}

async function triggerPretrainBC() {
  const btn = document.getElementById("bcActionBtn");
  const topBtn = document.getElementById("bcPretrainBtn");
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Training...";
  }
  if (topBtn) topBtn.disabled = true;

  showToast("🧠 Running Offline Behavioral Cloning from expert demonstrations...");
  try {
    const res = await fetch(`${API_BASE}/api/ai/pretrain-bc`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes: 6, epochs: 3, batch_size: 64 })
    });
    const json = await res.json();
    if (json.success) {
      const r = json.result;
      showToast(`🎯 BC Pre-training complete! Accuracy: ${r.accuracy}%, Loss: ${r.final_loss}`);
      const badge = document.getElementById("bcAccuracyBadge");
      if (badge) badge.innerText = `Accuracy: ${r.accuracy}%`;
      loadAIStats();
      loadAIHistory();
    } else {
      showToast("Behavioral cloning failed.");
    }
  } catch (e) {
    showToast("Behavioral cloning request failed.");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "Warm-Start Policy";
    }
    if (topBtn) topBtn.disabled = false;
  }
}

// ===== FORMATION BUILDER CANVAS =====
function initFormationCanvas() {
  if (!canvas) return;
  window.addEventListener("resize", () => {
    resizeCanvas();
    drawPitch();
  });
  resizeCanvas();
  loadPresetFormation("4-3-3");

  // Mouse interaction
  canvas.addEventListener("mousedown", onCanvasMouseDown);
  canvas.addEventListener("mousemove", onCanvasMouseMove);
  canvas.addEventListener("mouseup", onCanvasMouseUp);
}

function resizeCanvas() {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
}

function loadPresetFormation(name) {
  currentFormationType = name;
  const preset = DEFAULT_PRESETS[name] || DEFAULT_PRESETS["4-4-2"];
  players = preset.map(p => ({ ...p }));
  document.getElementById("formationNameInput").value = `${name} Custom`;
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.classList.toggle("active", chip.innerText.includes(name));
  });
  drawPitch();
}

function drawPitch() {
  if (!ctx || !canvas) return;
  const w = canvas.width;
  const h = canvas.height;

  // Background
  ctx.fillStyle = "#15803d";
  ctx.fillRect(0, 0, w, h);

  // Alternating stripes
  ctx.fillStyle = "#166534";
  const stripes = 12;
  const stripeW = w / stripes;
  for (let i = 0; i < stripes; i += 2) {
    ctx.fillRect(i * stripeW, 0, stripeW, h);
  }

  // Pitch lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
  ctx.lineWidth = 3;

  // Outer border
  ctx.strokeRect(20, 20, w - 40, h - 40);

  // Halfway line
  ctx.beginPath();
  ctx.moveTo(w / 2, 20);
  ctx.lineTo(w / 2, h - 20);
  ctx.stroke();

  // Center circle
  ctx.beginPath();
  ctx.arc(w / 2, h / 2, h * 0.18, 0, Math.PI * 2);
  ctx.stroke();

  // Penalty Boxes
  const boxW = w * 0.16;
  const boxH = h * 0.52;
  ctx.strokeRect(20, (h - boxH) / 2, boxW, boxH);
  ctx.strokeRect(w - 20 - boxW, (h - boxH) / 2, boxW, boxH);

  // Draw tactical passing lanes (connecting nearby teammates)
  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.setLineDash([6, 6]);
  for (let i = 0; i < players.length; i++) {
    for (let j = i + 1; j < players.length; j++) {
      const p1 = players[i];
      const p2 = players[j];
      const dist = Math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h);
      if (dist < w * 0.28) {
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      }
    }
  }
  ctx.setLineDash([]);

  // Draw players
  const radius = Math.max(14, w * 0.018);
  players.forEach(p => {
    const px = p.x * w;
    const py = p.y * h;

    // Outer glow
    ctx.fillStyle = (p === draggingPlayer) ? "#38bdf8" : "#1e40af";
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Role text
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.round(radius * 0.75)}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(p.role, px, py);
  });
}

function onCanvasMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = (e.clientX - rect.left) / rect.width;
  const mouseY = (e.clientY - rect.top) / rect.height;

  for (const p of players) {
    const dist = Math.hypot(p.x - mouseX, (p.y - mouseY) * (rect.height / rect.width));
    if (dist < 0.04) {
      draggingPlayer = p;
      break;
    }
  }
}

function onCanvasMouseMove(e) {
  if (!draggingPlayer) return;
  const rect = canvas.getBoundingClientRect();
  let mouseX = (e.clientX - rect.left) / rect.width;
  let mouseY = (e.clientY - rect.top) / rect.height;

  mouseX = Math.max(0.04, Math.min(0.96, mouseX));
  mouseY = Math.max(0.06, Math.min(0.94, mouseY));

  draggingPlayer.x = mouseX;
  draggingPlayer.y = mouseY;
  drawPitch();
}

function onCanvasMouseUp() {
  draggingPlayer = null;
}

async function saveFormation() {
  const nameInput = document.getElementById("formationNameInput");
  const name = (nameInput ? nameInput.value.trim() : "") || "Custom Formation";
  const coords = players.map(p => [Math.round(p.x * 1000) / 1000, Math.round(p.y * 1000) / 1000]);

  try {
    const res = await fetch(`${API_BASE}/api/formations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name,
        formation_type: currentFormationType,
        coordinates: coords
      })
    });
    const json = await res.json();
    if (json.success) {
      showToast(`Tactical Formation '${name}' saved to Database!`);
      loadFormations();
    }
  } catch (e) {
    showToast("Error saving formation.");
  }
}

async function loadFormations() {
  try {
    const res = await fetch(`${API_BASE}/api/formations`);
    const json = await res.json();
    const list = document.getElementById("savedFormationsList");
    if (!list) return;
    list.innerHTML = "";

    if (json.formations) {
      json.formations.forEach(f => {
        const item = document.createElement("div");
        item.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:rgba(255,255,255,0.03);border:1px solid var(--glass-border);border-radius:8px;font-size:13px;margin-bottom:8px;";
        item.innerHTML = `
          <div>
            <strong>${f.name}</strong> <span style="color:var(--text-muted)">(${f.formation_type})</span>
          </div>
          <button class="btn-secondary" style="padding:4px 10px;font-size:11px;" onclick="applySavedFormation(${f.id})">Load</button>
        `;
        list.appendChild(item);
      });
      window._allFormations = json.formations;
    }
  } catch (e) {}
}

function applySavedFormation(id) {
  if (!window._allFormations) return;
  const f = window._allFormations.find(item => item.id === id);
  if (!f) return;

  currentFormationType = f.formation_type;
  const roles = (DEFAULT_PRESETS[f.formation_type] || DEFAULT_PRESETS["4-4-2"]).map(p => p.role);
  players = f.coordinates.map((c, i) => ({
    role: roles[i] || "PL",
    x: c[0],
    y: c[1]
  }));
  document.getElementById("formationNameInput").value = f.name;
  drawPitch();
  showToast(`Loaded '${f.name}' onto pitch.`);
}

// ===== AUTH MODAL =====
function openAuthModal() {
  document.getElementById("authModal").classList.add("show");
}

function closeAuthModal() {
  document.getElementById("authModal").classList.remove("show");
}

async function handleLogin() {
  const u = document.getElementById("authUsername").value.trim();
  const p = document.getElementById("authPassword").value.trim();
  if (!u || !p) return showToast("Please fill in username and password.");

  try {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p })
    });
    const json = await res.json();
    if (json.success) {
      localStorage.setItem("token", json.access_token);
      localStorage.setItem("user", JSON.stringify(json.user));
      showToast(`Welcome back, ${json.user.username}!`);
      closeAuthModal();
      checkAuth();
    } else {
      showToast(json.detail || "Sign in failed.");
    }
  } catch (e) {
    showToast("Network error during sign in.");
  }
}

async function handleRegister() {
  const u = document.getElementById("authUsername").value.trim();
  const p = document.getElementById("authPassword").value.trim();
  const email = `${u.toLowerCase()}@stadium.ai`;
  if (!u || !p) return showToast("Please fill in username and password.");

  try {
    const res = await fetch(`${API_BASE}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, email: email, password: p, role: "Player / Coach" })
    });
    const json = await res.json();
    if (json.success) {
      showToast("Account created! Signing you in...");
      handleLogin();
    } else {
      showToast(json.detail || "Registration failed.");
    }
  } catch (e) {
    showToast("Network error during registration.");
  }
}

function checkAuth() {
  const userStr = localStorage.getItem("user");
  const authBtn = document.getElementById("navAuthBtn");
  if (userStr && authBtn) {
    const user = JSON.parse(userStr);
    authBtn.innerText = `👤 ${user.username}`;
    authBtn.onclick = () => {
      if (confirm(`Signed in as ${user.username}. Log out?`)) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        checkAuth();
      }
    };
  } else if (authBtn) {
    authBtn.innerText = "Sign In / Register";
    authBtn.onclick = openAuthModal;
  }
}
