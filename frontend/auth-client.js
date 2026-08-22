// ==========================================================================
// AI GAMING FOOTBALL SHARED AUTH & BILLING CLIENT (v5.0)
// ==========================================================================

const LOCAL_HOSTS = ["127.0.0.1", "localhost"];
const LOCAL_PORTS = [8005, 8000, 8080];
const REMOTE_API_BASE = "https://ai-gaming-backend.onrender.com";
let activeApiBaseUrl = "http://127.0.0.1:8005";

let currentUser = null;
let authToken = null;

// Sound FX chimes using Web Audio API
const SoundFX = {
  ctx: null,
  init() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
  },
  playClick() {
    this.init();
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.type = "sine";
    osc.frequency.setValueAtTime(800, this.ctx.currentTime);
    gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.05);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.05);
  },
  playSuccess() {
    this.init();
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const notes = [523.25, 659.25, 783.99];
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.type = "triangle";
      osc.frequency.setValueAtTime(freq, t + idx * 0.1);
      gain.gain.setValueAtTime(0.08, t + idx * 0.1);
      gain.gain.exponentialRampToValueAtTime(0.001, t + idx * 0.1 + 0.25);
      osc.start(t + idx * 0.1);
      osc.stop(t + idx * 0.1 + 0.25);
    });
  }
};

async function checkBackendConnection() {
  for (const host of LOCAL_HOSTS) {
    for (const port of LOCAL_PORTS) {
      try {
        const url = `http://${host}:${port}`;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600);
        const res = await fetch(`${url}/`, { method: "GET", signal: controller.signal });
        clearTimeout(timeoutId);
        if (res.ok) {
          activeApiBaseUrl = url;
          return;
        }
      } catch (err) {}
    }
  }
  activeApiBaseUrl = REMOTE_API_BASE;
}

function getAuthHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  return headers;
}

function showToast(msg, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerText = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.classList.add("removing"), 3000);
  setTimeout(() => toast.remove(), 3400);
}

// UI Controls
function showAuthModal() {
  document.getElementById("authModal").classList.add("active");
  switchAuthTab("login");
}
window.closeAuthModal = function() {
  document.getElementById("authModal").classList.remove("active");
};

window.switchAuthTab = function(tab) {
  document.getElementById("authTabLogin").classList.toggle("active", tab === "login");
  document.getElementById("authTabRegister").classList.toggle("active", tab === "register");
  document.getElementById("authPaneLogin").classList.toggle("active", tab === "login");
  document.getElementById("authPaneRegister").classList.toggle("active", tab === "register");
  document.getElementById("authMsg").style.display = "none";
};

// Form submits
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value.trim().toLowerCase();
  const password = document.getElementById("loginPassword").value;
  const msgEl = document.getElementById("authMsg");

  try {
    const res = await fetch(`${activeApiBaseUrl}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (data.success) {
      authToken = data.access_token;
      currentUser = data.user;
      localStorage.setItem("game_token", authToken);
      localStorage.setItem("game_user", JSON.stringify(currentUser));
      closeAuthModal();
      updateUserUI();
      showToast("Authentication successful!", "success");
      SoundFX.playSuccess();
      loadMe();
    } else {
      msgEl.innerText = data.message || data.detail;
      msgEl.className = "auth-msg error";
    }
  } catch (err) {
    msgEl.innerText = "Server connection failed.";
    msgEl.className = "auth-msg error";
  }
});

document.getElementById("registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("regUsername").value.trim().toLowerCase();
  const email = document.getElementById("regEmail").value.trim();
  const fullname = document.getElementById("regFullname").value.trim();
  const role = document.getElementById("regRole").value;
  const password = document.getElementById("regPassword").value;
  const msgEl = document.getElementById("authMsg");

  try {
    const res = await fetch(`${activeApiBaseUrl}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, full_name: fullname, role, password })
    });
    const data = await res.json();
    if (data.success) {
      msgEl.innerText = data.message;
      msgEl.className = "auth-msg success";
      setTimeout(() => switchAuthTab("login"), 1200);
    } else {
      msgEl.innerText = data.message || data.detail;
      msgEl.className = "auth-msg error";
    }
  } catch (err) {
    msgEl.innerText = "Connection error.";
    msgEl.className = "auth-msg error";
  }
});

async function loadMe() {
  if (!authToken) return;
  try {
    const res = await fetch(`${activeApiBaseUrl}/auth/me`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.success) {
      currentUser = data.user;
      localStorage.setItem("game_user", JSON.stringify(currentUser));
      updateUserUI();
    }
  } catch (e) {}
}

function updateUserUI() {
  const loginBtn = document.getElementById("loginBtn");
  const userChipBtn = document.getElementById("userChipBtn");
  
  if (currentUser) {
    if (loginBtn) loginBtn.style.display = "none";
    if (userChipBtn) userChipBtn.style.display = "flex";
    
    const initials = currentUser.full_name ? currentUser.full_name.substring(0, 2).toUpperCase() : currentUser.username.substring(0, 2).toUpperCase();
    document.getElementById("sidebarAvatar").innerText = initials;
    document.getElementById("sidebarUsername").innerText = currentUser.full_name || currentUser.username;
    
    const plan = currentUser.plan || "free";
    const badge = document.getElementById("sidebarPlanBadge");
    badge.innerText = plan;
    badge.className = `plan-badge ${plan}`;
    
    // Dispatch session event to dashboard
    window.dispatchEvent(new CustomEvent("auth-change", { detail: { loggedIn: true, user: currentUser } }));
  } else {
    if (loginBtn) loginBtn.style.display = "flex";
    if (userChipBtn) userChipBtn.style.display = "none";
    window.dispatchEvent(new CustomEvent("auth-change", { detail: { loggedIn: false } }));
  }
}

window.doLogout = function() {
  authToken = null;
  currentUser = null;
  localStorage.removeItem("game_token");
  localStorage.removeItem("game_user");
  updateUserUI();
  closeProfileModal();
  showToast("Logged out successfully.", "info");
  SoundFX.playClick();
};

window.openProfileModal = function() {
  if (!currentUser) return;
  document.getElementById("profileModal").classList.add("active");
  switchProfileTab("info");
  loadProfileModalData();
};
window.closeProfileModal = function() {
  document.getElementById("profileModal").classList.remove("active");
};

window.switchProfileTab = function(tab) {
  document.querySelectorAll(".profile-tab").forEach(t => t.classList.toggle("active", t.dataset.ptab === tab));
  document.querySelectorAll(".profile-pane").forEach(p => p.classList.toggle("active", p.id === "ptab-" + tab));

  if (tab === "usage") loadUsageTab();
  if (tab === "activity") loadActivityTab();
};

function loadProfileModalData() {
  document.getElementById("pfFullname").value = currentUser.full_name || "";
  document.getElementById("pfEmail").value = currentUser.email || "";
  
  const initials = currentUser.full_name ? currentUser.full_name.substring(0, 2).toUpperCase() : currentUser.username.substring(0, 2).toUpperCase();
  document.getElementById("profileAvatarLarge").innerText = initials;
  document.getElementById("profileHeaderName").innerText = currentUser.full_name || currentUser.username;
  document.getElementById("profileRoleTag").innerText = currentUser.role || "learner";
  
  const pb = document.getElementById("profilePlanBadge");
  pb.innerText = currentUser.plan || "free";
  pb.className = `plan-badge ${currentUser.plan || "free"}`;
}

window.saveProfileInfo = async function() {
  const fullName = document.getElementById("pfFullname").value.trim();
  const email = document.getElementById("pfEmail").value.trim();
  const infoMsg = document.getElementById("profileInfoMsg");

  try {
    const res = await fetch(`${activeApiBaseUrl}/auth/profile/update`, {
      method: "PATCH",
      headers: getAuthHeaders(),
      body: JSON.stringify({ full_name: fullName, email })
    });
    const data = await res.json();
    if (data.success) {
      currentUser = { ...currentUser, ...data.user };
      localStorage.setItem("game_user", JSON.stringify(currentUser));
      updateUserUI();
      loadProfileModalData();
      infoMsg.innerText = "Account info updated successfully.";
      infoMsg.className = "auth-msg success";
    }
  } catch (err) {
    infoMsg.innerText = "Failed to update profile.";
    infoMsg.className = "auth-msg error";
  }
};

function loadUsageTab() {
  const savedCount = Object.keys(currentUser.saved_formations || {}).length;
  let maxFormations = 3;
  if (currentUser.plan === "pro") maxFormations = 50;
  if (currentUser.plan === "developer") maxFormations = 9999;

  document.getElementById("pfFormationsSaved").innerText = savedCount;
  document.getElementById("pfMaxLabel").innerText = maxFormations === 9999 ? `${savedCount} / ∞` : `${savedCount} / ${maxFormations}`;
  const progress = maxFormations === 9999 ? 0 : (savedCount / maxFormations) * 100;
  document.getElementById("pfFormationsBar").style.width = `${progress}%`;

  // Toggle Stripe portal button visibility
  const portalBox = document.getElementById("stripePortalSection");
  if (currentUser.plan !== "free") {
    portalBox.style.display = "block";
  } else {
    portalBox.style.display = "none";
  }
}

async function loadActivityTab() {
  const container = document.getElementById("profileActivityContent");
  try {
    const res = await fetch(`${activeApiBaseUrl}/auth/activity`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.success && data.activity.length > 0) {
      container.innerHTML = `
        <table style="width:100%;text-align:left;border-collapse:collapse">
          <tr style="color:var(--text-dim)"><th style="padding:6px">Event Log</th><th style="padding:6px">Timestamp</th></tr>
          ${data.activity.map(a => `
            <tr style="border-top:1px solid var(--glass-border)">
              <td style="padding:6px"><code>${a.endpoint}</code></td>
              <td style="padding:6px">${new Date(a.timestamp).toLocaleTimeString()}</td>
            </tr>
          `).join("")}
        </table>`;
    } else {
      container.innerText = "No recent training or tactical activity recorded.";
    }
  } catch (err) {
    container.innerText = "Activity logs unavailable.";
  }
}

window.openSubscriptionModal = function() {
  document.getElementById("subscriptionModal").classList.add("active");
  renderPlansGrid();
};
window.closeSubscriptionModal = function() {
  document.getElementById("subscriptionModal").classList.remove("active");
};

function renderPlansGrid() {
  const grid = document.getElementById("plansGrid");
  const userPlan = currentUser ? currentUser.plan : "free";

  const plans = [
    {
      id: "free",
      name: "Tactical Sandbox",
      price: "$0",
      desc: "For basic formation editing and simulation observation",
      features: ["Max 3 custom formations", "Standard formations editor", "No model checkpoints download"]
    },
    {
      id: "pro",
      name: "Pro Trainer",
      price: "$9",
      desc: "For advanced analysts saving dynamic tactics files",
      features: ["Max 50 custom formations", "3D dynamic graphics overlays", "Download GAT-PPO policy weights (.pt)"]
    },
    {
      id: "developer",
      name: "Developer/Manager",
      price: "$49",
      desc: "For engineers training custom self-play RL models",
      features: ["Unlimited saved formations", "Direct API key keygen access", "GAT matrix spatial analyzer JSON"]
    }
  ];

  grid.innerHTML = plans.map(p => {
    const isActive = p.id === userPlan;
    let btnHtml = "";
    if (isActive) {
      btnHtml = `<button class="pricing-btn current" disabled>Active Plan</button>`;
    } else {
      btnHtml = `<button class="pricing-btn upgrade" onclick="triggerCheckout('${p.id}')">Subscribe to ${p.name}</button>`;
    }
    return `
      <div class="pricing-card${isActive ? ' active' : ''}">
        <h3 class="pricing-title">${p.name}</h3>
        <div class="pricing-price">${p.price}<span>/month</span></div>
        <p class="pricing-desc">${p.desc}</p>
        <ul class="pricing-features">
          ${p.features.map(f => `<li>${f}</li>`).join("")}
        </ul>
        ${btnHtml}
      </div>`;
  }).join("");
}

window.triggerCheckout = async function(plan) {
  if (!currentUser) {
    closeSubscriptionModal();
    showAuthModal();
    showToast("Please sign in to upgrade subscription.", "warning");
    return;
  }
  try {
    const res = await fetch(`${activeApiBaseUrl}/payment/checkout`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ 
        plan: plan,
        success_url: window.location.href.split('?')[0] + "?payment=success&plan=" + plan 
      })
    });
    const data = await res.json();
    if (data.success && data.checkout_url) {
      showToast("Redirecting to Stripe Billing Sandbox...", "info");
      SoundFX.playClick();
      setTimeout(() => {
        window.location.href = data.checkout_url;
      }, 800);
    }
  } catch (err) {
    showToast("Stripe sandbox currently offline.", "error");
  }
}

window.openStripePortal = async function() {
  try {
    showToast("Redirecting to Stripe Billing Portal...", "info");
    SoundFX.playClick();
    const res = await fetch(`${activeApiBaseUrl}/payment/portal`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ return_url: window.location.href })
    });
    const data = await res.json();
    if (data.success && data.portal_url) {
      setTimeout(() => {
        window.location.href = data.portal_url;
      }, 800);
    } else {
      showToast(data.detail || "Failed to open billing portal.", "error");
    }
  } catch (err) {
    showToast("Billing portal currently offline.", "error");
  }
};

async function handlePaymentRedirect() {
  const params = new URLSearchParams(window.location.search);
  const payment = params.get("payment");
  const plan = params.get("plan");
  const savedToken = localStorage.getItem("game_token");
  const savedUser = localStorage.getItem("game_user");

  if (payment === "success" && plan && savedToken && savedUser) {
    authToken = savedToken;
    currentUser = JSON.parse(savedUser);
    
    try {
      const resp = await fetch(`${activeApiBaseUrl}/payment/webhook`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "checkout.session.completed",
          data: {
            object: {
              client_reference_id: currentUser.username,
              metadata: { plan }
            }
          }
        })
      });
      if (resp.ok) {
        window.history.replaceState({}, document.title, window.location.pathname);
        showToast(`Upgrade Successful! Subscribed to ${plan}.`, "success");
        SoundFX.playSuccess();
        await loadMe();
      }
    } catch (err) {}
  }
}

// Attach page triggers
document.getElementById("loginBtn").addEventListener("click", showAuthModal);
document.getElementById("userChipBtn").addEventListener("click", openProfileModal);

async function init() {
  await checkBackendConnection();
  const savedToken = localStorage.getItem("game_token");
  const savedUser = localStorage.getItem("game_user");
  if (savedToken && savedUser) {
    authToken = savedToken;
    currentUser = JSON.parse(savedUser);
    updateUserUI();
    loadMe();
  }
  await handlePaymentRedirect();
}
init();
