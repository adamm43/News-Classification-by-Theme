// frontend/script.js

const API = "http://127.0.0.1:5000/predict";
const chatBox = document.getElementById("chatBox");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const loading = document.getElementById("loading");
const themeToggle = document.getElementById("themeToggle");
const clearBtn = document.getElementById("clearHistory");
const exportBtn = document.getElementById("exportHistory");

let historyKey = "news_classifier_history_v1";
let history = JSON.parse(localStorage.getItem(historyKey) || "[]");

// render saved history
function renderHistory() {
  chatBox.innerHTML = "";
  history.forEach(item => {
    appendMessage("user", item.text);
    appendMessage("bot", `Thème: ${item.prediction} • Confiance: ${item.confidence}%`);
  });
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendMessage(who, text) {
  const tpl = document.getElementById(who === "user" ? "tplUser" : "tplBot");
  const node = tpl.content.cloneNode(true);
  node.querySelector(".bubble").innerHTML = text;
  chatBox.appendChild(node);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// auto-resize
textInput.addEventListener("input", () => {
  textInput.style.height = "auto";
  textInput.style.height = textInput.scrollHeight + "px";
});

// send
async function sendText() {
  const txt = textInput.value.trim();
  if (!txt) return;

  appendMessage("user", txt);
  textInput.value = "";
  textInput.style.height = "48px";
  loading.classList.remove("hidden");

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: txt})
    });

    if (!res.ok) {
      appendMessage("bot", "❌ Erreur du serveur. Réessayez plus tard.");
      loading.classList.add("hidden");
      return;
    }

    const data = await res.json();
    const msg = `Thème: <b>${data.prediction}</b><br>Confiance: <b>${data.confidence}%</b>`;
    appendMessage("bot", msg);
    // save to history
    history.push({text: txt, prediction: data.prediction, confidence: data.confidence});
    localStorage.setItem(historyKey, JSON.stringify(history));
  } catch (err) {
    console.error(err);
    appendMessage("bot", "❌ Impossible de joindre le serveur.");
  } finally {
    loading.classList.add("hidden");
  }
}

// events
sendBtn.addEventListener("click", sendText);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    sendText();
  }
});

// theme
themeToggle.addEventListener("change", () => {
  document.body.classList.toggle("dark");
});

// history controls
clearBtn.addEventListener("click", () => {
  if (!confirm("Effacer tout l'historique ?")) return;
  history = [];
  localStorage.removeItem(historyKey);
  renderHistory();
});

exportBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(history, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "history.json"; a.click();
  URL.revokeObjectURL(url);
});

renderHistory();
