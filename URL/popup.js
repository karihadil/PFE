console.log("[STEP 1] popup.js loaded");

let lastResults = [];

const scanBtn = document.getElementById("scanBtn");
const copyBtn = document.getElementById("copyBtn");
const logBox = document.getElementById("log");
const summaryBox = document.getElementById("summary");
const loader = document.getElementById("loader");

const cardTrusted = document.getElementById("card-trusted");
const cardSuspicious = document.getElementById("card-suspicious");
const cardMalicious = document.getElementById("card-malicious");

scanBtn.addEventListener("click", () => {
  loader.classList.remove("hidden");
  summaryBox.innerText = "";
  logBox.innerHTML = "";
  hideCards();

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;

    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func: scanEmailLinksOnly,
      world: "MAIN"
    }, async (results) => {
      const urls = (results?.[0]?.result) || [];
      if (!urls.length) {
        loader.classList.add("hidden");
        summaryBox.innerHTML = "<span style='color: #8b949e;'>No URLs found in the email.</span>";
        return;
      }

      try {
        const response = await fetch("http://127.0.0.1:10000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls })
        });

        const data = await response.json();
        lastResults = data.results;
        loader.classList.add("hidden");

        let red = 0, orange = 0, green = 0;
        const phishingToBlock = [];

        lastResults.forEach(r => {
          const label = r.final_label.toLowerCase();
          let classType = "safe";

          if (label.includes("high confidence") || label.includes("flagged")) {
            classType = "phishing";
            phishingToBlock.push(r.url);
            red++;
          } else if (label.includes("suspicious")) {
            classType = "suspicious";
            orange++;
          } else {
            green++;
          }

          const div = document.createElement("div");
          div.className = `link-item fade-in ${classType}`;
          div.innerHTML = `
            <div><strong>${r.url}</strong></div>
            <div>🌲 XGBoost: ${
              r.xgb.label === "Legitimate"
                ? "Legitimate"
                : `${r.xgb.label} (${Math.round(r.xgb.confidence * 100)}%)`
            }</div>
            <div>🔐 Google Safe Browsing: ${r.google ? '⚠️ Flagged' : '✅ Clean'}</div>
            <div>🔎 Verdict: <strong>${r.final_label}</strong></div>
          `;
          logBox.appendChild(div);
        });

        if (red > 0) cardMalicious.classList.remove("hidden");
        else if (orange > 0) cardSuspicious.classList.remove("hidden");
        else cardTrusted.classList.remove("hidden");

        summaryBox.innerHTML = `✅ Legitimate: <strong>${green}</strong><br>⚠️ Suspicious: <strong>${orange}</strong><br>🚨 Phishing: <strong>${red}</strong>`;

        // ✅ Inject blocking logic directly
        if (phishingToBlock.length > 0) {
          chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: blockPhishingLinks,
            args: [phishingToBlock],
            world: "MAIN"
          });
        }

      } catch (err) {
        console.error("[ERROR] Backend fetch failed:", err);
        loader.classList.add("hidden");
        summaryBox.innerText = "Error connecting to backend.";
      }
    });
  });
});

copyBtn.addEventListener("click", () => {
  const phishingLinks = lastResults.filter(r =>
    r.final_label.toLowerCase().includes("high confidence") ||
    r.final_label.toLowerCase().includes("flagged")
  ).map(r => r.url);

  if (!phishingLinks.length) return alert("⚠️ No phishing links to copy.");

  navigator.clipboard.writeText(phishingLinks.join("\n")).then(() => {
    alert("✅ Phishing links copied to clipboard.");
  }).catch(err => {
    console.error("[ERROR] Clipboard copy failed:", err);
    alert("❌ Could not copy links.");
  });
});

function scanEmailLinksOnly() {
  return new Promise((resolve) => {
    setTimeout(() => {
      const emailContainer = document.querySelector("div.a3s") || document.querySelector("div.ii.gt");
      if (!emailContainer) return resolve([]);
      const anchors = Array.from(emailContainer.querySelectorAll("a"));
      const urls = anchors.map(a => a.href).filter(href => href.startsWith("http"));
      resolve([...new Set(urls)]);
    }, 1200);
  });
}

function hideCards() {
  cardTrusted.classList.add("hidden");
  cardSuspicious.classList.add("hidden");
  cardMalicious.classList.add("hidden");
}

// ✅ INJECTED BLOCKING FUNCTION
function blockPhishingLinks(urls) {
  const emailBody = document.querySelector("div.a3s");
  if (!emailBody) return;

  const links = emailBody.querySelectorAll("a");
  links.forEach(link => {
    urls.forEach(phishUrl => {
      if (link.href.includes(phishUrl)) {
        link.removeAttribute("href");
        link.style.pointerEvents = "none";
        link.style.color = "#fff";
        link.style.backgroundColor = "#a00";
        link.style.textDecoration = "line-through";
        link.style.padding = "4px 6px";
        link.style.border = "2px solid red";
        link.style.borderRadius = "5px";
        link.style.fontWeight = "bold";
        link.title = "⚠️ BLOCKED by PhishGuard";

        if (!link.nextSibling || !link.nextSibling.textContent.includes("PHISHING BLOCKED")) {
          const warning = document.createElement("span");
          warning.innerText = " ⚠️ [PHISHING BLOCKED]";
          warning.style.color = "#ff4e4e";
          warning.style.fontWeight = "bold";
          warning.style.marginLeft = "6px";
          link.after(warning);
        }
      }
    });
  });
}

document.getElementById("about-btn").addEventListener("click", () => {
  document.getElementById("aboutModal").classList.toggle("active");
});
document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("aboutModal").classList.remove("active");
});
document.getElementById("close-btn").addEventListener("click", () => {
  window.close();
});
