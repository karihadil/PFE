console.log("[STEP 4] popup.js loaded");

let lastResults = [];

document.addEventListener("DOMContentLoaded", () => {
  const scanBtn = document.getElementById("scanBtn");
  const copyBtn = document.getElementById("copyBtn");
  const logBox = document.getElementById("log");
  const summaryBox = document.getElementById("summary");

  if (!scanBtn || !copyBtn) {
    console.log("[ERROR] Buttons not found in DOM.");
    return;
  }

  scanBtn.addEventListener("click", () => {
    console.log("[STEP 5] Scan button clicked");

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) {
        console.log("[ERROR] No active tab found.");
        return;
      }

      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        func: scanEmailAndReturnResults,
        world: "MAIN"
      }, (injectionResults) => {
        if (chrome.runtime.lastError) {
          logBox.innerText = "Erreur lors de l'injection du script.";
          return;
        }

        const results = injectionResults[0].result;
        console.log("[STEP 6] Results received in popup:", results);
        lastResults = results;

        if (results.length === 0) {
          summaryBox.innerText = "Résumé : Aucun lien détecté.";
          logBox.innerText = "Aucun lien détecté.";
          return;
        }

        const phishingLinks = results.filter(r => r.prediction === 'Phishing');
        const safeLinks = results.filter(r => r.prediction === 'Legitimate');

        summaryBox.innerText = `Résumé : ✅ ${safeLinks.length} légitimes, ⚠️ ${phishingLinks.length} suspects`;

        logBox.innerHTML = "";
        results.forEach(r => {
          const div = document.createElement("div");
          div.className = `link-item ${r.prediction === 'Phishing' ? 'phishing' : 'safe'}`;
          div.innerText = `${r.url} → ${r.prediction}`;
          logBox.appendChild(div);
        });
      });
    });
  });

  copyBtn.addEventListener("click", () => {
    const phishingLinks = lastResults.filter(r => r.prediction === 'Phishing').map(r => r.url);
    if (phishingLinks.length === 0) {
      alert("Aucun lien suspect à copier.");
      return;
    }

    const textToCopy = phishingLinks.join("\n");
    navigator.clipboard.writeText(textToCopy).then(() => {
      alert("✅ Liens suspects copiés dans le presse-papiers !");
    }).catch(err => {
      console.error("[ERROR] Échec de la copie :", err);
      alert("❌ Erreur lors de la copie.");
    });
  });
});

function scanEmailAndReturnResults() {
  return new Promise((resolve) => {
    console.log("[STEP 7] Waiting 2.5s for Gmail content...");
    setTimeout(() => {
      const emailContainer =
        document.querySelector("div.a3s") || document.querySelector("div.ii.gt");

      if (!emailContainer) {
        console.log("[STEP 8] No email body container found.");
        resolve([]);
        return;
      }

      const bodyLinks = Array.from(emailContainer.querySelectorAll("a"));
      const htmlUrls = bodyLinks.map(link => link.href).filter(href => href.startsWith("http"));

      const text = emailContainer.innerText;
      const regexUrls = text.match(/https?:\/\/[^"]+/g) || [];

      const allUrls = [...new Set([...htmlUrls, ...regexUrls])];

      if (allUrls.length === 0) {
        console.log("[STEP 9] No URLs found in email body.");
        resolve([]);
        return;
      }

      console.log("[STEP 10] URLs to send to backend:", allUrls);

      fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: allUrls })
      })
        .then(res => res.json())
        .then(data => {
          console.log("[STEP 11] Backend responded:", data.results);
          resolve(data.results);
        })
        .catch(err => {
          console.error("[ERROR] Backend fetch failed:", err);
          resolve([]);
        });
    }, 2500);
  });
}
