console.log("[STEP 1] content.js loaded");

const showBanner = () => {
  console.log("[STEP 2] Showing banner");
  const banner = document.createElement("div");
  banner.innerText = "⚠️ Utilisez notre extension pour scanner ce mail";
  banner.style.position = "fixed";
  banner.style.top = "0";
  banner.style.left = "0";
  banner.style.width = "100%";
  banner.style.padding = "15px";
  banner.style.backgroundColor = "#ffcc00";
  banner.style.color = "black";
  banner.style.fontSize = "18px";
  banner.style.textAlign = "center";
  banner.style.zIndex = "9999";
  document.body.appendChild(banner);
};

window.addEventListener("load", () => {
  console.log("[STEP 3] Gmail page fully loaded");
  showBanner();
});
