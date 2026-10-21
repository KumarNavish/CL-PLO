import { initApp } from "./ui/render.js?v=20260216r4";

function bootErrorMessage(error) {
  if (!error) {
    return "Unknown startup error.";
  }
  if (typeof error === "string") {
    return error;
  }
  if (error && typeof error.message === "string" && error.message.trim()) {
    return error.message;
  }
  return "Unknown startup error.";
}

function showBootError(error) {
  const host = document.getElementById("status-line");
  if (!host) {
    return;
  }
  host.textContent = `Rendering error: ${bootErrorMessage(error)}`;
  host.classList.add("error");
}

window.addEventListener("error", (event) => {
  const nextError = event && (event.error || event.message);
  showBootError(nextError);
});

window.addEventListener("unhandledrejection", (event) => {
  const nextError = event ? event.reason : null;
  showBootError(nextError);
});

try {
  initApp();
} catch (error) {
  showBootError(error);
  throw error;
}
