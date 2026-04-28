const BASE = "/api";

export async function uploadDocument(file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(JSON.parse(xhr.responseText)?.detail ?? "Upload failed"));
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(form);
  });
}

export async function askQuestion(question, k = 4) {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

export async function clearIndex() {
  const res = await fetch(`${BASE}/index`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear index");
  return res.json();
}
