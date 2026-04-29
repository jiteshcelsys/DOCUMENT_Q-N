import { useRef, useState } from "react";
import { uploadDocument } from "../api";

export default function Uploader({ onUploaded }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState("");
  const [uploads, setUploads] = useState([]);

  async function handleFile(file) {
    if (!file) return;
    setError("");
    setProgress(0);
    // Reset input so the same file can be selected again (triggers onChange)
    if (inputRef.current) inputRef.current.value = "";
    try {
      const result = await uploadDocument(file, setProgress);
      console.log(`Uploaded ${file.name}:`, result);
      setUploads((prev) => [result, ...prev]);
      onUploaded(result);
    } catch (e) {
      setError(e.message);
      setTimeout(() => setError(""), 5000);
    } finally {
      setProgress(null);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }

  return (
    <div className="uploader">
      <h2>Upload Document</h2>
      <div
        className={`drop-zone ${dragging ? "drag-over" : ""}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <span>Drop a PDF, DOCX, or TXT file here<br />or click to browse</span>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {progress !== null && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {uploads.length > 0 && (
        <ul className="upload-list">
          {uploads.map((u, i) => (
            <li key={i}>
              <span className="upload-name">{u.filename}</span>
              <span className="upload-meta">{u.chunks_indexed} chunks</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
