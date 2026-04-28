import { useState } from "react";
import Chat from "./components/Chat";
import Uploader from "./components/Uploader";
import "./App.css";

export default function App() {
  const [hasDocuments, setHasDocuments] = useState(false);

  return (
    <div className="layout">
      <header>
        <h1>Document Q&amp;A</h1>
        <span className="subtitle">Powered by Groq · LangChain · FAISS</span>
      </header>
      <main>
        <Uploader onUploaded={() => setHasDocuments(true)} />
        <Chat hasDocuments={hasDocuments} />
      </main>
    </div>
  );
}
