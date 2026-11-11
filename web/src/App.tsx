import { useEffect, useState } from "react";

type Health = { status: string; core_version: string; intended_use: string };

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 720, margin: "2rem auto", padding: "0 1rem" }}>
      <div style={{ background: "#fff3cd", border: "1px solid #e0c36a", padding: "0.5rem 1rem", borderRadius: 6 }}>
        Planning aid, not for clinical use.
      </div>
      <h1>Bronchoscopy Navigation Planner</h1>
      <p>Scaffold only. The MPR viewer, airway tree and virtual bronchoscopy views come in later phases.</p>
      <h2>Backend</h2>
      {error && <p style={{ color: "#b00020" }}>API unreachable: {error}</p>}
      {health && (
        <p>
          API {health.status}, core v{health.core_version}
        </p>
      )}
      {!health && !error && <p>Checking API…</p>}
    </main>
  );
}
