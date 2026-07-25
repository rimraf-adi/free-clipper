import { Icons } from "../../components/Icons.jsx";
import ClipPhone from "../../components/ClipPhone.jsx";

const STEPS = [["downloading", "Download"], ["transcribing", "Transcribe"], ["selecting", "Analyze"], ["rendering", "Render"]];

// Screen 5 — Clip Rendering & Review. Auto-renders every clip on entry (no
// export yet); each rendered clip shows exactly two actions — Download and
// Reframe — so reviewers can fix a bad crop before moving on to Export.
export default function Screen5Review({ busy, snap, clips, error, onCancel, onOpenReframe, onBack, onNext }) {
  const curStep = STEPS.findIndex(([k]) => k === snap?.stage);
  const done = snap?.status === "done";
  const pct = Math.round((snap?.progress || 0) * 100);

  return (
    <div className="wizard-screen">
      <div className="card">
        <div className="card-h"><h2>Rendering your clips</h2><span className="hint">{clips.length} ready</span></div>

        {busy && (
          <button className="btn btn-cancel" style={{ marginBottom: 14 }} onClick={onCancel}>Cancel</button>
        )}
        {error && <div className="error">{error}</div>}

        {snap && (
          <>
            <div className="cc-progress" style={{ padding: 0 }}>
              <div className="steps">
                {STEPS.map(([k, lbl], i) => (
                  <div key={k} className={"step" + (done || i < curStep ? " done" : i === curStep ? " active" : "")}>
                    <div className="ring">{done || i < curStep ? "✓" : i + 1}</div><div className="lbl">{lbl}</div>
                  </div>
                ))}
              </div>
              <div className="bar-row"><span className="msg">{busy && <span className="spinner" />}{snap.message}</span><span className="pct">{pct}%</span></div>
              <div className="track"><div className="fill" style={{ width: pct + "%" }} /></div>
            </div>

            <details style={{ marginTop: 16, background: "#18181B", borderRadius: 8, padding: "10px 14px", border: "1px solid var(--line, #27272A)" }}>
              <summary style={{ cursor: "pointer", fontWeight: 600, color: "#E4E4E7", fontSize: 13, userSelect: "none" }}>
                📜 Live Pipeline Console Logs ({snap.logs?.length || 0} entries)
              </summary>
              <div style={{
                marginTop: 10, maxHeight: 220, overflowY: "auto", fontFamily: "monospace",
                fontSize: 11, background: "#09090B", color: "#10B981", padding: 10, borderRadius: 6
              }}>
                {(!snap.logs || snap.logs.length === 0) ? (
                  <div style={{ color: "#71717A" }}>No log entries recorded yet...</div>
                ) : (
                  snap.logs.map((l, idx) => (
                    <div key={idx} style={{ marginBottom: 4, lineHeight: 1.4 }}>
                      <span style={{ color: "#71717A", marginRight: 8 }}>[{l.time}]</span>
                      <span style={{ color: "#38BDF8", marginRight: 8 }}>[{(l.stage || "LOG").toUpperCase()}]</span>
                      <span style={{ color: l.stage === "error" ? "#F87171" : "#10B981" }}>{l.message}</span>
                    </div>
                  ))
                )}
              </div>
            </details>
          </>
        )}
      </div>

      {clips.length > 0 && (
        <div className="card">
          <div className="clips">
            {clips.map((c) => (
              <div className="clip" key={c.index}>
                <ClipPhone src={c.url} filename={c.filename} />
                <div className="meta">
                  <h3>{c.title}</h3>
                  <div className="sub">{(c.end - c.start).toFixed(1)}s · {c.start.toFixed(1)}–{c.end.toFixed(1)}s</div>
                  <div className="acts">
                    <a href={c.url} download={c.filename || ""}>Download</a>
                    <button onClick={() => onOpenReframe(c)}><Icons.crop /> Reframe</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="wizard-nav">
        <button className="btn btn-ghost" onClick={onBack} disabled={busy}>← Back to effects</button>
        <button className="btn btn-primary" disabled={!done || !clips.length} onClick={onNext}>Continue to export →</button>
      </div>
    </div>
  );
}
