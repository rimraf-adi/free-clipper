import { useEffect, useState } from "react";
import { api } from "./api.js";
import { Icons } from "./components/Icons.jsx";
import Create from "./pages/Create.jsx";
import Library from "./pages/Library.jsx";
import Settings from "./pages/Settings.jsx";

const NAV = [
  { id: "create", label: "Create", icon: Icons.create },
  { id: "library", label: "Library", icon: Icons.library },
  { id: "settings", label: "Settings", icon: Icons.settings },
];

const TITLES = {
  create: { h: "Create clips", sub: "Turn any video into captioned vertical shorts" },
  library: { h: "Library", sub: "Every clip you've generated" },
  settings: { h: "Settings", sub: "Compute device & environment" },
};

const THEMES = [
  { id: "dark", label: "Dark" },
  { id: "light", label: "Light" },
];

// First-run (and every relaunch, briefly) splash while the whisper model
// loads. On a fresh install this can take a few minutes to download — this
// screen is the fix for that looking "stuck": real progress instead of a
// refused connection or a blank tab.
function ModelSplash({ status }) {
  const pct = status.progress != null ? Math.round(status.progress * 100) : null;
  const failed = status.status === "error";
  return (
    <div className="model-splash">
      <div className="model-splash-mark"><Icons.bolt /></div>
      <div className="model-splash-title">ClipForge</div>
      {failed ? (
        <>
          <div className="model-splash-msg" style={{ color: "var(--danger)" }}>Could not start the AI engine.</div>
          <div className="model-splash-detail">{status.message}</div>
          <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => window.location.reload()}>Retry</button>
        </>
      ) : (
        <>
          <div className="model-splash-msg">{status.message || "Starting up…"}</div>
          <div className="track model-splash-track">
            <div className={"fill" + (pct == null ? " indeterminate" : "")} style={pct == null ? {} : { width: pct + "%" }} />
          </div>
          {status.status === "downloading" && (
            <div className="model-splash-detail">One-time download — this machine won't need it again.</div>
          )}
        </>
      )}
    </div>
  );
}

// Segmented light/dark switch. Writes the choice to <html data-theme> and
// remembers it across reloads. "dark" is the CSS default, so we clear the attr.
function ThemeSwitch({ theme, setTheme, fixed }) {
  return (
    <div className={"theme-switch" + (fixed ? " theme-switch-fixed" : "")} role="group" aria-label="Theme">
      {THEMES.map((t) => (
        <button key={t.id} className={theme === t.id ? "active" : ""}
          onClick={() => setTheme(t.id)} title={`${t.label} theme`}>{t.label}</button>
      ))}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("create");
  const [step, setStep] = useState(1);
  const [device, setDevice] = useState(null);
  const [online, setOnline] = useState(null);
  const [navOpen, setNavOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("cf-theme") || "dark"; } catch { return "dark"; }
  });
  const [modelStatus, setModelStatus] = useState({ status: "idle", message: "", progress: null });

  // Poll until the whisper model is ready (or fails) — fast while it's
  // loading/downloading, then stops (no need to keep polling afterward).
  useEffect(() => {
    let alive = true;
    let timer;
    async function tick() {
      try {
        const s = await api.modelStatus();
        if (!alive) return;
        setModelStatus(s);
        if (s.status === "ready" || s.status === "error") return;
      } catch { /* server still coming up — keep trying */ }
      if (alive) timer = setTimeout(tick, 1000);
    }
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("cf-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => {
    let alive = true;
    const ping = () =>
      api.health()
        .then((d) => { if (alive) { setOnline(true); setDevice(d.device); } })
        .catch(() => { if (alive) setOnline(false); });
    ping();
    const t = setInterval(ping, 8000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  // The Create landing (step 1) is a clean, full-screen page: the sidebar + topbar
  // are HIDDEN (via the .is-landing class), not unmounted. <Create> stays mounted
  // across landing→editor so its source/url/upload state survives.
  const [showLogsDrawer, setShowLogsDrawer] = useState(false);
  const [globalLogs, setGlobalLogs] = useState([]);

  // Poll job logs when drawer is open or job is active
  useEffect(() => {
    let alive = true;
    const pollLogs = async () => {
      try {
        const hist = await api.history();
        if (!alive) return;
        // Collect history logs
        const logs = [];
        (hist || []).slice(0, 5).forEach((entry) => {
          logs.push({
            time: new Date(entry.created_at * 1000).toLocaleTimeString(),
            stage: "HISTORY",
            message: `Past job: ${entry.source} (${entry.clips?.length || 0} clips)`
          });
        });
        setGlobalLogs(logs);
      } catch {}
    };
    pollLogs();
  }, [showLogsDrawer]);

  const chromeless = page === "create" && step === 1;
  const t = TITLES[page];

  // Preserve step when navigating tabs! Do NOT reset setStep(1)
  const go = (id) => { setPage(id); setNavOpen(false); };

  if (modelStatus.status !== "ready") {
    return (
      <div className={"shell is-landing" + (navOpen ? " nav-open" : "")}>
        <ThemeSwitch theme={theme} setTheme={setTheme} fixed />
        <div className="content"><ModelSplash status={modelStatus} /></div>
      </div>
    );
  }

  return (
    <div className={"shell" + (chromeless ? " is-landing" : "") + (navOpen ? " nav-open" : "")}>
      {/* On the chromeless landing the topbar is hidden, so float the switcher. */}
      {chromeless && <ThemeSwitch theme={theme} setTheme={setTheme} fixed />}

      <div className="nav-scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />

      <aside className="sidebar">
        <div className="brand">
          <span className="logo"><Icons.bolt /></span>
          <span className="brand-text">ClipForge<span className="brand-by">by Haris AI</span></span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <button
              key={n.id}
              className={"nav-item" + (page === n.id ? " active" : "")}
              onClick={() => go(n.id)}
            >
              <n.icon /> {n.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          100% local pipeline<br />yt-dlp · faster-whisper · ffmpeg
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <button className="nav-toggle" onClick={() => setNavOpen((o) => !o)} aria-label="Menu">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
          <div className="topbar-titles">
            <h1>{t.h}</h1>
            <div className="sub">{t.sub}</div>
          </div>
          <div className="topbar-right">
            <button
              className="chip"
              onClick={() => setShowLogsDrawer(!showLogsDrawer)}
              style={{ cursor: "pointer", background: showLogsDrawer ? "var(--accent)" : "transparent", color: showLogsDrawer ? "#000" : "inherit" }}
              title="Toggle Pipeline Console Logs"
            >
              📜 Logs
            </button>
            <ThemeSwitch theme={theme} setTheme={setTheme} />
            <span className={"badge " + (online ? "ok" : online === false ? "warn" : "")}>
              <span className="dot" />
              {online == null ? "Connecting…" : online ? `Backend · ${(device || "ready").toUpperCase()}` : "Backend offline"}
            </span>
          </div>
        </header>

        {/* Global Console Logs Drawer */}
        {showLogsDrawer && (
          <div style={{
            position: "fixed", right: 20, top: 70, width: 420, maxWidth: "90vw", zIndex: 9999,
            background: "#18181B", border: "1px solid var(--line, #333)", borderRadius: 10,
            boxShadow: "0 10px 25px rgba(0,0,0,0.5)", padding: 14
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, borderBottom: "1px solid var(--line)", paddingBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 13, color: "#E4E4E7" }}>📜 Pipeline Console Logs</span>
              <button onClick={() => setShowLogsDrawer(false)} style={{ background: "none", border: "none", color: "#9CA3AF", cursor: "pointer", fontSize: 16 }}>✕</button>
            </div>
            <div style={{ maxHeight: 300, overflowY: "auto", fontFamily: "monospace", fontSize: 11, background: "#09090B", color: "#10B981", padding: 10, borderRadius: 6 }}>
              {globalLogs.length === 0 ? (
                <div style={{ color: "#71717A" }}>No pipeline activity recorded yet...</div>
              ) : (
                globalLogs.map((l, idx) => (
                  <div key={idx} style={{ marginBottom: 4, lineHeight: 1.4 }}>
                    <span style={{ color: "#71717A", marginRight: 6 }}>[{l.time}]</span>
                    <span style={{ color: "#38BDF8", marginRight: 6 }}>[{l.stage}]</span>
                    <span>{l.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        <div className="content">
          {/* Orbs are ALWAYS in the tree (hidden via CSS off the landing) so that
              <Create> never changes sibling-index. */}
          <div className="landing-orbs" aria-hidden="true">
            <span className="orb orb-1" />
            <span className="orb orb-2" />
            <span className="orb orb-3" />
            <span className="orb orb-4" />
            <span className="orb orb-5" />
          </div>
          {/* Preserve <Create> state across tab switches so wizard state is never lost */}
          <div style={{ display: page === "create" ? "block" : "none" }}>
            <Create key="create" step={step} setStep={setStep} />
          </div>
          {page === "library" && <Library />}
          {page === "settings" && <Settings />}
        </div>
      </div>
    </div>
  );
}
