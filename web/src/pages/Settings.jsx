import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Settings() {
  const [activeTab, setActiveTab] = useState("general");
  const [devices, setDevices] = useState(null);
  const [config, setConfig] = useState(null);
  const [envKeys, setEnvKeys] = useState({});
  const [newKeyInputs, setNewKeyInputs] = useState({});

  const [savingConfig, setSavingConfig] = useState(false);
  const [savingKeys, setSavingKeys] = useState(false);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    api.devices().then(setDevices).catch(() => setDevices({ devices: [], cuda_available: false }));
    api.getConfig().then(setConfig).catch((e) => console.error("Failed to load config", e));
    api.getEnvKeys().then((res) => {
      setEnvKeys(res.keys || {});
    }).catch((e) => console.error("Failed to load API keys", e));
  }, []);

  const showToast = (msg, type = "success") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const handleSaveConfig = async () => {
    setSavingConfig(true);
    try {
      const res = await api.saveConfig(config);
      if (res.status === "ok") {
        setConfig(res.config);
        showToast("Configuration saved to config.yaml!");
      }
    } catch (err) {
      showToast(err.message || "Failed to save config.yaml", "error");
    } finally {
      setSavingConfig(false);
    }
  };

  const handleSaveKeys = async () => {
    setSavingKeys(true);
    try {
      const res = await api.saveEnvKeys(newKeyInputs);
      setEnvKeys(res.keys || {});
      setNewKeyInputs({});
      showToast("API keys saved to .env file!");
    } catch (err) {
      showToast(err.message || "Failed to save API keys", "error");
    } finally {
      setSavingKeys(false);
    }
  };

  if (!config) {
    return <div className="card" style={{ maxWidth: 720 }}><div className="note">Loading settings from config.yaml…</div></div>;
  }

  return (
    <div style={{ maxWidth: 840, margin: "0 auto" }}>
      {toast && (
        <div style={{
          position: "fixed", top: 20, right: 20, zIndex: 9999,
          background: toast.type === "error" ? "#EF4444" : "#10B981",
          color: "#FFF", padding: "12px 20px", borderRadius: 8,
          boxShadow: "0 4px 12px rgba(0,0,0,0.3)", fontWeight: 600
        }}>
          {toast.msg}
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginBottom: 20, borderBottom: "1px solid var(--line)", paddingBottom: 10 }}>
        {[
          { id: "general", label: "⚙️ General Config" },
          { id: "keys", label: "🔑 API Keys (Rotation)" },
          { id: "categories", label: "📊 Clip Categories" },
          { id: "prompt", label: "💬 System Prompt" },
          { id: "subtitles", label: "🎨 Subtitle Defaults" },
          { id: "compute", label: "💻 Compute Hardware" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`chip ${activeTab === tab.id ? "active" : ""}`}
            style={{ padding: "8px 14px", cursor: "pointer", fontSize: 13 }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: General Config */}
      {activeTab === "general" && (
        <div className="card">
          <div className="card-h"><h2>General & Pipeline Settings (config.yaml)</h2></div>
          <div className="row" style={{ flexDirection: "column", alignItems: "stretch", gap: 14, marginTop: 10 }}>
            <div>
              <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>Default Output Directory</label>
              <input
                type="text"
                value={config.output_dir || "output"}
                onChange={(e) => setConfig({ ...config, output_dir: e.target.value })}
                className="input"
                style={{ width: "100%" }}
              />
              <span className="note">Directory where generated short clips and metadata are stored.</span>
            </div>

            <div>
              <label style={{ display: "block", marginBottom: 4, fontWeight: 600 }}>Default Aspect Ratio</label>
              <select
                value={config.aspect_ratio || "16:9"}
                onChange={(e) => setConfig({ ...config, aspect_ratio: e.target.value })}
                className="input"
                style={{ width: "100%" }}
              >
                <option value="16:9">16:9 (Landscape Widescreen)</option>
                <option value="9:16">9:16 (Vertical Shorts / Reels)</option>
              </select>
              <span className="note">Default aspect ratio for CLI and pipeline operations.</span>
            </div>

            <div style={{ marginTop: 10 }}>
              <button onClick={handleSaveConfig} disabled={savingConfig} className="btn primary">
                {savingConfig ? "Saving config.yaml…" : "💾 Save Settings to config.yaml"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: API Keys Rotation Pool */}
      {activeTab === "keys" && (
        <div className="card">
          <div className="card-h">
            <h2>Groq API Key Rotation Pool (.env)</h2>
          </div>
          <p className="note" style={{ marginBottom: 14 }}>
            Configure up to 10 Groq API keys. The 2D Intelligent Router rotates across keys and models continuously to prevent rate limits and maximize daily throughput (up to 1,000,000+ TPD).
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {Array.from({ length: 10 }).map((_, i) => {
              const keyName = `LLM_API_KEY_${i + 1}`;
              const maskedVal = envKeys[keyName] || "";
              const currentInput = newKeyInputs[keyName] !== undefined ? newKeyInputs[keyName] : maskedVal;

              return (
                <div key={keyName} style={{ background: "var(--bg-card)", padding: 10, borderRadius: 6, border: "1px solid var(--line)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>Key #{i + 1} ({keyName})</span>
                    <span style={{ fontSize: 11, color: maskedVal ? "#10B981" : "#9CA3AF" }}>
                      {maskedVal ? "✓ Configured" : "Empty"}
                    </span>
                  </div>
                  <input
                    type="password"
                    placeholder="gsk_..."
                    value={currentInput}
                    onChange={(e) => setNewKeyInputs({ ...newKeyInputs, [keyName]: e.target.value })}
                    className="input"
                    style={{ width: "100%", fontSize: 12 }}
                  />
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 16 }}>
            <button onClick={handleSaveKeys} disabled={savingKeys} className="btn primary">
              {savingKeys ? "Saving .env…" : "🔑 Save API Keys to .env"}
            </button>
          </div>
        </div>
      )}

      {/* Tab 3: Clip Categories */}
      {activeTab === "categories" && (
        <div className="card">
          <div className="card-h"><h2>Hierarchical Clip Extraction Categories</h2></div>
          <p className="note" style={{ marginBottom: 14 }}>
            Configure short, mid, and long duration tiers for clip extraction in config.yaml.
          </p>

          {["short", "mid", "long"].map((catKey) => {
            const cat = (config.clip_categories && config.clip_categories[catKey]) || {
              enabled: true, min_duration: 20, max_duration: 60, count: 2
            };

            const updateCategory = (field, val) => {
              setConfig({
                ...config,
                clip_categories: {
                  ...config.clip_categories,
                  [catKey]: { ...cat, [field]: val }
                }
              });
            };

            return (
              <div key={catKey} style={{ border: "1px solid var(--line)", padding: 14, borderRadius: 8, marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <h3 style={{ margin: 0, textTransform: "uppercase", fontSize: 14 }}>{catKey} Clips</h3>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
                    <input
                      type="checkbox"
                      checked={cat.enabled}
                      onChange={(e) => updateCategory("enabled", e.target.checked)}
                    />
                    Enabled
                  </label>
                </div>

                {cat.enabled && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                    <div>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Min Duration (sec)</label>
                      <input
                        type="number"
                        value={cat.min_duration}
                        onChange={(e) => updateCategory("min_duration", parseInt(e.target.value) || 10)}
                        className="input"
                        style={{ width: "100%" }}
                      />
                    </div>
                    <div>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Max Duration (sec)</label>
                      <input
                        type="number"
                        value={cat.max_duration}
                        onChange={(e) => updateCategory("max_duration", parseInt(e.target.value) || 60)}
                        className="input"
                        style={{ width: "100%" }}
                      />
                    </div>
                    <div>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Target Clip Count</label>
                      <input
                        type="number"
                        value={cat.count}
                        onChange={(e) => updateCategory("count", parseInt(e.target.value) || 1)}
                        className="input"
                        style={{ width: "100%" }}
                      />
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          <div style={{ marginTop: 10 }}>
            <button onClick={handleSaveConfig} disabled={savingConfig} className="btn primary">
              {savingConfig ? "Saving config.yaml…" : "💾 Save Categories to config.yaml"}
            </button>
          </div>
        </div>
      )}

      {/* Tab 4: System Prompt */}
      {activeTab === "prompt" && (
        <div className="card">
          <div className="card-h"><h2>Groq LLM System Prompt Template</h2></div>
          <p className="note" style={{ marginBottom: 10 }}>
            Custom prompt for LLM clip extraction. Use placeholders <code>{"{min_duration}"}</code> and <code>{"{max_duration}"}</code>.
          </p>

          <textarea
            rows={12}
            value={config.system_prompt || ""}
            onChange={(e) => setConfig({ ...config, system_prompt: e.target.value })}
            className="input"
            style={{ width: "100%", fontFamily: "monospace", fontSize: 12, lineHeight: 1.4 }}
          />

          <div style={{ marginTop: 12 }}>
            <button onClick={handleSaveConfig} disabled={savingConfig} className="btn primary">
              {savingConfig ? "Saving config.yaml…" : "💾 Save Prompt to config.yaml"}
            </button>
          </div>
        </div>
      )}

      {/* Tab 5: Subtitle Defaults */}
      {activeTab === "subtitles" && (
        <div className="card">
          <div className="card-h"><h2>Default Subtitle Styling</h2></div>
          <p className="note" style={{ marginBottom: 12 }}>
            Configure default caption font, size, and ASS color parameters in config.yaml.
          </p>

          {(() => {
            const sub = config.subtitles || {
              font_name: "Arial", font_size: 18, bold: true, primary_color: "&H00FFFFFF&", outline_color: "&H00000000&", margin_v: 30
            };
            const updateSub = (field, val) => {
              setConfig({
                ...config,
                subtitles: { ...sub, [field]: val }
              });
            };

            return (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Font Family</label>
                  <input
                    type="text"
                    value={sub.font_name}
                    onChange={(e) => updateSub("font_name", e.target.value)}
                    className="input"
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Font Size</label>
                  <input
                    type="number"
                    value={sub.font_size}
                    onChange={(e) => updateSub("font_size", parseInt(e.target.value) || 18)}
                    className="input"
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Primary Color (ASS Format)</label>
                  <input
                    type="text"
                    value={sub.primary_color}
                    onChange={(e) => updateSub("primary_color", e.target.value)}
                    className="input"
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Outline Color (ASS Format)</label>
                  <input
                    type="text"
                    value={sub.outline_color}
                    onChange={(e) => updateSub("outline_color", e.target.value)}
                    className="input"
                    style={{ width: "100%" }}
                  />
                </div>
                <div>
                  <label style={{ display: "block", fontSize: 12, marginBottom: 2 }}>Vertical Margin (px)</label>
                  <input
                    type="number"
                    value={sub.margin_v}
                    onChange={(e) => updateSub("margin_v", parseInt(e.target.value) || 30)}
                    className="input"
                    style={{ width: "100%" }}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", marginTop: 16 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: 600 }}>
                    <input
                      type="checkbox"
                      checked={sub.bold}
                      onChange={(e) => updateSub("bold", e.target.checked)}
                    />
                    Bold Font Weight
                  </label>
                </div>
              </div>
            );
          })()}

          <div style={{ marginTop: 16 }}>
            <button onClick={handleSaveConfig} disabled={savingConfig} className="btn primary">
              {savingConfig ? "Saving config.yaml…" : "💾 Save Subtitle Defaults to config.yaml"}
            </button>
          </div>
        </div>
      )}

      {/* Tab 6: Compute Hardware */}
      {activeTab === "compute" && (
        <div className="card">
          <div className="card-h"><h2>Compute & Acceleration Hardware</h2></div>
          {!devices ? <div className="note">Checking hardware…</div> : (
            <>
              <div className="row" style={{ justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
                <span className="note" style={{ margin: 0 }}>GPU (CUDA Acceleration)</span>
                <b>{devices.cuda_available ? (devices.gpu_name || "Available") : "Not available"}</b>
              </div>
              <div className="row" style={{ justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
                <span className="note" style={{ margin: 0 }}>Active Compute Device</span>
                <b>{(devices.default || "—").toUpperCase()}</b>
              </div>
              <div className="row" style={{ justifyContent: "space-between", padding: "10px 0" }}>
                <span className="note" style={{ margin: 0 }}>Supported Devices</span>
                <b>{(devices.devices || []).map((x) => x.toUpperCase()).join(" · ") || "—"}</b>
              </div>
              <div className="note" style={{ marginTop: 16 }}>
                Groq API accelerates transcription and 2-pass LLM event extraction. Local Whisper (CUDA/CPU) and FFmpeg act as high-performance local fallbacks.
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
