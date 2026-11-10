import { useState, useEffect } from 'react';
import { Activity, Info, XCircle, ArrowRight, ShieldCheck, AlertCircle } from 'lucide-react';
import { getHealth, getPresets } from './api/client';
import type { HealthResponse, DemoPreset } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<'console' | 'presets' | 'metrics' | 'history'>('console');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [presets, setPresets] = useState<DemoPreset[]>([]);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Verification form state
  const [claimText, setClaimText] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<DemoPreset | null>(null);
  const [fusionMode, setFusionMode] = useState<'feature_fusion' | 'decision_fusion'>('feature_fusion');

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        setLoadingHealth(true);
        const data = await getHealth();
        setHealth(data);
        setHealthError(null);
      } catch (err: any) {
        setHealthError(err.message || 'Backend connection failed');
      } finally {
        setLoadingHealth(false);
      }
    };

    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchDemoPresets = async () => {
      try {
        const data = await getPresets();
        setPresets(data);
      } catch (err) {
        console.error('Failed to load demo presets', err);
      }
    };
    fetchDemoPresets();
  }, []);

  const handleSelectPreset = (preset: DemoPreset) => {
    setSelectedPreset(preset);
    setClaimText(preset.text);
    setActiveTab('console');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Academic Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded bg-slate-900 flex items-center justify-center text-white font-bold text-lg tracking-tight">
                V
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="text-lg font-semibold text-slate-900 tracking-tight">Veritas AI</h1>
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-mono border border-slate-200">
                    NLP Capstone
                  </span>
                </div>
                <p className="text-xs text-slate-500">
                  Multimodal Content Verification & Uncertainty Quantification System
                </p>
              </div>
            </div>

            {/* Live System Status Pill */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-full border border-slate-200 bg-slate-50">
                <span className={`w-2 h-2 rounded-full ${loadingHealth ? 'bg-amber-400 animate-pulse' : health ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                <span className="text-slate-600">
                  {loadingHealth && !health
                    ? 'Connecting...'
                    : health
                    ? `Backend Online (v${health.version}) • ${health.device.toUpperCase()}`
                    : 'Backend Disconnected'}
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Bar */}
          <nav className="flex space-x-8 -mb-px">
            <button
              onClick={() => setActiveTab('console')}
              className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'console'
                  ? 'border-slate-900 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              Verification Console
            </button>
            <button
              onClick={() => setActiveTab('presets')}
              className={`py-3 text-sm font-medium border-b-2 transition-colors flex items-center space-x-1.5 ${
                activeTab === 'presets'
                  ? 'border-slate-900 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <span>Demo Presets</span>
              <span className="text-xs bg-slate-100 text-slate-600 px-1.5 py-0.2 rounded border border-slate-200 font-mono">
                {presets.length}
              </span>
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'metrics'
                  ? 'border-slate-900 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              Benchmark Metrics & ECE
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'history'
                  ? 'border-slate-900 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              Audit History
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Backend Disconnected Alert */}
        {healthError && (
          <div className="mb-6 p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 flex items-start space-x-3 text-sm">
            <XCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Backend API Unreachable</p>
              <p className="text-xs text-rose-700 mt-1">
                Unable to contact the FastAPI backend service at <code className="font-mono bg-rose-100 px-1 rounded">http://127.0.0.1:8000/api/health</code>. Please start the backend server with <code className="font-mono bg-rose-100 px-1 rounded">python backend/run.py</code>.
              </p>
            </div>
          </div>
        )}

        {/* Tab 1: Verification Console */}
        {activeTab === 'console' && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-base font-semibold text-slate-900">Multimodal Claim Ingestion</h2>
                  <p className="text-xs text-slate-500">
                    Submit paired textual claim and accompanying visual media to evaluate content veracity, local explainability attributions, and selective calibration confidence.
                  </p>
                </div>
                {selectedPreset && (
                  <span className="text-xs px-2.5 py-1 rounded bg-slate-100 border border-slate-200 text-slate-600 font-mono">
                    Loaded Preset: {selectedPreset.id} ({selectedPreset.ground_truth})
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-2">
                    Textual Claim / Submission Title
                  </label>
                  <textarea
                    rows={5}
                    value={claimText}
                    onChange={(e) => setClaimText(e.target.value)}
                    placeholder="Enter social media headline, news caption, or submitted textual claim..."
                    className="w-full text-sm p-3 border border-slate-300 rounded-md focus:ring-1 focus:ring-slate-900 focus:border-slate-900 bg-white"
                  />
                  <div className="flex justify-between items-center text-xs text-slate-400 mt-1 font-mono">
                    <span>{claimText ? claimText.split(/\s+/).filter(Boolean).length : 0} tokens</span>
                    <span>max 128 tokens</span>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-2">
                    Paired Visual Context
                  </label>
                  {selectedPreset?.image_url ? (
                    <div className="border border-slate-200 rounded-md p-3 bg-slate-50 flex items-center space-x-4">
                      <img
                        src={selectedPreset.image_url}
                        alt="Preset thumbnail"
                        className="w-24 h-24 object-cover rounded border border-slate-300 bg-slate-200"
                      />
                      <div className="flex-1 text-xs text-slate-600 space-y-1">
                        <p className="font-semibold text-slate-900">{selectedPreset.image_file}</p>
                        <p className="text-slate-500">{selectedPreset.description}</p>
                        <p className="font-mono text-slate-400">License: {selectedPreset.license}</p>
                        <button
                          onClick={() => setSelectedPreset(null)}
                          className="text-xs text-rose-600 hover:underline pt-1 block"
                        >
                          Remove image / Clear preset
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="border-2 border-dashed border-slate-300 rounded-md p-6 flex flex-col items-center justify-center text-center hover:bg-slate-50 transition-colors cursor-pointer min-h-[140px]">
                      <div className="text-xs text-slate-600">
                        <span className="font-semibold text-slate-900">Click to upload image</span> or drag and drop
                      </div>
                      <p className="text-xs text-slate-400 mt-1">PNG, JPG, WEBP up to 5MB (processed at 224×224 RGB)</p>
                      <button
                        onClick={() => setActiveTab('presets')}
                        className="mt-3 text-xs text-indigo-600 hover:underline font-medium"
                      >
                        Or pick a sample from Demo Presets →
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center space-x-4 text-xs text-slate-600">
                  <span className="font-semibold">Fusion Mode:</span>
                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="radio"
                      name="fusion"
                      checked={fusionMode === 'feature_fusion'}
                      onChange={() => setFusionMode('feature_fusion')}
                      className="text-slate-900"
                    />
                    <span>Feature Fusion (Joint Concatenation)</span>
                  </label>
                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="radio"
                      name="fusion"
                      checked={fusionMode === 'decision_fusion'}
                      onChange={() => setFusionMode('decision_fusion')}
                      className="text-slate-900"
                    />
                    <span>Decision Fusion (OOF Ensembled)</span>
                  </label>
                </div>

                <button
                  type="button"
                  disabled={!claimText.trim()}
                  className="w-full sm:w-auto px-5 py-2.5 bg-slate-900 text-white rounded text-sm font-medium hover:bg-slate-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                >
                  Verify Multimodal Claim
                </button>
              </div>
            </div>

            {/* Methodology Note */}
            <div className="p-4 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-600 space-y-1">
              <div className="flex items-center space-x-2 font-semibold text-slate-800">
                <Info className="w-4 h-4 text-slate-600" />
                <span>Academic Scientific Framing & Scope Note</span>
              </div>
              <p>
                Veritas AI detects multimodal misinformation correlations between text and imagery. Visual explainability maps (Grad-CAM) depict model activation saliency rather than forensic digital tampering detection. Selective confidence thresholding routes ambiguous claims ($\max P &lt; \tau^*$) to <span className="font-semibold text-amber-700">UNCERTAIN</span> status to prioritize high precision among accepted determinations.
              </p>
            </div>
          </div>
        )}

        {/* Tab 2: Demo Presets */}
        {activeTab === 'presets' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-2">
              <div>
                <h2 className="text-base font-semibold text-slate-900">Curated Development & Demo Dataset</h2>
                <p className="text-xs text-slate-500">
                  Isolated 20 verified multimodal samples under explicit Public Domain (CC0) or Creative Commons licenses. Kept strictly distinct from Fakeddit benchmark data.
                </p>
              </div>
              <span className="text-xs font-mono bg-white px-2.5 py-1 rounded border border-slate-200 text-slate-600">
                {presets.length} Verified Entries
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {presets.map((preset) => (
                <div key={preset.id} className="bg-white rounded-lg border border-slate-200 p-4 flex flex-col justify-between shadow-sm hover:border-slate-300 transition-colors">
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <span className={`text-xs px-2 py-0.5 rounded font-mono font-medium border flex items-center space-x-1 ${
                        preset.ground_truth === 'REAL'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : 'bg-rose-50 text-rose-700 border-rose-200'
                      }`}>
                        {preset.ground_truth === 'REAL' ? (
                          <ShieldCheck className="w-3 h-3 text-emerald-600" />
                        ) : (
                          <AlertCircle className="w-3 h-3 text-rose-600" />
                        )}
                        <span>{preset.ground_truth}</span>
                      </span>
                      <span className="text-xs text-slate-400 font-mono">{preset.category}</span>
                    </div>

                    {preset.image_url && (
                      <div className="mb-3 rounded overflow-hidden border border-slate-100 bg-slate-50">
                        <img
                          src={preset.image_url}
                          alt={preset.id}
                          className="w-full h-32 object-cover"
                          loading="lazy"
                        />
                      </div>
                    )}

                    <p className="text-sm font-medium text-slate-900 line-clamp-3 mb-2">
                      "{preset.text}"
                    </p>
                    <p className="text-xs text-slate-500 line-clamp-2 mb-3">
                      {preset.description}
                    </p>
                  </div>

                  <div className="pt-3 border-t border-slate-100 text-xs text-slate-500 space-y-2">
                    <div className="space-y-0.5">
                      <p className="truncate">
                        <span className="font-semibold text-slate-700">License:</span> {preset.license}
                      </p>
                      <p className="truncate">
                        <span className="font-semibold text-slate-700">Source:</span>{' '}
                        <a href={preset.source_url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
                          {preset.source_url}
                        </a>
                      </p>
                    </div>

                    <button
                      onClick={() => handleSelectPreset(preset)}
                      className="w-full py-1.5 px-3 bg-slate-50 hover:bg-slate-100 text-slate-800 border border-slate-200 rounded font-medium text-xs flex items-center justify-center space-x-1 transition-colors"
                    >
                      <span>Load into Console</span>
                      <ArrowRight className="w-3 h-3 text-slate-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Benchmark Metrics & ECE */}
        {activeTab === 'metrics' && (
          <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm space-y-6">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Experimental Evaluation & Calibration Dashboard</h2>
              <p className="text-xs text-slate-500">
                Empirical metrics computed strictly on the untouched test split (750 samples) from Fakeddit. Zero fabricated data.
              </p>
            </div>

            <div className="p-8 text-center border-2 border-dashed border-slate-200 rounded-lg text-slate-500 text-sm">
              <Activity className="w-8 h-8 mx-auto text-slate-400 mb-2" />
              <p className="font-semibold text-slate-700">Awaiting Google Colab Model Training Run</p>
              <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
                Once unimodal and multimodal models are trained in Google Colab (Days 2 & 3), actual test-set evaluations will be exported to <code className="font-mono bg-slate-100 px-1 rounded">backend/models/experiment_results.json</code> and live metrics will render here.
              </p>
            </div>
          </div>
        )}

        {/* Tab 4: Audit History */}
        {activeTab === 'history' && (
          <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
            <h2 className="text-base font-semibold text-slate-900 mb-1">Verification Audit Log</h2>
            <p className="text-xs text-slate-500 mb-4">
              Chronological log of submitted multimodal claims persisted in the local SQLite database (<code className="font-mono text-slate-700">veritas.db</code>).
            </p>
            <div className="text-center py-12 text-slate-400 text-xs">
              No verification records logged yet. Records will populate as claims are evaluated.
            </div>
          </div>
        )}
      </main>

      {/* Academic Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <span>Veritas AI — Multimodal NLP Capstone</span>
            <span>•</span>
            <span>Fakeddit Benchmark</span>
            <span>•</span>
            <span>DistilBERT + ResNet50</span>
          </div>
          <div className="mt-2 sm:mt-0 font-mono text-slate-400">
            PyTorch • SHAP Partition • Grad-CAM • FastAPI
          </div>
        </div>
      </footer>
    </div>
  );
}
