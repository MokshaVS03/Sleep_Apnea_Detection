import React, { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Play,
  Square,
  Settings,
  Download,
  TrendingUp,
  Wifi,
  WifiOff,
} from "lucide-react";

const ECGApneaDashboard = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [ecgData, setEcgData] = useState([]);
  const [currentSample, setCurrentSample] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [showSettings, setShowSettings] = useState(false);
  const [numSamples, setNumSamples] = useState(1200);
  const [timeout, setTimeout] = useState(30);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);

  const intervalRef = useRef(null);

  // Check API Connection
  const checkConnection = async () => {
    try {
      const response = await fetch(`${apiUrl}/health`);
      if (response.ok) {
        setIsConnected(true);
        setError(null);
        const statusResponse = await fetch(`${apiUrl}/status`);
        const statusData = await statusResponse.json();
        setSystemStatus(statusData);
      } else {
        setIsConnected(false);
        setError("API health check failed");
      }
    } catch (err) {
      setIsConnected(false);
      setError("Cannot connect to API. Make sure backend is running.");
      setSystemStatus(null);
    }
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  // Reconnect Serial
  const reconnectSerial = async () => {
    setIsReconnecting(true);
    setError(null);
    try {
      const response = await fetch(`${apiUrl}/reconnect-serial`, {
        method: "POST",
      });
      if (response.ok) {
        setError(null);
        alert("✅ Serial port reconnected!");
        await checkConnection();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || "Failed to reconnect");
      }
    } catch (err) {
      setError("Reconnect failed: " + err.message);
    } finally {
      setIsReconnecting(false);
    }
  };

  // Start Recording
  const startRecording = async () => {
    setIsRecording(true);
    setEcgData([]);
    setCurrentSample(0);
    setPrediction(null);
    setError(null);

    // Simulate visualization
    intervalRef.current = window.setInterval(() => {
      setCurrentSample((prev) => {
        if (prev >= numSamples) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          return prev;
        }

        const t = prev / 100;
        const value =
          Math.sin(t * 2 * Math.PI * 1.2) * 0.5 +
          Math.sin(t * 2 * Math.PI * 0.2) * 0.3 +
          (Math.random() - 0.5) * 0.1;

        setEcgData((prevData) => [...prevData, { x: prev, y: value }]);
        return prev + 1;
      });
    }, 10);

    // Call API
    try {
      setIsProcessing(true);
      const response = await fetch(`${apiUrl}/collect-and-predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          num_samples: numSamples,
          timeout: timeout,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "API request failed");
      }

      const result = await response.json();

      if (intervalRef.current) clearInterval(intervalRef.current);
      setIsRecording(false);
      setIsProcessing(false);

      setPrediction(result);
      setHistory((prev) => [result, ...prev.slice(0, 9)]);
    } catch (err) {
      setError(err.message || "Unknown error");
      setIsRecording(false);
      setIsProcessing(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
  };

  const stopRecording = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsRecording(false);
    setIsProcessing(false);
  };

  const exportData = () => {
    const dataStr = JSON.stringify({ ecgData, prediction, history }, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ecg_data_${Date.now()}.json`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* HEADER */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <Activity className="w-10 h-10 text-blue-400" strokeWidth={2.5} />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                ECG Apnea Detection System
              </h1>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-lg">
                {isConnected ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-400" />
                    <span className="text-sm text-green-400">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-red-400" />
                    <span className="text-sm text-red-400">Disconnected</span>
                  </>
                )}
              </div>

              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
              >
                <Settings className="w-6 h-6" />
              </button>
            </div>
          </div>

          <p className="text-slate-400 ml-12">
            Real-time sleep apnea detection from ECG signals
          </p>
        </div>

        {/* ERROR BANNER */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl flex items-center gap-3">
            <AlertCircle className="w-6 h-6 text-red-400" />
            <div>
              <div className="font-semibold text-red-400">Error</div>
              <div className="text-sm text-red-300">{error}</div>
            </div>
          </div>
        )}

        {/* SETTINGS PANEL */}
        {showSettings && (
          <div className="mb-6 p-6 bg-slate-800 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4">Configuration</h3>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="text-sm text-slate-400 mb-2 block">API URL</label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                />
              </div>

              <div>
                <label className="text-sm text-slate-400 mb-2 block">
                  Number of Samples
                </label>
                <input
                  type="number"
                  value={numSamples}
                  onChange={(e) => setNumSamples(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  min="100"
                  max="12000"
                />
              </div>

              <div>
                <label className="text-sm text-slate-400 mb-2 block">
                  Timeout (seconds)
                </label>
                <input
                  type="number"
                  value={timeout}
                  onChange={(e) => setTimeout(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  min="10"
                  max="120"
                />
              </div>
            </div>

            {systemStatus && (
              <div className="mt-4 p-4 bg-slate-900 rounded-lg">
                <h4 className="text-sm font-semibold mb-2 text-slate-300">
                  System Status
                </h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <span
                      className={
                        systemStatus.status === "operational"
                          ? "text-green-400"
                          : "text-yellow-400"
                      }
                    >
                      {systemStatus.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Model:</span>
                    <span
                      className={
                        systemStatus.model_loaded ? "text-green-400" : "text-red-400"
                      }
                    >
                      {systemStatus.model_loaded ? "Loaded" : "Not Loaded"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Serial:</span>
                    <span
                      className={
                        systemStatus.serial_connected
                          ? "text-green-400"
                          : "text-red-400"
                      }
                    >
                      {systemStatus.serial_connected ? "Connected" : "Disconnected"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Device:</span>
                    <span className="text-white">{systemStatus.device}</span>
                  </div>
                </div>

                {!systemStatus.serial_connected && (
                  <div className="mt-4">
                    <button
                      onClick={reconnectSerial}
                      disabled={isReconnecting}
                      className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                    >
                      {isReconnecting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          Reconnecting...
                        </>
                      ) : (
                        <>
                          <Wifi className="w-4 h-4" />
                          Reconnect Serial Port
                        </>
                      )}
                    </button>
                    <p className="text-xs text-slate-400 mt-2 text-center">
                      Close Arduino IDE/Serial Monitor first
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* CONTROL PANEL */}
        <div className="mb-6 p-6 bg-slate-800 rounded-xl border border-slate-700">
          <div className="flex items-center justify-between">
            <div className="flex gap-4">
              <button
                onClick={startRecording}
                disabled={isRecording || isProcessing || !isConnected}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                  isRecording || isProcessing || !isConnected
                    ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                    : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 shadow-lg"
                }`}
              >
                <Play className="w-5 h-5" />
                Start Recording
              </button>

              <button
                onClick={stopRecording}
                disabled={!isRecording}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                  !isRecording
                    ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                    : "bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700"
                }`}
              >
                <Square className="w-5 h-5" />
                Stop
              </button>
            </div>

            <button
              onClick={exportData}
              disabled={!prediction}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg disabled:opacity-50"
            >
              <Download className="w-5 h-5" />
              Export Data
            </button>
          </div>

          <div className="mt-4 flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isRecording
                    ? "bg-red-500 animate-pulse"
                    : isProcessing
                    ? "bg-yellow-500 animate-pulse"
                    : "bg-slate-600"
                }`}
              />
              <span className="text-slate-400">
                {isRecording ? "Recording" : isProcessing ? "Processing" : "Ready"}
              </span>
            </div>

            <div className="text-slate-400">
              Samples:{" "}
              <span className="text-white font-mono">
                {currentSample}/{numSamples}
              </span>
            </div>

            <div className="text-slate-400">
              Progress:{" "}
              <span className="text-white font-mono">
                {((currentSample / numSamples) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* MAIN GRID */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          {/* ECG CHART */}
          <div className="col-span-2 bg-slate-800 rounded-xl border border-slate-700 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              ECG Signal (Real-time)
            </h3>

            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ecgData.slice(-500)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="x" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "1px solid #475569",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* DETECTION RESULT */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
            <h3 className="text-lg font-semibold mb-4">Detection Result</h3>

            {isProcessing && (
              <div className="flex flex-col items-center justify-center h-64">
                <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
                <p className="text-slate-400">Analyzing ECG signal...</p>
              </div>
            )}

            {!isProcessing && prediction && (
              <div className="space-y-4">
                <div
                  className={`p-4 rounded-lg ${
                    prediction.prediction === "APNEA"
                      ? "bg-red-500/20 border border-red-500/50"
                      : "bg-green-500/20 border border-green-500/50"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {prediction.prediction === "APNEA" ? (
                      <AlertCircle className="w-6 h-6 text-red-400" />
                    ) : (
                      <CheckCircle className="w-6 h-6 text-green-400" />
                    )}
                    <span className="font-semibold">
                      {prediction.prediction === "APNEA"
                        ? "Apnea Detected"
                        : "Normal ECG"}
                    </span>
                  </div>

                  <div className="text-3xl font-bold">
                    {(prediction.probability * 100).toFixed(2)}%
                  </div>
                  <div className="text-sm text-slate-400 mt-2">Probability</div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Timestamp:</span>
                    <span className="font-mono text-xs">
                      {new Date(prediction.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Confidence:</span>
                    <span className="font-mono">
                      {prediction.confidence.toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Samples:</span>
                    <span className="font-mono">{prediction.samples_analyzed}</span>
                  </div>
                </div>
              </div>
            )}

            {!isProcessing && !prediction && (
              <div className="flex flex-col items-center justify-center h-64 text-slate-500">
                <Activity className="w-12 h-12 mb-4" />
                <p>Start recording to see results</p>
              </div>
            )}
          </div>
        </div>

        {/* HISTORY */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-4">Detection History</h3>

          {history.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No recordings yet</p>
          ) : (
            <div className="space-y-2">
              {history.map((record, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-slate-900 rounded-lg hover:bg-slate-700 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    {record.prediction === "APNEA" ? (
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    ) : (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    )}
                    <div>
                      <div className="font-semibold">
                        {record.prediction === "APNEA" ? "Apnea Detected" : "Normal"}
                      </div>
                      <div className="text-sm text-slate-400">
                        {new Date(record.timestamp).toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-xl font-bold">
                      {(record.probability * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-400">
                      {record.samples_analyzed} samples
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ECGApneaDashboard;