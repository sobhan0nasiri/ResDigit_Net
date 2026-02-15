import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import {
  Trash2,
  Send,
  Activity,
  CheckCircle,
  Clock,
  AlertCircle,
  RefreshCw,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";
const CANVAS_SIZE = 280;

export default function App() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // تغییر: لیست مدل‌ها حالا خالی است و پر می‌شود
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);

  const [history, setHistory] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true); // وضعیت بارگذاری مدل‌ها

  // تنظیمات اولیه بوم نقاشی (Canvas)
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    const ctx = canvas.getContext("2d");
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 12;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctxRef.current = ctx;
  }, []);

  // --- بخش جدید: دریافت لیست مدل‌ها از پوشه سرور ---
  const fetchModels = async () => {
    setIsLoadingModels(true);
    try {
      const response = await axios.get(`${API_URL}/models`);
      setAvailableModels(response.data);

      // اگر مدل وجود داشت، اولی را به صورت پیش‌فرض انتخاب کن
      if (response.data.length > 0) {
        // اختیاری: اگر بخواهی همه مدل‌ها پیش‌فرض انتخاب شوند یا فقط اولی
        // setSelectedModels([response.data[0].id]);
      }
    } catch (error) {
      console.error("Error fetching models:", error);
      alert("خطا در ارتباط با سرور برای دریافت لیست مدل‌ها");
    } finally {
      setIsLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);
  // ------------------------------------------------

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = nativeEvent;
    ctxRef.current.lineTo(offsetX, offsetY);
    ctxRef.current.strokeStyle = "black";
    ctxRef.current.stroke();
  };

  const stopDrawing = () => {
    ctxRef.current.closePath();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    ctxRef.current.fillStyle = "white";
    ctxRef.current.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  };

  const handleProcess = async () => {
    if (selectedModels.length === 0) {
      alert("Please select at least one model to process the image.");
      return;
    }

    try {
      const imageBase64 = canvasRef.current.toDataURL("image/png");

      const modelsPayload = availableModels
        .filter((model) => selectedModels.includes(model.id))
        .map((model) => ({
          id: model.id,
          name: model.name,
        }));

      const response = await axios.post(`${API_URL}/submit_job`, {
        image: imageBase64,
        models: modelsPayload,
      });

      const { job_id } = response.data;

      const newHistoryItem = {
        id: job_id,
        image: imageBase64,
        models: modelsPayload,
        status: "pending",
        results: null,
        timestamp: new Date().toLocaleTimeString("fa-IR"),
        error: null,
      };

      setHistory((prev) => [newHistoryItem, ...prev]);
    } catch (error) {
      console.error("Error submitting job:", error);

      if (error.response && error.response.data) {
        alert(`Error: ${JSON.stringify(error.response.data)}`);
      } else {
        alert("Failed to submit job. Please try again.");
      }
    }
  };

  useEffect(() => {
    const pendingJobs = history.filter((item) => item.status === "pending");

    if (pendingJobs.length === 0) return;

    const intervalId = setInterval(async () => {
      for (const job of pendingJobs) {
        try {
          const res = await axios.get(`${API_URL}/job_status/${job.id}`);
          const data = res.data;

          if (data.status === "completed") {
            setHistory((prevHistory) =>
              prevHistory.map((item) =>
                item.id === job.id
                  ? { ...item, status: "completed", results: data.results }
                  : item,
              ),
            );
          }
        } catch (err) {
          console.error("Polling error for job:", job.id, err);
        }
      }
    }, 2000);

    return () => clearInterval(intervalId);
  }, [history]);

  const toggleModel = (modelId) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId],
    );
  };

  return (
    <div className="container">
      {/* پنل تاریخچه */}
      <div className="panel">
        <h2>
          History & Queue <Activity size={16} />
        </h2>
        <div className="history-list">
          {history.length === 0 && (
            <p style={{ opacity: 0.5, textAlign: "center" }}>
              Drawing digit and processing.
            </p>
          )}

          {history.map((item) => (
            <div
              key={item.id}
              className={`history-item ${item.status === "completed" ? "done" : "pending"}`}
            >
              <img src={item.image} alt="drawn" className="history-img" />

              <div className="history-info">
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.8rem",
                    opacity: 0.7,
                  }}
                >
                  <span>{item.timestamp}</span>
                  {item.status === "pending" ? (
                    <Clock size={14} />
                  ) : (
                    <CheckCircle size={14} color="#a6e3a1" />
                  )}
                </div>

                {item.status === "pending" ? (
                  <div style={{ color: "#f9e2af", marginTop: "5px" }}>
                    Processing ({item.models.length} Model)...
                  </div>
                ) : (
                  <div style={{ marginTop: "5px" }}>
                    {/* نمایش نتایج مدل‌ها */}
                    {item.results &&
                      Object.entries(item.results).map(
                        ([modelName, resultData]) => (
                          <div
                            key={modelName}
                            style={{
                              marginBottom: "4px",
                              borderBottom: "1px dashed #444",
                              paddingBottom: "2px",
                            }}
                          >
                            <span
                              style={{ color: "#89b4fa", fontSize: "0.85rem" }}
                            >
                              {modelName}:{" "}
                            </span>
                            <strong
                              style={{ color: "#a6e3a1", fontSize: "1rem" }}
                            >
                              {resultData.digit}
                            </strong>
                            <span
                              style={{
                                fontSize: "0.7rem",
                                color: "#666",
                                marginLeft: "5px",
                              }}
                            >
                              ({resultData.confidence})
                            </span>
                          </div>
                        ),
                      )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* پنل نقاشی */}
      <div className="panel canvas-area">
        <h2 style={{ border: "none", marginBottom: 0 }}>Drawing Board</h2>
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
        />
        <div className="btn-group">
          <button className="btn-danger" onClick={clearCanvas}>
            <Trash2 size={18} style={{ marginRight: 5 }} /> Clear
          </button>
          <button className="btn-primary" onClick={handleProcess}>
            Process <Send size={18} style={{ marginLeft: 5 }} />
          </button>
        </div>
        <p style={{ fontSize: "0.8rem", opacity: 0.6 }}>
          {" "}
          Write a Digit (0-9){" "}
        </p>
      </div>

      {/* پنل انتخاب مدل‌ها */}
      <div className="panel">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h2>Select Models</h2>
          <button
            onClick={fetchModels}
            style={{
              padding: "5px",
              background: "transparent",
              border: "none",
            }}
            title="Refresh Models"
          >
            <RefreshCw size={16} color="#89b4fa" />
          </button>
        </div>

        {isLoadingModels ? (
          <p style={{ opacity: 0.5, fontStyle: "italic" }}>
            Loading models form folder...
          </p>
        ) : availableModels.length === 0 ? (
          <div
            style={{
              color: "#f38ba8",
              fontSize: "0.9rem",
              border: "1px dashed #f38ba8",
              padding: "10px",
              borderRadius: "8px",
            }}
          >
            <AlertCircle size={16} style={{ marginBottom: "-3px" }} /> هیچ مدلی
            یافت نشد. <br />
            لطفا فایل‌های <code>.pth</code> را در پوشه <code>models</code> قرار
            دهید.
          </div>
        ) : (
          availableModels.map((model) => (
            <label key={model.id} className="model-option">
              <input
                type="checkbox"
                checked={selectedModels.includes(model.id)}
                onChange={() => toggleModel(model.id)}
              />
              <span style={{ wordBreak: "break-all" }}>{model.name}</span>
            </label>
          ))
        )}

        <div
          style={{
            marginTop: "auto",
            fontSize: "0.8rem",
            opacity: 0.5,
            borderTop: "1px solid #444",
            paddingTop: "10px",
          }}
        >
          Selected Models: {selectedModels.length}
        </div>
      </div>
    </div>
  );
}
