import React, { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const classNames = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]; // Update with your classes

  const onFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      const fileBuffer = await selectedFile.arrayBuffer();
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
        },
        body: fileBuffer,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data = await response.json();
      setPrediction({
        className: classNames[data.predicted_class] || "Unknown",
        confidence: (data.confidence * 100).toFixed(2),
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Garbage Classifier</h1>
      <form onSubmit={onSubmit} style={styles.form}>
        <input type="file" accept="image/*" onChange={onFileChange} style={styles.fileInput} />
        <button type="submit" disabled={loading || !selectedFile} style={styles.button}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {prediction && (
        <div style={styles.result}>
          <h2>Prediction Result</h2>
          <p>
            <strong>Class:</strong> {prediction.className}
          </p>
          <p>
            <strong>Confidence:</strong> {prediction.confidence}%
          </p>
        </div>
      )}

      {error && <p style={styles.error}>{error}</p>}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 480,
    margin: "40px auto",
    padding: 20,
    borderRadius: 10,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    backgroundColor: "#f9f9f9",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    textAlign: "center",
  },
  title: {
    color: "#333",
    marginBottom: 24,
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    alignItems: "center",
  },
  fileInput: {
    fontSize: 16,
  },
  button: {
    padding: "10px 24px",
    fontSize: 16,
    backgroundColor: "#007bff",
    color: "#fff",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    transition: "background-color 0.3s ease",
  },
  result: {
    marginTop: 30,
    color: "#222",
    backgroundColor: "#e7f3ff",
    borderRadius: 8,
    padding: 20,
  },
  error: {
    marginTop: 20,
    color: "#d9534f",
    fontWeight: "600",
  },
};

export default App;
