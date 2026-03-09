import { useState } from "react";
import axios from "axios";

function App() {
  const initialForm = {
    Hours_Studied: "",
    Sleep_Hours: "",
    Attendance: "",
    Previous_Scores: ""
  };
  const [form, setForm] = useState(initialForm);

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      setError("");
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, Number(v)])
      );
      const response = await axios.post(`${API_URL}/predict`, payload);
      setResult(response.data["Predicted Score"]);
      setForm(initialForm);
    } catch (err) {
      setResult(null);
      setError(err?.response?.data?.detail || "Prediction request failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center overflow-y-hidden bg-gray-100">
      <div className="bg-white p-8 rounded shadow-lg w-96">
        <h2 className="text-center py-4 text-2xl font-bold">Student Score Predictor</h2>

        {Object.keys(form).map((key) => (
          <div key={key} className="flex flex-col">
            <h5 className="text-gray-400 font-bold">{key}</h5>
            <input
              type="number"
              name={key}
              placeholder={key}
              value={form[key]}
              onChange={handleChange}
              className="border p-2 rounded mb-4 border-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        ))}

        <button
          onClick={handleSubmit}
          className="px-4 py-1 border-2 border-indigo-600 shadow-sm rounded-md bg-indigo-500 cursor-pointer hover:bg-indigo-600 text-white"
        >
          Predict
        </button>

        {result !== null && <h3 className="text-lg font-bold text-gray-500 my-2">Predicted Score: {result.toFixed(2)}</h3>}
        {error && <p className="mt-3 text-sm text-red-600">{String(error)}</p>}
      </div>
    </div>
  );
}

export default App;
