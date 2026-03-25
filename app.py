from flask import Flask, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import os

app = Flask(__name__)

# Initialize storage safely
app.last_results = []

# Training Data
texts = [
    "server crashed badly",
    "disk failure detected",
    "system error occurred",
    "network failure",
    "critical failure detected",
    "system critical crash",
    "major outage occurred",

    "cpu usage high",
    "memory usage high",
    "latency high",
    "disk almost full",

    "system running smoothly",
    "login success",
    "service started successfully",
    "backup completed",
    "operation completed"
]

labels = [
    "CRITICAL","CRITICAL","CRITICAL","CRITICAL","CRITICAL","CRITICAL","CRITICAL",
    "WARNING","WARNING","WARNING","WARNING",
    "INFO","INFO","INFO","INFO","INFO"
]

# Train Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)


# Hybrid classification
def classify_log(text):
    text = text.lower()

    # Rule-based
    if any(word in text for word in ["crash", "failed", "failure", "error", "outage", "critical"]):
        return "CRITICAL"
    elif any(word in text for word in ["high", "slow", "latency", "delay", "usage", "full"]):
        return "WARNING"
    elif any(word in text for word in ["success", "started", "completed", "running", "login"]):
        return "INFO"

    # ML fallback
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)
    return prediction[0]


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    critical = warning = info = 0
    logs = ""

    if request.method == "POST":
        logs = request.form.get("logs", "")
        lines = logs.split("\n")

        for line in lines:
            if line.strip():
                message = line.split(":", 1)[-1]

                category = classify_log(message)

                if category == "CRITICAL":
                    critical += 1
                elif category == "WARNING":
                    warning += 1
                else:
                    info += 1

                results.append((line, category))

    app.last_results = results

    return render_template(
        "index.html",
        results=results,
        critical=critical,
        warning=warning,
        info=info,
        logs=logs
    )


@app.route("/download")
def download():
    if not app.last_results:
        return "No data available to download."

    with open("report.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Log", "Category"])

        for line, category in app.last_results:
            writer.writerow([line, category])

    return send_file("report.csv", as_attachment=True)


# Run app (important for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)