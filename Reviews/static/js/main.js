function analyzeReview() {

    const reviewText = document.getElementById("review_text").value;
    const modelChoice = document.getElementById("model").value;

    if (reviewText.trim() === "") {
        alert("Please enter review text.");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            review_text: reviewText,
            model: modelChoice
        })
    })
    .then(response => response.json())
    .then(data => {

        const resultBox = document.getElementById("resultBox");
        const predictionText = document.getElementById("predictionText");
        const confidenceText = document.getElementById("confidenceText");
        const confidenceBar = document.getElementById("confidenceBar");
        const wordDiv = document.getElementById("wordImportance");

        resultBox.style.display = "block";

        predictionText.innerText = data.prediction;
        confidenceText.innerText = data.confidence;

        confidenceBar.style.width = data.confidence + "%";
        confidenceBar.style.backgroundColor =
            data.prediction.includes("Fake") ? "#dc3545" : "#28a745";

        resultBox.className = "result-box " +
            (data.prediction.includes("Fake") ? "fake" : "genuine");

        // ------------------------------
        // ✅ WORD IMPORTANCE SECTION
        // ------------------------------
        wordDiv.innerHTML = "";

        if (data.explanation && Object.keys(data.explanation).length > 0) {

            let html = "<h3>🔎 Word Importance</h3>";

            if (data.explanation.fake_indicators.length > 0) {
                html += "<b>Fake Indicators:</b><ul>";
                data.explanation.fake_indicators.forEach(word => {
                    html += `<li style="color:red">${word}</li>`;
                });
                html += "</ul>";
            }

            if (data.explanation.truthful_indicators.length > 0) {
                html += "<b>Truthful Indicators:</b><ul>";
                data.explanation.truthful_indicators.forEach(word => {
                    html += `<li style="color:green">${word}</li>`;
                });
                html += "</ul>";
            }

            wordDiv.innerHTML = html;
        }

    })
    .catch(error => {
        alert("Backend connection error.");
        console.error(error);
    });
}

function uploadCSV() {

    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);

    fetch("/batch_predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        const tableBody = document.querySelector("#batchTable tbody");
        tableBody.innerHTML = "";

        data.results.forEach(item => {

            const row = `
                <tr>
                    <td>${item.review}</td>
                    <td>${item.prediction}</td>
                    <td>${item.confidence}%</td>
                </tr>
            `;

            tableBody.innerHTML += row;
        });
    });
}