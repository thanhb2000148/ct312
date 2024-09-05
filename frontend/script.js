function classify() {
    const data = {
        features: [
            parseFloat(document.getElementById('sepal_length').value),
            parseFloat(document.getElementById('sepal_width').value),
            parseFloat(document.getElementById('petal_length').value),
            parseFloat(document.getElementById('petal_width').value)
        ]
    };

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('result').innerText = "Kết quả phân lớp: " + result.prediction;
    })
    .catch(error => console.error('Error:', error));
}
