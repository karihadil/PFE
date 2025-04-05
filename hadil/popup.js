document.getElementById('calculate').addEventListener('click', async function () {
    const num1 = parseFloat(document.getElementById('number1').value);
    const num2 = parseFloat(document.getElementById('number2').value);

    // Validate input range (FastAPI already enforces this, but we check to prevent unnecessary requests)
    if (num1 <= 20 || num1 >= 200 || num2 <= 20 || num2 >= 200) {
        document.getElementById('result').innerText = "❌ Values must be between 20 and 200.";
        return;
    }

    try {
        const response = await fetch(`http://127.0.0.1:8000/code?number1=${num1}&number2=${num2}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Ensure FastAPI response has the correct structure
        if (typeof data.bmi === "number" && typeof data.message === "string") {
            document.getElementById('result').innerHTML = `✅ <b>BMI:</b> ${data.bmi.toFixed(4)}<br>💬 <b>Message:</b> ${data.message}`;
        } else {
            throw new Error("Unexpected response format.");
        }
    } catch (error) {
        document.getElementById('result').innerText = `❌ Error: ${error.message}`;
    }
});
