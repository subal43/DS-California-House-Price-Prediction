document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');
    const predictedValueElement = document.getElementById('predictedValue');
    const submitBtn = document.getElementById('predictBtn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            
            if (result.error) {
                alert('Error: ' + result.error);
            } else {
                // Format currency
                const formattedPrice = new Intl.NumberFormat('en-US', {
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                }).format(result.prediction);

                // Update UI
                predictedValueElement.textContent = formattedPrice;
                resultContainer.classList.remove('hidden');
                
                // Scroll to result
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while fetching the prediction. Please try again.');
        } finally {
            // Reset button state
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });
});
