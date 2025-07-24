document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('uploadForm');
  const imageInput = document.getElementById('imageInput');
  const previewImage = document.getElementById('previewImage');
  const resultText = document.getElementById('resultText');
  const submitBtn = document.getElementById('submitBtn');
  const confidence = document.getElementById('confidence');
  const processingInfo = document.getElementById('processingInfo');

  imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
      };
      reader.readAsDataURL(file);
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Update UI to show processing state
    resultText.textContent = '🔄 Processing image and detecting license plate...';
    submitBtn.disabled = true;
    submitBtn.textContent = '🔄 Processing...';
    confidence.textContent = '';
    processingInfo.textContent = '';

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.error) {
        resultText.textContent = `❌ Error: ${data.error}`;
        confidence.textContent = '';
        processingInfo.textContent = '';
      } else {
        // Display the detected license plate
        if (data.extracted_text) {
          resultText.textContent = data.extracted_text;
          
          // Show confidence level
          if (data.confidence) {
            const confidenceEmoji = {
              'high': '🟢',
              'medium': '🟡',
              'low': '🔴'
            };
            confidence.textContent = `${confidenceEmoji[data.confidence] || '⚪'} Confidence: ${data.confidence.toUpperCase()}`;
            confidence.className = `confidence-indicator ${data.confidence}`;
          }
          
          // Show processing information
          if (data.processing_steps) {
            let info = [];
            if (data.processing_steps.plate_detected) {
              info.push('✅ License plate region detected');
            } else {
              info.push('ℹ️ Used full image (no plate region detected)');
            }
            if (data.processing_steps.preprocessed) {
              info.push('✅ Image preprocessed for OCR');
            }
            if (data.processing_steps.validated) {
              info.push('✅ Text validated and cleaned');
            }
            processingInfo.innerHTML = info.join('<br>');
          }
        } else {
          resultText.textContent = '❌ No license plate number could be detected in the image.';
          confidence.textContent = '';
          processingInfo.textContent = 'Try uploading a clearer image with a visible license plate.';
        }
      }
    } catch (error) {
      console.error('Error:', error);
      resultText.textContent = '❌ Server error! Please try again.';
      confidence.textContent = '';
      processingInfo.textContent = '';
    } finally {
      // Reset button state
      submitBtn.disabled = false;
      submitBtn.textContent = '🔍 Recognize License Plate';
    }
  });
});
