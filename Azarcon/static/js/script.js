const uploadForm = document.getElementById('upload-form');
const resultsSection = document.getElementById('results-section');
const predictedNameElement = document.getElementById('predicted-name');
const diseaseInfoElement = document.getElementById('disease-info');
const uploadButton = document.getElementById('upload-button');
const buttonText = document.getElementById('button-text');
const spinner = document.getElementById('spinner');

const modal = document.getElementById('results-modal');
const modalPredictedLabel = document.getElementById('modal-predicted-label');
const modalPredictedName = document.getElementById('modal-predicted-name');
const modalDiseaseInfo = document.getElementById('modal-disease-info');
const closeButton = document.querySelector('.close-button');
const modalUploadedImage = document.getElementById('modal-uploaded-image');

        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            // Show spinner and hide button text
            buttonText.style.display = 'none';
            spinner.style.display = 'block';
            uploadButton.disabled = true;

            const formData = new FormData(uploadForm);
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();

                    // Extract and display the relevant information
                    const { predicted_label, predicted_name, disease_info } = data;

                    // Set the predicted label and name separately
                    modalPredictedLabel.innerHTML = `<strong>Predicted Label:</strong> ${predicted_label}`;
                    modalPredictedName.innerHTML = `<strong>Predicted Name:</strong> <span>${predicted_name}</span>`;

                    // Set the disease info
                    modalDiseaseInfo.innerHTML = `
                        <p><strong>Disease Info:</strong> ${disease_info.description || 'N/A'}</p>
                    `;

                    // Show the modal
                    modal.style.display = 'block';
                } else {
                    const errorData = await response.json();
                    alert(`Error: ${errorData.error}`);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while uploading the file.');
            } finally {
                // Reset button state
                buttonText.style.display = 'block';
                spinner.style.display = 'none';
                uploadButton.disabled = false;
            }
        });

        // Close the modal when the close button is clicked
        closeButton.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        // Close the modal when clicking outside of it
        window.addEventListener('click', (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });

        const fileInput = document.getElementById('file-upload');
        const uploadBox = document.getElementById('upload-box');

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();

                reader.onload = (e) => {
                    // Set the uploaded image as the background of the upload-box
                    uploadBox.style.backgroundImage = `url('${e.target.result}')`;
                    uploadBox.classList.add('filled'); // Add a class to hide initial content

                    // Set the uploaded image in the modal
                    modalUploadedImage.src = e.target.result;
                };

                reader.readAsDataURL(file); // Read the file as a data URL
            }
        });

        // Add event listeners to example images
        const exampleImages = document.querySelectorAll('.example-box img');

        exampleImages.forEach((image) => {
            image.addEventListener('click', () => {
                const imageSrc = image.src;

                // Set the clicked example image as the uploaded image
                uploadBox.style.backgroundImage = `url('${imageSrc}')`;
                uploadBox.classList.add('filled'); // Add a class to hide initial content

                // Set the uploaded image in the modal
                modalUploadedImage.src = imageSrc;

                // Simulate form submission
                simulateUpload(imageSrc);
            });
        });

        // Function to simulate the upload process
        async function simulateUpload(imageSrc) {
            // Show spinner and hide button text
            buttonText.style.display = 'none';
            spinner.style.display = 'block';
            uploadButton.disabled = true;

            try {
                // Simulate a fetch request to the server
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: JSON.stringify({ image_url: imageSrc }), // Send the image URL as data
                    headers: {
                        'Content-Type': 'application/json',
                    },
                });

                if (response.ok) {
                    const data = await response.json();

                    // Extract and display the relevant information
                    const { predicted_label, predicted_name, disease_info } = data;

                    // Set the predicted label and name separately
                    modalPredictedLabel.innerHTML = `<strong>Predicted Label:</strong> ${predicted_label}`;
                    modalPredictedName.innerHTML = `<strong>Predicted Name:</strong> <span>${predicted_name}</span>`;

                    // Set the disease info
                    modalDiseaseInfo.innerHTML = `
                        <p><strong>Disease Info:</strong> ${disease_info.description || 'N/A'}</p>
                    `;

                    // Show the modal
                    modal.style.display = 'block';
                } else {
                    const errorData = await response.json();
                    alert(`Error: ${errorData.error}`);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while uploading the file.');
            } finally {
                // Reset button state
                buttonText.style.display = 'block';
                spinner.style.display = 'none';
                uploadButton.disabled = false;
            }
        }