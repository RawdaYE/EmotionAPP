let currentAudioFile = null;
let currentMultimodalAudioFile = null;
let audioEmbedding = null;
let textEmbedding = null;

// Tab switching
function switchTab(tabName, event) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Activate corresponding button
    if (event && event.target) {
        event.target.classList.add('active');
    } else {
        // Fallback: find button by text content
        document.querySelectorAll('.tab-button').forEach(btn => {
            if (btn.textContent.toLowerCase().includes(tabName.toLowerCase())) {
                btn.classList.add('active');
            }
        });
    }
}

// Audio file handling
function handleAudioFile(event) {
    const file = event.target.files[0];
    if (file) {
        currentAudioFile = file;
        const audioPreview = document.getElementById('audio-preview');
        audioPreview.src = URL.createObjectURL(file);
        audioPreview.style.display = 'block';
        document.getElementById('audio-predict-btn').disabled = false;
    }
}

function handleMultimodalAudioFile(event) {
    const file = event.target.files[0];
    if (file) {
        currentMultimodalAudioFile = file;
        const audioPreview = document.getElementById('multimodal-audio-preview');
        audioPreview.src = URL.createObjectURL(file);
        audioPreview.style.display = 'block';
        updateMultimodalButton();
    }
}

// Recording variables
let mediaRecorder = null;
let audioChunks = [];
let recognition = null;
let isRecording = false;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US'; // Default to English

    recognition.onresult = function (event) {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            }
        }

        if (finalTranscript) {
            const tempTextArea = document.getElementById('multimodal-text-input');
            const currentText = tempTextArea.value;
            // Append with space if needed
            tempTextArea.value = currentText + (currentText.length > 0 ? ' ' : '') + finalTranscript;
            updateMultimodalButton(); // Update button state
        }
    };

    recognition.onerror = function (event) {
        console.error('Speech recognition error', event.error);
    };
} else {
    console.warn("Web Speech API not supported in this browser.");
}

// WAV Encoding helpers
function writeUTFBytes(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function mergeBuffers(audioData) {
    let result = new Float32Array(audioData.length * audioData[0].length);
    let offset = 0;
    for (let i = 0; i < audioData.length; i++) {
        result.set(audioData[i], offset);
        offset += audioData[i].length;
    }
    return result;
}

function encodeWAV(samples, sampleRate) {
    let buffer = new ArrayBuffer(44 + samples.length * 2);
    let view = new DataView(buffer);

    /* RIFF identifier */
    writeUTFBytes(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 32 + samples.length * 2, true);
    /* RIFF type */
    writeUTFBytes(view, 8, 'WAVE');
    /* format chunk identifier */
    writeUTFBytes(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeUTFBytes(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);

    return view;
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

// Variables for AudioContext recording
let audioContext = null;
let scriptProcessor = null;
let mediaStreamSource = null;
let audioBuffers = [];

async function startRecording(type) {
    if (isRecording) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        audioBuffers = []; // Reset buffers

        scriptProcessor.onaudioprocess = function (event) {
            if (!isRecording) return;
            const inputBuffer = event.inputBuffer.getChannelData(0);
            // Clone the data
            audioBuffers.push(new Float32Array(inputBuffer));
        };

        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        isRecording = true;

        // UI Updates
        document.getElementById(`${type}-record-btn`).style.display = 'none';
        document.getElementById(`${type}-stop-btn`).style.display = 'inline-flex';
        document.getElementById(`${type}-recording-indicator`).style.display = 'flex';

        // Start Speech Recognition for multimodal
        if (type === 'multimodal' && recognition) {
            recognition.start();
        }

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Microphone access denied or not available.");
    }
}

function stopRecording(type) {
    if (!isRecording) return;

    if (scriptProcessor && audioContext) {
        scriptProcessor.disconnect();
        mediaStreamSource.disconnect();
        isRecording = false;

        // Merge buffers
        let mergedBuffers = mergeBuffers(audioBuffers);
        let dataview = encodeWAV(mergedBuffers, audioContext.sampleRate);
        let audioBlob = new Blob([dataview], { type: 'audio/wav' });
        let audioFile = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });

        // Handle as file upload
        const event = { target: { files: [audioFile] } };

        if (type === 'audio') {
            handleAudioFile(event);
        } else if (type === 'multimodal') {
            handleMultimodalAudioFile(event);
        }

        // Close context
        audioContext.close();
    }

    // UI Updates
    document.getElementById(`${type}-record-btn`).style.display = 'inline-flex';
    document.getElementById(`${type}-stop-btn`).style.display = 'none';
    document.getElementById(`${type}-recording-indicator`).style.display = 'none';

    // Stop Speech Recognition
    if (type === 'multimodal' && recognition) {
        recognition.stop();
    }
}

function updateMultimodalButton() {
    const hasAudio = currentMultimodalAudioFile !== null;
    const hasText = document.getElementById('multimodal-text-input').value.trim() !== '';
    document.getElementById('multimodal-predict-btn').disabled = !(hasAudio && hasText);
}

// Show loading overlay
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}
// Display results
function displayResult(containerId, result, type = 'single') {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.classList.add('show');

    if (type === 'single') {
        const emotion = result.emotion;
        const confidence = (result.confidence * 100).toFixed(1);

        container.innerHTML = `
            <div class="result-item">
                <h3>Predicted Emotion</h3>
                <div class="emotion-badge">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%">
                        ${confidence}% Confidence
                    </div>
                </div>
            </div>
        `;
    } else if (type === 'multimodal') {
        container.innerHTML = `
            <div class="result-item">
                <h3>🎵 Audio Prediction</h3>
                <div class="emotion-badge">${result.audio.emotion.charAt(0).toUpperCase() + result.audio.emotion.slice(1)}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(result.audio.confidence * 100).toFixed(1)}%">
                        ${(result.audio.confidence * 100).toFixed(1)}% Confidence
                    </div>
                </div>
            </div>
            <div class="result-item">
                <h3>📝 Text Prediction</h3>
                <div class="emotion-badge">${result.text.emotion.charAt(0).toUpperCase() + result.text.emotion.slice(1)}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(result.text.confidence * 100).toFixed(1)}%">
                        ${(result.text.confidence * 100).toFixed(1)}% Confidence
                    </div>
                </div>
            </div>
            <div class="result-item" style="border-left-color: #764ba2; background: linear-gradient(135deg, #f8f9ff 0%, #fff 100%);">
                <h3>🎯 Fusion Prediction (Combined)</h3>
                <div class="emotion-badge" style="background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);">
                    ${result.fusion.emotion.charAt(0).toUpperCase() + result.fusion.emotion.slice(1)}
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(result.fusion.confidence * 100).toFixed(1)}%; background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);">
                        ${(result.fusion.confidence * 100).toFixed(1)}% Confidence
                    </div>
                </div>
                <p style="margin-top: 10px; color: #666; font-style: italic;">
                    This prediction combines both audio and text features for improved accuracy.
                </p>
            </div>
        `;
    }
}

function displayError(containerId, message) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="error-message">${message}</div>`;
    container.classList.add('show');
}
// API calls
async function predictAudio() {
    if (!currentAudioFile) {
        displayError('audio-result', 'Please select an audio file first.');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('audio', currentAudioFile);

        console.log('Sending audio prediction request...');
        const response = await fetch('/api/predict/audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Response received:', data);

        if (data.success) {
            audioEmbedding = data.audio_embedding;
            displayResult('audio-result', data, 'single');
        } else {
            displayError('audio-result', data.error || 'Failed to predict emotion');
        }
    } catch (error) {
        console.error('Error predicting audio:', error);
        displayError('audio-result', `Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function predictText() {
    const textInput = document.getElementById('text-input');
    if (!textInput) {
        console.error('Text input element not found');
        return;
    }

    const text = textInput.value.trim();

    if (!text) {
        displayError('text-result', 'Please enter some text to analyze.');
        return;
    }

    showLoading();

    try {
        console.log('Sending text prediction request...', text);
        const response = await fetch('/api/predict/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Response received:', data);

        if (data.success) {
            textEmbedding = data.text_embedding;
            displayResult('text-result', data, 'single');
        } else {
            displayError('text-result', data.error || 'Failed to predict emotion');
        }
    } catch (error) {
        console.error('Error predicting text:', error);
        displayError('text-result', `Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}
async function predictMultimodal() {
    if (!currentMultimodalAudioFile) {
        displayError('multimodal-result', 'Please select an audio file.');
        return;
    }

    const textInput = document.getElementById('multimodal-text-input');
    if (!textInput) {
        console.error('Multimodal text input element not found');
        return;
    }

    const text = textInput.value.trim();
    if (!text) {
        displayError('multimodal-result', 'Please enter text corresponding to the audio.');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('audio', currentMultimodalAudioFile);
        formData.append('text', text);

        console.log('Sending multimodal prediction request...');
        const response = await fetch('/api/predict/multimodal', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Response received:', data);

        if (data.success) {
            displayResult('multimodal-result', data, 'multimodal');
        } else {
            displayError('multimodal-result', data.error || 'Failed to predict emotion');
        }
    } catch (error) {
        console.error('Error predicting multimodal:', error);
        displayError('multimodal-result', `Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    // Enable multimodal button when text changes
    const multimodalTextInput = document.getElementById('multimodal-text-input');
    if (multimodalTextInput) {
        multimodalTextInput.addEventListener('input', updateMultimodalButton);
    }

    // Add click handlers for tab buttons
    document.querySelectorAll('.tab-button').forEach((btn) => {
        btn.addEventListener('click', function (e) {
            const tabName = this.getAttribute('data-tab');
            if (tabName) {
                switchTab(tabName, e);
            }
        });
    });

    // Add drag and drop support for audio upload areas
    setupDragAndDrop('audio-upload-area', 'audio-file', handleAudioFile);
    setupDragAndDrop('multimodal-audio-upload-area', 'multimodal-audio-file', handleMultimodalAudioFile);

    console.log('Emotion Recognition App initialized');
});

// Drag and drop support
function setupDragAndDrop(uploadAreaId, fileInputId, handler) {
    const uploadArea = document.getElementById(uploadAreaId);
    const fileInput = document.getElementById(fileInputId);

    if (!uploadArea || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = '#764ba2';
            uploadArea.style.background = '#f0f2ff';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = '#f8f9ff';
        }, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            const event = { target: fileInput };
            handler(event);
        }
    }, false);
}
