$(document).ready(function() {
    let patientData = {};

    // Load conversation history if available
    function loadConversationHistory() {
        $.ajax({
            url: '/conversation',
            type: 'GET',
            success: function(history) {
                // Clear existing messages
                if (history.length > 0) {
                    $('#chat-box').empty();
                }
                // Append each message from history
                history.forEach(msg => {
                    if (msg.is_patient_data) {
                        // Use formatPatientData for patient data
                        appendMessage(msg.role, formatPatientData(msg.content));
                        // Update patientData storage
                        if (msg.role === 'assistant') {
                            patientData = msg.content;
                        }
                    } else {
                        appendMessage(msg.role, msg.content);
                    }
                });
                // Scroll to bottom after loading history
                const chatBox = document.getElementById('chat-box');
                chatBox.scrollTop = chatBox.scrollHeight;
            },
            error: function(err) {
                console.error('Error loading conversation history:', err);
                // Add fallback welcome message if history loading fails
                appendMessage('assistant', 'Welcome! I can help you generate synthetic patient data. What kind of patient data would you like to create?');
            }
        });
    }

    // Load conversation history when page loads
    loadConversationHistory();

    // Auto-resize textarea
    const textarea = document.getElementById('message-input');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Handle form submission
    $('#chat-form').on('submit', function(e) {
        e.preventDefault();
        const message = $('#message-input').val().trim();
        if (message) {
            // Clear input
            $('#message-input').val('');
            textarea.style.height = 'auto';
            if (!message) {
                appendMessage('assistant', 'Please enter a query first.');
                return;
            }
            // Send to backend
            $.ajax({
                url: '/send',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({query: message}),
                success: function(data) {
                    if (data.visits) {
                        // For patient data, format it before displaying
                        appendMessage('user', message);
                        appendMessage('assistant', formatPatientData(data));
                        patientData = data;
                    } else {
                        // For regular messages
                        appendMessage('user', message);
                        appendMessage('assistant', data.message || data);
                    }
                },
                error: function() {
                    appendMessage('assistant', 'Sorry, there was an error processing your request.');
                }
            });
        }
    });

    // Download functionality
    $('#download-data').click(function() {
        if (Object.keys(patientData).length === 0) {
            appendMessage('assistant', 'No patient data available to download yet. Please generate some data first.');
            return;
        }
        downloadObjectAsJson(patientData, "patient_data");
    });

    // Update the batch button handler:
    $('#batch-data').click(function() {
        const message = $('#message-input').val().trim();
        if (!message) {
            appendMessage('assistant', 'Please enter a query first.');
            return;
        }

        // Clear the text area
        $('#message-input').val('');
        textarea.style.height = 'auto';

        // Add user's query to message window
        appendMessage('user', message);

        // Start batch generation
        $.ajax({
            url: '/batch',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({query: message}),
            success: function(data) {
                // Inform the user in the chat
                appendMessage('assistant', `${data.message}`);
            },
            error: function() {
                appendMessage('assistant', 'Sorry, there was an error starting the batch generation.');
            }
        });
    });

    function appendMessage(sender, content) {
        const messageDiv = $('<div>')
            .addClass(`message ${sender}-message`);

        const contentDiv = $('<div>')
            .addClass('message-content')
            .html(content);

        messageDiv.append(contentDiv);
        $('#chat-box').append(messageDiv);

        // Scroll up by a fixed amount
        const chatBox = document.getElementById('chat-box');
        const scrollAmount = 100;
        chatBox.scrollTop += scrollAmount;
    }
});
