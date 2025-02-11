$(document).ready(function() {
    let patientData = {};

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
            // Add user message
            appendMessage('user', message);
            $('#message-input').val('');
            textarea.style.height = 'auto';

            // Send to backend
            $.ajax({
                url: '/send',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({message: message}),
                success: function(data) {
                    if (data.visits) {
                        patientData = data;
                        appendMessage('assistant', formatPatientData(data));
                    } else {
                        appendMessage('assistant', data.message);
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

    function appendMessage(sender, content) {
        const messageDiv = $('<div>')
            .addClass(`message ${sender}-message`);

        const contentDiv = $('<div>')
            .addClass('message-content')
            .html(content);

        messageDiv.append(contentDiv);
        $('#chat-box').append(messageDiv);

        // Scroll to bottom
        const chatBox = document.getElementById('chat-box');
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
