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
                // Open the task status page in a new window/tab
                window.open(`/task/${data.task_id}`, '_blank');

                // Inform the user in the chat
                appendMessage('assistant', `Batch generation started. Task ID: ${data.task_id}`);
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
