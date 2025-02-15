// Formatting functions
function formatPatientData(data) {
    let html = '<div class="patient-data">';

    // Demographics
    html += '<h5>Patient Demographics</h5>';
    html += `<p>Gender: ${data.gender}<br>`;
    html += `Race: ${data.race}</p>`;

    // Visits
    if (data.visits && data.visits.length > 0) {
        html += '<h5>Visits</h5>';
        data.visits.forEach(visit => {
            html += formatVisit(visit, new Date(data.birth_datetime));
        });
    }

    html += '</div>';
    return html;
}

function formatVisit(visit, birthDate) {
    let visitStartDate = new Date(visit.visit_start_datetime);
    let age = visitStartDate.getFullYear() - birthDate.getFullYear();
    let html = `<div class="visit-data">`;
    html += `<strong>${visit.visit_type}</strong> on ${visitStartDate.toLocaleDateString()} (Age: ${age})<br>`;

    if (visit.events) {
        if (isInpatient(visit.visit_concept_id)) {
            // Handle inpatient visits - group by date first
            let eventsByDate = {};
            visit.events.forEach(event => {
                let eventDate = new Date(event.time).toLocaleDateString();
                if (!eventsByDate[eventDate]) {
                    eventsByDate[eventDate] = {};
                }
                if (!eventsByDate[eventDate][event.domain]) {
                    eventsByDate[eventDate][event.domain] = [];
                }
                eventsByDate[eventDate][event.domain].push(event.code_label);
            });

            // Display events by date
            for (let [date, domains] of Object.entries(eventsByDate)) {
                let daysFromAdmission = dayDifference(new Date(date), visitStartDate);
                html += `<div style="margin-left: 1rem"><strong>Day ${daysFromAdmission}:</strong>`;
                for (let [domain, concepts] of Object.entries(domains).sort()) {
                    html += `<div style="margin-left: 1rem"><strong>${domain}:</strong><ul>`;
                    concepts.forEach(concept => {
                        html += `<li>${concept}</li>`;
                    });
                    html += '</ul></div>';
                }
                html += '</div>';
            }
        } else {
            // Handle outpatient visits - group by domain only
            let eventsByDomain = {};
            visit.events.forEach(event => {
                if (!eventsByDomain[event.domain]) {
                    eventsByDomain[event.domain] = [];
                }
                eventsByDomain[event.domain].push(event.code_label);
            });

            for (let [domain, concepts] of Object.entries(eventsByDomain).sort()) {
                html += `<div style="margin-left: 1rem"><strong>${domain}:</strong><ul>`;
                concepts.forEach(concept => {
                    html += `<li>${concept}</li>`;
                });
                html += '</ul></div>';
            }
        }
    }

    html += '</div>';
    return html;
}
