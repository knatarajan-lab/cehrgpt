// Get references to React and Recharts from the global scope
const { useState, useEffect, Fragment } = React;
const {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell
} = Recharts;

// Define colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d',
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'];

function Dashboard() {
    const [stats, setStats] = useState(null);
    const [patients, setPatients] = useState(null);
    const [topDomainEvents, setTopDomainEvents] = useState({
        conditions: [],
        drugs: [],
        procedures: []
    });
    const [userQuery, setUserQuery] = useState(null);

    // Function to extract and process top events by domain
    const processTopEventsByDomain = (patients) => {
        if (!patients) return null;

        // Collect all events across all patients
        const allEvents = patients.flatMap(patient =>
            patient.visits.flatMap(visit => visit.events)
        );

        // Process top events for each domain
        const domainTopEvents = {
            conditions: [],
            drugs: [],
            procedures: []
        };

        // Helper function to get top events for a specific domain
        const getTopEventsForDomain = (domain) => {
            const domainEvents = allEvents.filter(event =>
                event.domain.toUpperCase() === domain
            );

            const eventCounts = domainEvents.reduce((acc, event) => {
                const key = event.code_label || event.code;
                acc[key] = (acc[key] || 0) + 1;
                return acc;
            }, {});

            return Object.entries(eventCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([label, count]) => ({ label, count }));
        };

        // Get top events for each domain
        domainTopEvents.conditions = getTopEventsForDomain('CONDITION');
        domainTopEvents.drugs = getTopEventsForDomain('DRUG');
        domainTopEvents.procedures = getTopEventsForDomain('PROCEDURE');

        return domainTopEvents;
    };

    // Function to fetch stats data
    const fetchStats = async () => {
        try {
            // Use global patient stats if available
            let statsData = window.patientStats;

            // If no global stats, attempt to fetch
            if (!statsData) {
                const taskId = window.taskId || '';
                const response = await fetch(`/api/patient-stats/${taskId}`);
                statsData = await response.json();
            }

            setStats(statsData);

            // If global patients exist, use them
            if (window.generatedPatients) {
                const patientsData = window.generatedPatients.map(patient => {
                    // Assuming you're using the schema from cehrgpt_patient_schema.py
                    return {
                        ...patient,
                        birth_datetime: new Date(patient.birth_datetime),
                        visits: patient.visits.map(visit => ({
                            ...visit,
                            visit_start_datetime: new Date(visit.visit_start_datetime),
                            visit_end_datetime: visit.visit_end_datetime ? new Date(visit.visit_end_datetime) : null,
                        }))
                    };
                });

                setPatients(patientsData);
                setTopDomainEvents(processTopEventsByDomain(patientsData));

                // Set user query if available
                if (window.userQuery) {
                    setUserQuery(window.userQuery);
                }
            }
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    };

    // Fetch stats on component mount
    useEffect(() => {
        fetchStats();
    }, []);

    // Make the fetch function available globally for the batch button
    window.updateDashboard = fetchStats;

    if (!stats) return <div>Loading dashboard...</div>;

    // Format data for charts
    const genderData = Object.entries(stats.demographics.gender).map(([name, value]) => ({
        name,
        value
    }));

    const raceData = Object.entries(stats.demographics.race).map(([name, value]) => ({
        name,
        value
    }));

    const ageData = Object.entries(stats.demographics.age_groups)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .map(([name, value]) => ({
            name,
            value
        }));

    const visitTypeData = Object.entries(stats.visits.types).map(([name, value]) => ({
        name,
        value
    }));

    // Render function for domain-specific top events
    const renderDomainEventsChart = (events, title, fillColor) => {
        // Only render if events exist
        if (!events || events.length === 0) return null;

        return (
            <div className="chart-container">
                <h3>{title}</h3>
                <div style={{ width: '100%', height: 400 }}>
                    <ResponsiveContainer>
                        <BarChart data={events}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="label"
                                angle={-45}
                                textAnchor="end"
                                interval={0}
                                height={100}
                            />
                            <YAxis />
                            <Tooltip
                                formatter={(value, name, props) => {
                                    const event = props.payload;
                                    return [
                                        value,
                                        event.label
                                    ];
                                }}
                            />
                            <Bar dataKey="count" fill={fillColor}>
                                {events.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={COLORS[index % COLORS.length]}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    };

    return (
        <div className="dashboard-container">
            <h2 className="chart-title">Patient Statistics Dashboard</h2>

            {/* User Query */}
            {userQuery && (
                <div className="alert alert-info mt-3">
                    <strong>Original Query:</strong> {userQuery}
                </div>
            )}

            {/* Batch Patient Details (if available) */}
            {patients && (
                <div className="patient-details mt-4">
                    <h3>Generated Patients</h3>
                    <div className="card">
                        <div className="card-body">
                            <p>Total Patients Generated: {patients.length}</p>
                            <button
                                className="btn btn-primary"
                                data-bs-toggle="modal"
                                data-bs-target="#patientsModal"
                            >
                                View Patient Details
                            </button>
                        </div>
                    </div>

                    {/* Modal for Patient Details */}
                    <div className="modal fade" id="patientsModal" tabIndex="-1">
                        <div className="modal-dialog modal-xl">
                            <div className="modal-content">
                                <div className="modal-header">
                                    <h5 className="modal-title">Patient Details</h5>
                                    <button type="button" className="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div className="modal-body">
                                    {patients.map((patient, index) => (
                                        <div key={index} className="card mb-2">
                                            <div className="card-header">Patient {index + 1}</div>
                                            <div className="card-body">
                                                <pre>{JSON.stringify(patient, null, 2)}</pre>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Top Events by Domain */}
            {topDomainEvents && (
                <React.Fragment>
                    {renderDomainEventsChart(topDomainEvents.conditions, 'Top 10 Conditions', '#8884d8')}
                    {renderDomainEventsChart(topDomainEvents.drugs, 'Top 10 Drugs', '#82ca9d')}
                    {renderDomainEventsChart(topDomainEvents.procedures, 'Top 10 Procedures', '#ffc658')}
                </React.Fragment>
            )}

            {/* Existing charts */}
            <div className="chart-container">
                <h3>Gender Distribution</h3>
                <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                        <PieChart>
                            <Pie
                                data={genderData}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                outerRadius={80}
                                label
                            >
                                {genderData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Age Distribution */}
            <div className="chart-container">
                <h3>Age Distribution</h3>
                <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                        <BarChart data={ageData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value" fill="#8884d8" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Race Distribution */}
            <div className="chart-container">
                <h3>Race Distribution</h3>
                <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                        <PieChart>
                            <Pie
                                data={raceData}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                outerRadius={80}
                                label
                            >
                                {raceData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Visit Types */}
            <div className="chart-container">
                <h3>Visit Types</h3>
                <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                        <BarChart data={visitTypeData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value" fill="#82ca9d" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

// Render the dashboard
ReactDOM.render(
    <Dashboard />,
    document.getElementById('dashboard-container')
);
