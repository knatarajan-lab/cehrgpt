// Get references to React and Recharts from the global scope
const { useState, useEffect } = React;
const {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell
} = Recharts;

// Define colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

function Dashboard() {
    const [stats, setStats] = useState(null);
    const [patients, setPatients] = useState(null);

    // Function to fetch stats data
    const fetchStats = async () => {
        try {
            const response = await fetch('/api/patient-stats');
            const data = await response.json();
            setStats(data);

            // If global patients exist, use them
            if (window.generatedPatients) {
                setPatients(window.generatedPatients);
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

    return (
        <div className="dashboard-container">
            <h2 className="chart-title">Patient Statistics Dashboard</h2>

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

            {/* Gender Distribution */}
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
