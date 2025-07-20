// Utility functions
function isInpatient(visitConceptId) {
    return ["9201", "262", "8971", "8920"].includes(String(visitConceptId));
}

function dayDifference(date1, date2) {
    const timeDifference = Math.abs(date2.getTime() - date1.getTime());
    return Math.ceil(timeDifference / (1000 * 60 * 60 * 24));
}

function downloadObjectAsJson(exportObj, exportName) {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}
