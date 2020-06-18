$(document).ready(function(){
    // Do something
});


function filterContent() {
    var selectedSection = document.getElementById('filter-results').value;
    var resultSections = document.getElementsByClassName('search-output-section');
    
    for (var i = 0; i < resultSections.length; i++)
        {
            if (resultSections[i].id == selectedSection) {
                resultSections[i].style.display = 'block';
            } else {
                resultSections[i].style.display = 'none';
            }
        }
}


function openDocument(form_id) {
    form = document.getElementById(form_id).submit();
    location.reload();
    return false;
}


// Automatic user logout functionality when inactive
$(function checkCurrentUser() {
    // If user, start timing (non-) activity
    $.get('/check_current_user', function(user, status) {
        if (user) { activityWatcher() }
    });
});


function activityWatcher() {
    var timer;
    var secondsSinceLastActivity = 0;
    var maxInactivity = (60 * 10); // 10 minutes

    // Update activity tracker every 5 seconds
    setInterval(function() {
        secondsSinceLastActivity += 5;
        console.log(secondsSinceLastActivity + 'sec since last activity!');
        
        // Log user out
        if (secondsSinceLastActivity > maxInactivity) { location.href = '/logout'; }

    }, 5000);


    function activity() { secondsSinceLastActivity = 0; }

    var activityEvents = [
        'mousedown', 'mouseclick', 'keydown', 
        'scroll', 'touchstart'
    ];

    activityEvents.forEach(function(eventName) {
        document.addEventListener(eventName, activity, true);
    });

}
