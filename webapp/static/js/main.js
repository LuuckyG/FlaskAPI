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


// Log user into sharepoint, if credentials
function checkSharepointCredentials() {
    $.get('/check_sharepoint_credentials', function(user_data) {
        // If credentials
        if (user_data) { 
            $.getJSON('/sharepoint_login');
        } else { 
            $('#loginModal').modal('show'); 
        }
    });
};


// Close modal after filling in credentials
$(function closeModalOnSubmit() {
    $('#modalSubmitBtn').on('click', function() {
        $('#loginModal').modal('hide');
    });
});


// Automatic user logout functionality when inactive
$(function checkCurrentUser() {
    // If user, start timing (non-) activity
    $.get('/check_current_user', function(user) {
        if (user) { activityWatcher() }
    });
});


function activityWatcher() {
    var warningTime = (9 * 60 * 1000); // 9 minutes
    var maxInactivity = (10 * 60 * 1000); // 10 minutes

    var warningTimer;
    var logoutTimer;

    function warningMessage() {
        alert('Without any activity you are going to get logged out in 60 seconds!')
    }

    function logoutUser() { location.href = '/logout' }
    
    function startTimers() {
        warningTimer = setTimeout(warningMessage, warningTime);
        logoutTimer = setTimeout(logoutUser, maxInactivity);
    }
    
    function resetTimers() {
        clearTimeout(warningTimer);
        clearTimeout(logoutTimer);
        startTimers();
    }

    function activity() { resetTimers(); }

    var activityEvents = [
        'mousedown', 'mouseclick', 'mousemove',
        'keydown', 'scroll', 'touchstart'
    ];

    activityEvents.forEach(function(eventName) {
        document.addEventListener(eventName, activity, true);
    });

}
