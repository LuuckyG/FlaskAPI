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



window.onload = function() {
    let filter = document.getElementById('filter-results');
    filter.addEventListener('change', filterContent())
}
