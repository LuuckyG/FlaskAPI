$(document).ready(function(){
    // Do something
});


function filterContent() {
    var selectedSection = document.getElementById('filter-results').value
    console.log("selection is:", selectedSection)
    var resultSections = document.getElementsByClassName('search-output-section')
    console.log(resultSections)
    
    for (var i = 0; i < resultSections.length; i++)
        {
            console.log("in for loop ",resultSections[i].id, selectedSection)
            if (resultSections[i].id == selectedSection) {
                resultSections[i].style.display = 'block';
            } else {
                resultSections[i].style.display = 'none';
            }
        }
}


function openDocument(form_id) {
    form = document.getElementById(form_id).submit();
    return false;
}



window.onload = function() {
    let filter = document.getElementById('filter-results');
    filter.addEventListener('change', filterContent())
}
