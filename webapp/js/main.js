$(function filterContent() {
    var dropDownSelection = document.getElementById('output-selector').value
    var resultSections = document.getElementsByClassName('search-output-section')

    for (var i = 0; i < resultSections.length; i++)
        {
            if (resultSections[i].id == "dropDownSelection") {
                resultSections[i].style.display == 'relative';
            } else {
                resultSections[i].style.display == 'none';
            }
        }
})