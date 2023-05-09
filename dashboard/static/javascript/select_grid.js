function showImage(tabName) {
    var tabs = Array.from(document.getElementsByClassName("tab")).concat(Array.from(document.getElementsByClassName("tab-default")));
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].style.display = "none";
    }
    document.getElementById(tabName).style.display = "block";
}
