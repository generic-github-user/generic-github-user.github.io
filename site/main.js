
// Array shuffling function from https://stackoverflow.com/a/12646864
function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

function animate(e, delay) {
      let content = e.text();
      function step(c, i) {
            e.text(c);
            c += content[i];
            if (i < content.length) {
                  setTimeout(() => step(c, i+1), delay);
            }
      }
      step("", 0);
}

$(function(){
      $("#header").load("./header.html", function() {
            $(".animate").each(function () {
                  animate($(this), 100);
            });
      });
});
