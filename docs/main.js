// Copy the BibTeX entry to the clipboard.
(function () {
  "use strict";
  var btn = document.getElementById("copy-bibtex");
  var src = document.getElementById("bibtex-text");
  if (!btn || !src) return;

  function done(ok) {
    btn.textContent = ok ? "Copied" : "Press Ctrl+C";
    window.setTimeout(function () {
      btn.textContent = "Copy";
    }, 1800);
  }

  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    document.body.removeChild(ta);
    done(ok);
  }

  btn.addEventListener("click", function () {
    var text = src.textContent;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        function () {
          done(true);
        },
        function () {
          fallback(text);
        }
      );
    } else {
      fallback(text);
    }
  });
})();
