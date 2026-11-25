/* courses/fme: a formula cannot be broken over two lines, so one that is wider than its table cell
   would be cut off at the edge of the cell.  After MathJax has set the page, every formula in a
   table cell of an FME page that is wider than its cell is set smaller until it fits (never below
   60 %; what is still too wide then can be scrolled inside its cell).  Runs again when the window
   changes its width. */
(function () {
    function fit() {
        var formulas = document.querySelectorAll('section[id^="transistor-"] td mjx-container');
        formulas.forEach(function (m) { m.style.removeProperty('zoom'); });
        formulas.forEach(function (m) {
            var cell = m.closest('td');
            if (!cell) { return; }
            var style = window.getComputedStyle(cell);
            var room = cell.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
            var need = m.getBoundingClientRect().width;
            if (room > 0 && need > room) {
                m.style.zoom = Math.max(0.6, room / need);
            }
        });
    }
    function whenSet(tries) {
        if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
            window.MathJax.startup.promise.then(fit);
        } else if (tries < 150) {
            window.setTimeout(function () { whenSet(tries + 1); }, 100);
        }
    }
    function start() {
        if (!document.querySelector('section[id^="transistor-"]')) { return; }
        whenSet(0);
        window.addEventListener('resize', fit);
        /* a tab that was hidden has no width until it is shown: fit its formulas when a tab is chosen */
        document.addEventListener('change', function (event) {
            if (event.target && event.target.closest && event.target.closest('.sd-tab-set')) {
                window.setTimeout(fit, 60);
            }
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
