/**
 * Animation & instantiation logic for uniprot visualization.
 */
function kickoffUniprotVisual() {
  // scope globals into a function to fire on page load
  const visualAnchor = "#uniprot-circlepack-anchor";
  const width = document.querySelector(".container-xl .col").clientWidth * 0.5;
  const playButton = document.getElementById("uniprot-visual-play-button");
  const statusText = document.getElementById("uniprot-visual-status-text");
  const stepText = document.getElementById("uniprot-visual-step-text");
  const slider = document.getElementById("uniprot-visual-slider");
  const sentences = [
    "First, sample a random UniRef50 Cluster.",
    "Then, within this cluster, sample a random UniRef90 cluster.",
    "Finally, take this cluster's representative sequence and feed that as input to our model.",
  ];
  let currentSentenceIndex = 0;

  // instantiate chart
  const unirefViz = new UniprotCirclePackViz(visualAnchor, width, width);

  const duration = 11000;
  // animation slider
  playButton.addEventListener("click", function () {
    currentSentenceIndex = 0;
    unirefViz.resetZoomCase = true;
    unirefViz.resetZoom();
    playButton.disabled = true; // Disable the button at the start of the animation
    let currentValue = parseInt(slider.min); // Start at the minimum value
    const maxValue = parseInt(slider.max);
    const totalSteps = sentences.length;
    const stepTime = duration / (maxValue - currentValue);
    const animationDuration = (duration / sentences.length) * 0.85;
    const stepsPerSentence = Math.floor(maxValue / totalSteps); // How many steps before switching sentences

    stepText.innerHTML = `Step ${currentSentenceIndex + 1}: `;
    statusText.textContent = sentences[currentSentenceIndex];

    // highlighting animation code
    if (currentSentenceIndex === 0) {
      unirefViz.highlight50CLusters(20, animationDuration);
    }

    const interval = setInterval(() => {
      // increment slider and update
      currentValue += 1;
      slider.value = currentValue;
      // Check if it's time to update the sentence
      if (
        currentValue % stepsPerSentence === 0 &&
        currentSentenceIndex < totalSteps - 1
      ) {
        currentSentenceIndex++;
        stepText.innerHTML = `Step ${currentSentenceIndex + 1}: `;
        statusText.textContent = sentences[currentSentenceIndex];

        if (currentSentenceIndex === 1) {
          unirefViz.highlight90Clusters(animationDuration);
        }
      }

      if (currentValue >= maxValue) {
        clearInterval(interval);
        statusText.textContent = "Click Play to view the sampling process.";
        stepText.innerHTML = "";
        playButton.disabled = false;
        unirefViz.resetAnimation();
      }
    }, stepTime);
  });
}

// render visual after DOM fully loaded
document.addEventListener("DOMContentLoaded", function () {
  kickoffUniprotVisual();
});
