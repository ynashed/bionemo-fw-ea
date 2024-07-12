class UniprotCirclePackViz {
  /**
   * Creates an instance of the Visualization class.
   * @param {string} anchor - The DOM element selector where the SVG will be appended.
   * @param {number} width - Width of the SVG element.
   * @param {number} height - Height of the SVG element.
   */
  constructor(anchor, width, height) {
    this.anchor = anchor;
    this.width = width || 500;
    this.height = height || this.width; // if no input, default to square
    this.padding = 7;
    this.highlightStrokeWidth = 4;
    this.resetZoomCase = false;
    this.highlightedNode;
    this.MARGIN = { LEFT: 20, RIGHT: 20, TOP: 5, BOTTOM: 5 };

    this.sort = (a, b) => d3.descending(a.value, b.value);

    this.color = d3
      .scaleLinear()
      .domain([0, 5])
      .range(["#76b900", "white"])
      .interpolate(d3.interpolateHcl);

    this.draw();
  }

  /**
   * Truncate the given text to a specified length, appending "..." if needed.
   *
   * @param {string} text - The text to truncate.
   * @returns {string} The truncated text.
   */
  static truncateText(text) {
    const maxTextLength = 25;
    if (typeof text !== "string") {
      return "";
    }

    if (text.length <= maxTextLength) {
      return text;
    }

    return `Sequence: ${text.slice(0, maxTextLength)}...`;
  }

  /**
   * Scaffold the basic SVG chart with specified dimensions and margins.
   */
  scaffoldChart() {
    this.svg = d3
      .select(this.anchor)
      .append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .append("g") // Adding a 'g' element to act as a container for all other elements.
      .attr("transform", `translate(${this.width / 2}, ${this.height / 2})`); // Centering the content.
  }

  /**
   * Setup the layout for circle packing.
   */
  layout() {
    // compute pack layout
    const packed = d3
      .pack()
      .size([
        this.width - this.MARGIN.LEFT - this.MARGIN.RIGHT,
        this.height - this.MARGIN.TOP - this.MARGIN.BOTTOM,
      ])
      .padding(this.padding);

    this.root = packed(
      d3
        .hierarchy(unirefClusterSample)
        .sum((d) => Math.max(0, d.size))
        .sort((a, b) => d3.descending(a.value, b.value))
    );

    this.view;
    this.focus = this.root;
  }

  /**
   * Draw the circles based on the layout.
   */
  drawCircles() {
    // don't add event handler to outer-most circle
    const innerNodes = this.root.descendants().slice(1);

    this.nodes = this.svg
      .append("g")
      .selectAll("circle")
      .data(innerNodes)
      .join("circle")
      .attr("class", (d) => `depth-${d.depth}`)
      .attr("fill", (d) => (d.children ? this.color(d.depth) : "white"))
      .attr("label", (d) => d.data.name)
      .attr("pointer-events", (d) => (!d.children ? "none" : null))
      .attr("parent", (d) => (d.parent ? d.parent.data.name : null))
      .on("mouseover", function () {
        d3.select(this).attr("stroke", "#000");
      })
      .on("mouseout", function () {
        d3.select(this).attr("stroke", null);
      });
  }

  /**
   * Highlight a series of circles in sequence, ultimately selecting one to represent "50 Clusters".
   *
   * @param {number} numCircles - The number of circles to sample during the animation
   * @param {number} duration - The total duration for the highlighting animation sequence.
   */
  highlight50CLusters(numCircles, duration) {
    const self = this;
    const depthNodes = this.svg.selectAll(`circle.depth-1`).nodes();
    if (depthNodes.length === 0) return;

    let intervalCount = 0;
    const intervalTime = duration / numCircles;
    let highlightedNode;

    const interval = setInterval(() => {
      this.clearSelections();

      if (intervalCount >= numCircles) {
        clearInterval(interval);
        const finalNode =
          depthNodes[Math.floor(Math.random() * depthNodes.length)];
        d3.select(finalNode)
          .attr("stroke", "#000")
          .attr("stroke-width", self.highlightStrokeWidth)
          .classed("selected-cluster-50", true);
        // Store the label of the highlighted node
        highlightedNode = finalNode.__data__.data.name;
        this.highlightedNode = highlightedNode;
      } else {
        const selectedNode =
          depthNodes[Math.floor(Math.random() * depthNodes.length)];
        d3.select(selectedNode)
          .attr("stroke", "#000")
          .attr("stroke-width", self.highlightStrokeWidth);
      }
      intervalCount++;
    }, intervalTime);
  }

  /**
   * Highlight a series of circles in sequence. These circles represent uniref 90 clusters,
   * within a given uniref50 cluser.
   *
   * @param {number} numCircles - The number of circles to sample during the animation
   * @param {number} duration - The total duration for the highlighting animation sequence.
   */
  highlight90Clusters(duration) {
    const self = this;
    const finalCircle = d3.select(`circle[label="${this.highlightedNode}"]`);
    this.highlightAndZoom(finalCircle);

    // Use the highlightedNode from the previous function
    const depthNodes = this.svg
      .selectAll(`circle[parent="${this.highlightedNode}"]`)
      .nodes();

    const labels = this.svg
      .selectAll(`text[parent="${this.highlightedNode}"]`)
      .nodes();

    const numCircles = depthNodes.length;

    if (numCircles === 0) return;

    let intervalCount = 0;
    const intervalTime = duration / numCircles;
    let randomIndex = Math.floor(Math.random() * numCircles);

    const interval = setInterval(() => {
      this.svg
        .selectAll("circle")
        .attr("stroke", null)
        .attr("stroke-width", null);

      if (intervalCount >= numCircles) {
        clearInterval(interval);
        // const randomIndex = Math.floor(Math.random() * numCircles);
        const finalNode = depthNodes[randomIndex];
        const finalLabel = labels[randomIndex];
        d3.select(finalNode)
          .attr("stroke", "#000")
          .attr("stroke-width", self.highlightStrokeWidth)
          .classed("selected-cluster-90", true);

        this.svg
          .selectAll(`text[parent="${this.highlightedNode}"]`)
          .transition()
          .duration(500)
          .style("display", "none");
        d3.select(finalLabel)
          .transition()
          .duration(500)
          .style("display", "inline");
        d3.select(finalLabel)
          .select(".uniprot-visual-sequence-label")
          .transition()
          .duration(500)
          .style("display", "inline");
      } else {
        randomIndex = Math.floor(Math.random() * numCircles);
        const selectedNode = depthNodes[randomIndex];
        d3.select(selectedNode)
          .attr("stroke", "#000")
          .attr("stroke-width", self.highlightStrokeWidth);
      }
      intervalCount++;
    }, intervalTime);
  }

  /**
   * Draw annotations and labels for the elements.
   */
  drawAnnotations() {
    const innerNodes = this.root.descendants();

    this.labels = this.svg
      .append("g")
      .attr("text-anchor", "middle")
      .selectAll("text")
      .data(innerNodes)
      .join("text")
      .style("font-size", (d) => (d.parent == this.root ? "10px" : "8.5px"))
      .style("stroke-width", (d) => (d.parent == this.root ? "4px" : "3px"))
      .attr("class", "uniprot-visual-circle-label")
      .style("fill-opacity", (d) => (d.parent === this.root ? 1 : 0))
      .style("display", (d) => (d.parent === this.root ? "inline" : "none"))
      .attr("parent", (d) => (d.parent ? d.parent.data.name : null))
      .text((d) => d.data.name);

    this.labels
      .append("tspan")
      .attr("class", "uniprot-visual-sequence-label")
      .attr("x", 0)
      .attr("dy", "1.2em")
      .style("display", "none")
      .text((d) =>
        d.parent === this.root
          ? ""
          : UniprotCirclePackViz.truncateText(d.data.sequence)
      );
  }

  /**
   * Add interaction capabilities such as zooming and other events.
   */
  addInteraction() {
    const self = this;
    this.svg.on("click", (event) => this.zoom(this.root));
    // add click event to nodes
    this.nodes.on("click", (event, d) => {
      self.resetZoomCase = false;
      self.focus !== d && (this.zoom(d), event.stopPropagation());
    });
    this.zoomTo([this.focus.x, this.focus.y, this.focus.r * 2]);
  }

  /**
   * The final method to execute drawing of the complete chart.
   */
  draw() {
    // set up svg & chart params
    this.scaffoldChart();
    // run pack layout
    this.layout();
    // draw nodes
    this.drawCircles();
    // draw labels
    this.drawAnnotations();
    // add zoom capabilities, hover, etc.
    this.addInteraction();
  }

  /**
   * Zoom in method
   */
  zoomTo(v) {
    const k = this.width / v[2];

    this.view = v;

    this.labels.attr(
      "transform",
      (d) => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`
    );
    this.nodes.attr(
      "transform",
      (d) => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`
    );
    this.nodes.attr("r", (d) => d.r * k);
  }

  /**
   * Transitions
   */
  zoom(d) {
    const self = this;

    this.focus = d;

    const transition = this.svg
      .transition()
      .duration(500)
      .tween("zoom", (d) => {
        const i = d3.interpolateZoom(this.view, [
          this.focus.x,
          this.focus.y,
          this.focus.r * 2,
        ]);
        return (t) => self.zoomTo(i(t));
      });

    if (self.resetZoomCase !== true) {
      this.labels.style("stroke", null);
    }
    // }

    // hide/show labels on zoom
    this.labels
      .filter(function (d) {
        return d.parent === self.focus || this.style.display === "inline";
      })
      .style("stroke-width", self.resetZoomCase !== true ? 0 : "4px")
      .style("stroke", "rgba(1,1,1,0)")
      .transition(transition)
      .style("fill-opacity", (d) => (d.parent === self.focus ? 1 : 0))
      .style("stroke-width", "4px")
      .on("start", function (d) {
        if (d.parent === self.focus) {
          this.style.display = "inline";
          this.style.stroke = null;
        }
      })
      .on("end", function (d) {
        if (d.parent !== self.focus) {
          this.style.display = "none";
          this.style.stroke = null;
        }
      });
  }

  /**
   * Clear all stroke and stroke-width attributes from all circles.
   */
  clearSelections() {
    this.svg
      .selectAll("circle")
      .attr("stroke", null)
      .attr("stroke-width", null);
  }

  /**
   * Highlight and zoom into a specific circle element.
   *
   * @param {object} circle - The circle element to highlight and zoom.
   */
  highlightAndZoom(circle) {
    this.resetZoomCase = false;
    this.zoom(circle.datum());
  }

  /**
   * Reset the zoom to the root element.
   */
  resetZoom() {
    this.zoom(this.root);
    d3.selectAll(".uniprot-visual-sequence-label").style("display", "none");
  }

  /**
   * Clear selections and reset the zoom.
   */
  resetAnimation() {
    this.clearSelections();
    this.resetZoom();
  }
}
