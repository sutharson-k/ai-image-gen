class DenoiseCanvasManager {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.width = 512;
    this.height = 512;
    this.isGenerating = false;
    this.animationId = null;
    this.stepFrames = []; // stores {step, src, timestamp} for inspector scrubber
    this.currentImage = null;
    this.currentImageUrl = null; // gallery URL currently shown on the canvas

    this.initCanvas(512, 512);
    this.startIdleNoise();
  }

  initCanvas(width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
  }

  startIdleNoise() {
    // Generates subtle authentic visual TV-static/diffusion noise while idle
    if (this.currentImage || this.isGenerating) return;
    this.stopIdleNoise(); // never run two noise loops at once

    const noiseImageData = this.ctx.createImageData(this.width, this.height);
    const data = noiseImageData.data;

    const paint = () => {
      for (let i = 0; i < data.length; i += 4) {
        // Subtle grayscale noise
        const val = Math.floor(Math.random() * 90 + 20);
        data[i] = val;       // R
        data[i + 1] = val;   // G
        data[i + 2] = val;   // B
        data[i + 3] = 255;   // A
      }
      this.ctx.putImageData(noiseImageData, 0, 0);
    };

    // A static grain field for people who ask for reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      paint();
      return;
    }

    const renderNoise = () => {
      if (this.currentImage || this.isGenerating) return;
      paint();
      this.animationId = requestAnimationFrame(renderNoise);
    };

    renderNoise();
  }

  stopIdleNoise() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  startGeneration(width, height) {
    this.stopIdleNoise();
    this.isGenerating = true;
    this.initCanvas(width, height);
    this.stepFrames = [];
    this.currentImage = null;
    this.currentImageUrl = null;
    this.clearInspectorFilmstrip();
  }

  renderFrame(dataUrl, step, totalSteps, statusText) {
    const img = new Image();
    img.onload = () => {
      this.ctx.drawImage(img, 0, 0, this.width, this.height);
      this.currentImage = img;

      // Add to filmstrip history
      this.stepFrames.push({ step, totalSteps, src: dataUrl, status: statusText });
      this.addInspectorThumb(dataUrl, step);
    };
    img.src = dataUrl;
  }

  setFinalImage(imageUrl) {
    this.isGenerating = false;
    this.currentImageUrl = imageUrl;
    const img = new Image();
    img.onload = () => {
      this.ctx.drawImage(img, 0, 0, this.width, this.height);
      this.currentImage = img;
    };
    img.src = imageUrl;
  }

  /**
   * Forgets the image currently on screen: stops the idle noise loop, clears
   * the step filmstrip and starts fresh visual noise again.
   */
  resetToIdle() {
    this.stopIdleNoise();
    this.currentImage = null;
    this.currentImageUrl = null;
    this.isGenerating = false;
    this.stepFrames = [];
    this.clearInspectorFilmstrip();
    this.initCanvas(this.width, this.height);
    this.startIdleNoise();
  }

  clearInspectorFilmstrip() {
    const filmstrip = document.getElementById('stepFilmstrip');
    if (filmstrip) filmstrip.innerHTML = '';
    this.updateInspectorState();
  }

  updateInspectorState() {
    const filmstrip = document.getElementById('stepFilmstrip');
    const count = document.getElementById('stepCount');
    const empty = document.getElementById('filmstripEmpty');
    const frames = filmstrip ? filmstrip.querySelectorAll('.step-thumb').length : 0;

    if (count) count.textContent = `${frames} frame${frames === 1 ? '' : 's'}`;
    if (empty) empty.classList.toggle('hidden', frames > 0);
  }

  addInspectorThumb(dataUrl, step) {
    const filmstrip = document.getElementById('stepFilmstrip');
    if (!filmstrip) return;

    const thumb = document.createElement('div');
    thumb.className = 'step-thumb';
    thumb.innerHTML = `
      <img src="${dataUrl}" alt="Step ${step}">
      <span class="step-num">${step}</span>
    `;

    // Click on thumbnail to view that exact intermediate step on canvas
    thumb.addEventListener('click', () => {
      const img = new Image();
      img.onload = () => {
        this.ctx.drawImage(img, 0, 0, this.width, this.height);
      };
      img.src = dataUrl;

      document.querySelectorAll('.step-thumb').forEach(t => t.classList.remove('active'));
      thumb.classList.add('active');
    });

    filmstrip.appendChild(thumb);
    filmstrip.scrollLeft = filmstrip.scrollWidth;
    this.updateInspectorState();
  }

  downloadCanvas(filename = 'diffusion-generation.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = this.canvas.toDataURL('image/png');
    link.click();
  }
}
