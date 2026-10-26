document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Canvas and WebSocket
  const canvasManager = new DenoiseCanvasManager('denoiseCanvas');

  // DOM Elements
  const promptInput = document.getElementById('promptInput');
  const negativePromptInput = document.getElementById('negativePromptInput');
  const modelSelect = document.getElementById('modelSelect');
  const schedulerSelect = document.getElementById('schedulerSelect');
  const stepsRange = document.getElementById('stepsRange');
  const stepsValue = document.getElementById('stepsValue');
  const cfgRange = document.getElementById('cfgRange');
  const cfgValue = document.getElementById('cfgValue');
  const seedInput = document.getElementById('seedInput');
  const randomSeedBtn = document.getElementById('randomSeedBtn');
  const previewFreq = document.getElementById('previewFreq');
  const previewFreqValue = document.getElementById('previewFreqValue');
  const previewMode = document.getElementById('previewMode');

  const btnGenerate = document.getElementById('btnGenerate');
  const btnCancel = document.getElementById('btnCancel');
  const btnDownload = document.getElementById('btnDownload');
  const btnReset = document.getElementById('btnReset');

  const progressOverlay = document.getElementById('progressOverlay');
  const progressFill = document.getElementById('progressFill');
  const stepBadge = document.getElementById('stepBadge');
  const speedBadge = document.getElementById('speedBadge');
  const elapsedBadge = document.getElementById('elapsedBadge');
  const statusDescription = document.getElementById('statusDescription');

  const galleryGrid = document.getElementById('galleryGrid');
  const galleryCount = document.getElementById('galleryCount');
  const canvasSize = document.getElementById('canvasSize');
  const negativePromptToggle = document.getElementById('negativePromptToggle');
  const negativePromptWrap = document.getElementById('negativePromptWrap');

  let currentWidth = 512;
  let currentHeight = 512;
  let activeModelConfig = null;

  // Controls rail: show a fade when there is more to scroll to.
  const controlsScroll = document.querySelector('.controls-scroll');
  function updateRailOverflow() {
    const rail = controlsScroll.closest('.sidebar-controls');
    if (!rail) return;
    const hidden = controlsScroll.scrollHeight - controlsScroll.clientHeight;
    const atBottom = controlsScroll.scrollTop >= hidden - 4;
    rail.classList.toggle('is-scrollable', hidden > 4 && !atBottom);
  }
  updateRailOverflow();
  window.addEventListener('resize', updateRailOverflow);
  controlsScroll.addEventListener('scroll', updateRailOverflow, { passive: true });

  // 2. Initialize WebSocket Client
  const wsClient = new DiffusionWebSocketClient({
    onConnect: () => {
      console.log('Connected to progressive generation service');
    },
    onDisconnect: () => {
      console.log('Disconnected from service');
    },
    onMessage: (msg) => {
      handleServerMessage(msg);
    }
  });

  function handleServerMessage(msg) {
    if (msg.type === 'started') {
      setGeneratingState(true);
      statusDescription.textContent = msg.status || 'Starting diffusion...';
      stepBadge.textContent = 'Step 0';
      speedBadge.textContent = '-- it/s';
      elapsedBadge.textContent = '--';
      progressFill.style.width = '0%';
    } else if (msg.type === 'step_preview') {
      const { step, total_steps, progress, speed_it_per_sec, elapsed_sec, preview_image, status } = msg;
      stepBadge.textContent = `Step ${step} / ${total_steps}`;
      speedBadge.textContent = `${speed_it_per_sec} it/s`;
      elapsedBadge.textContent = `${elapsed_sec}s`;
      progressFill.style.width = `${progress}%`;
      statusDescription.textContent = status || `Denoising step ${step}`;

      if (preview_image) {
        canvasManager.renderFrame(preview_image, step, total_steps, status);
      }
    } else if (msg.type === 'complete') {
      setGeneratingState(false);
      progressFill.style.width = '100%';
      statusDescription.textContent = `Completed in ${msg.elapsed_sec}s, seed ${msg.seed}`;
      stepBadge.textContent = 'Done';
      speedBadge.textContent = `${msg.speed} it/s`;
      elapsedBadge.textContent = `${msg.elapsed_sec}s`;
      canvasManager.setFinalImage(msg.image_url);
      loadGallery();
    } else if (msg.type === 'cancelled') {
      setGeneratingState(false);
      statusDescription.textContent = 'Generation cancelled.';
      stepBadge.textContent = 'Cancelled';
      speedBadge.textContent = '-- it/s';
      elapsedBadge.textContent = '--';
    } else if (msg.type === 'error') {
      setGeneratingState(false);
      alert(`Generation Error: ${msg.message}`);
      statusDescription.textContent = 'Error occurred.';
    }
  }

  function setGeneratingState(generating) {
    if (generating) {
      btnGenerate.disabled = true;
      btnGenerate.innerHTML = `<span class="spinner"></span> Denoising...`;
      btnCancel.classList.add('visible');
      progressOverlay.style.opacity = '1';
    } else {
      btnGenerate.disabled = false;
      btnGenerate.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Generate image`;
      btnCancel.classList.remove('visible');
    }
  }

  // 3. User Controls Bindings
  // Sliders
  stepsRange.addEventListener('input', (e) => {
    stepsValue.textContent = e.target.value;
  });
  cfgRange.addEventListener('input', (e) => {
    cfgValue.textContent = Number(e.target.value).toFixed(1);
  });
  previewFreq.addEventListener('input', (e) => {
    previewFreqValue.textContent = e.target.value === '1' ? 'Every step' : `Every ${e.target.value} steps`;
  });

  // Random Seed
  randomSeedBtn.addEventListener('click', () => {
    seedInput.value = Math.floor(Math.random() * 10000000);
  });

  // Aspect Ratio buttons
  function updateCanvasSizeLabel() {
    if (canvasSize) canvasSize.textContent = `${currentWidth} × ${currentHeight}`;
  }

  function applyAspectFromDims(w, h) {
    currentWidth = w;
    currentHeight = h;
    document.querySelectorAll('.aspect-btn').forEach(b => {
      const match = parseInt(b.dataset.width) === w && parseInt(b.dataset.height) === h;
      b.classList.toggle('active', match);
    });
    updateCanvasSizeLabel();
  }

  document.querySelectorAll('.aspect-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.aspect-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentWidth = parseInt(btn.dataset.width);
      currentHeight = parseInt(btn.dataset.height);
      canvasManager.applyAspect(currentWidth, currentHeight);
      updateCanvasSizeLabel();
    });
  });

  // Negative prompt disclosure
  negativePromptToggle.addEventListener('click', () => {
    const collapsed = negativePromptToggle.getAttribute('aria-expanded') === 'false';
    negativePromptToggle.setAttribute('aria-expanded', String(collapsed));
    negativePromptWrap.classList.toggle('hidden', !collapsed);
    updateRailOverflow();
  });
  updateCanvasSizeLabel();

  // Prompt suggestions
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.dataset.prompt;
      if (promptInput.value.trim()) {
        promptInput.value += ', ' + text;
      } else {
        promptInput.value = text;
      }
    });
  });

  // Generate Button
  btnGenerate.addEventListener('click', () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      promptInput.focus();
      showToast('Describe an image first.');
      return;
    }

    const seed = seedInput.value ? parseInt(seedInput.value) : -1;
    canvasManager.startGeneration(currentWidth, currentHeight);

    wsClient.generate({
      prompt: prompt,
      negative_prompt: negativePromptInput.value.trim(),
      model_id: modelSelect.value,
      width: currentWidth,
      height: currentHeight,
      steps: parseInt(stepsRange.value),
      cfg_scale: parseFloat(cfgRange.value),
      seed: seed,
      scheduler: schedulerSelect.value,
      preview_frequency: parseInt(previewFreq.value),
      preview_mode: previewMode.value
    });
  });

  // Cancel Button
  btnCancel.addEventListener('click', () => {
    wsClient.cancel();
  });

  // Download Button
  btnDownload.addEventListener('click', () => {
    const seed = seedInput.value || 'seed';
    canvasManager.downloadCanvas(`diffusion_${seed}.png`);
  });

  // Reset Canvas to Idle Noise
  function resetCanvasToIdle() {
    canvasManager.resetToIdle();
    statusDescription.textContent = 'Initial Gaussian noise field';
    stepBadge.textContent = 'Idle';
    speedBadge.textContent = '-- it/s';
    elapsedBadge.textContent = '--';
    progressFill.style.width = '0%';
    progressOverlay.style.opacity = '0';
  }

  btnReset.addEventListener('click', resetCanvasToIdle);

  // Model Selection Change - Auto adjusts steps & cfg defaults
  modelSelect.addEventListener('change', () => {
    const selectedModel = modelSelect.value;
    if (selectedModel.includes('turbo')) {
      stepsRange.value = 4;
      stepsValue.textContent = '4';
      cfgRange.value = 1.0;
      cfgValue.textContent = '1.0';
    } else {
      stepsRange.value = 20;
      stepsValue.textContent = '20';
      cfgRange.value = 7.5;
      cfgValue.textContent = '7.5';
    }
  });

  // 4. Load Model List
  async function loadModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      modelSelect.innerHTML = '';
      for (const [key, val] of Object.entries(data.models)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${val.name}`;
        if (key === data.default) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    } catch (err) {
      console.warn('Failed to load models:', err);
    }
  }

  // Toast helper
  let toastTimer = null;
  function showToast(msg) {
    let toast = document.getElementById('galleryToast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'galleryToast';
      toast.className = 'gallery-toast';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2500);
  }

  // 5. Gallery Management
  async function loadGallery() {
    try {
      const res = await fetch('/api/gallery');
      const items = await res.json();
      galleryGrid.innerHTML = '';
      if (galleryCount) {
        galleryCount.textContent = items.length === 1 ? '1 photo' : `${items.length} photos`;
      }

      if (items.length === 0) {
        galleryGrid.innerHTML = '<div class="gallery-empty">Nothing developed yet.<br>Describe an image and generate it.</div>';
        return;
      }

      items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.title = `${item.prompt}\nSeed ${item.seed} · ${item.width}×${item.height} · ${item.steps} steps`;
        div.innerHTML = `<img src="${item.url}" alt="${item.prompt}">`;

        const frameLabel = document.createElement('span');
        frameLabel.className = 'frame-label';
        frameLabel.textContent = `#${item.seed}`;
        div.appendChild(frameLabel);

        // Delete button
        const delBtn = document.createElement('button');
        delBtn.className = 'btn-delete-img';
        delBtn.title = 'Delete image';
        delBtn.innerHTML = '&#x2715;';
        delBtn.addEventListener('click', async (e) => {
          e.stopPropagation(); // prevent loading image into canvas
          if (!confirm('Delete this image permanently?\n\nIt will be removed from the gallery, the canvas and the outputs folder.')) return;

          delBtn.disabled = true;
          try {
            const delRes = await fetch(`/api/gallery/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
            if (!delRes.ok) {
              showToast('Failed to delete image.');
              delBtn.disabled = false;
              return;
            }

            // Clear every surface that could still be showing the deleted photo.
            if (canvasManager.currentImageUrl === item.url) {
              resetCanvasToIdle();
            }

            div.remove();
            showToast('Image deleted.');

            // Re-sync with the server so the gallery matches the folder exactly.
            await loadGallery();
          } catch (err) {
            console.warn('Delete failed:', err);
            delBtn.disabled = false;
            showToast('Error deleting image.');
          }
        });
        div.appendChild(delBtn);

        // Click gallery item to reload prompt, seed and display on canvas
        div.addEventListener('click', () => {
          promptInput.value = item.prompt;
          if (item.negative_prompt) {
            negativePromptInput.value = item.negative_prompt;
            // Make sure the loaded constraint is actually visible.
            negativePromptToggle.setAttribute('aria-expanded', 'true');
            negativePromptWrap.classList.remove('hidden');
          }
          seedInput.value = item.seed;
          stepsRange.value = item.steps;
          stepsValue.textContent = item.steps;
          cfgRange.value = item.cfg_scale;
          cfgValue.textContent = item.cfg_scale.toFixed(1);
          // Restore the photo's real dimensions too, so the canvas shows it
          // un-squashed and the next Generate reuses the same aspect.
          if (item.width && item.height) applyAspectFromDims(item.width, item.height);
          canvasManager.setFinalImage(item.url);
          statusDescription.textContent = `Loaded from gallery — ${item.created_at.slice(0, 10)}`;
        });

        galleryGrid.appendChild(div);
      });
    } catch (err) {
      console.warn('Failed to load gallery:', err);
    }
  }

  // Keep other open tabs in sync: photos deleted in one tab disappear in the rest.
  window.addEventListener('focus', loadGallery);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) loadGallery();
  });

  // Initial Boot
  loadModels();
  loadGallery();
});
