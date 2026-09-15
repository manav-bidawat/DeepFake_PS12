const AppState = {
      currentVoiceId: null,
      currentTextId: null,
      currentText: "",
      selectedMode: "clone",
      generationConfig: { speed: 1.0, temperature: 0.7 },
      currentJobId: null,
      jobs: [],
      health: null,
      activeTextTab: "type",
      isGenerating: false,
      voiceFileInfo: null,
      lastOutputs: []
    };

    const UI = {
      navButtons: document.querySelectorAll('.nav-btn'),
      views: {
        new: document.getElementById('view-new'),
        classifier: document.getElementById('view-classifier'),
        history: document.getElementById('view-history'),
        status: document.getElementById('view-status')
      },
      toasts: document.getElementById('toasts')
    };

    function showToast(type, message) {
      const el = document.createElement('div');
      el.className = `toast ${type}`;
      el.textContent = message;
      UI.toasts.appendChild(el);
      setTimeout(() => {
        el.remove();
      }, 4000);
    }

    async function apiFetch(url, options = {}) {
      try {
        const response = await fetch(url, options);
        const contentType = response.headers.get('content-type') || '';
        const payload = contentType.includes('application/json') ? await response.json() : await response.text();

        if (!response.ok) {
          const detail = payload && payload.detail ? payload.detail : `Request failed (${response.status})`;
          throw new Error(detail);
        }
        return payload;
      } catch (err) {
        showToast('error', err.message || 'Network error');
        throw err;
      }
    }

    function switchView(name) {
      Object.keys(UI.views).forEach((k) => UI.views[k].classList.toggle('active', k === name));
      UI.navButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.view === name));

      if (name === 'history') {
        HistoryManager.loadJobs();
      }
      if (name === 'classifier') {
        ClassifierManager.loadHistory();
      }
      if (name === 'status') {
        StatusManager.refresh();
      }
    }

    function formatSeconds(s) {
      if (!isFinite(s)) return '--';
      if (s < 60) return `${s.toFixed(1)}s`;
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return `${m}m ${sec}s`;
    }

    function setGeneratingState(isGenerating) {
      AppState.isGenerating = isGenerating;
      const btn = document.getElementById('generate-btn');
      const spinner = document.getElementById('generate-spinner');
      const label = document.getElementById('generate-label');

      btn.disabled = isGenerating || !GenerationManager.canGenerate();
      spinner.classList.toggle('show', isGenerating);
      label.textContent = isGenerating ? 'Generating...' : 'Generate Voice';

      document.querySelectorAll('input, textarea, button').forEach((el) => {
        if (el.id === 'generate-btn') return;
        el.disabled = isGenerating;
      });
    }

    const PlayerManager = {
      async drawWaveformFromUrl(url, canvas, color) {
        if (!url || !canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        try {
          const res = await fetch(url);
          const arr = await res.arrayBuffer();
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const decoded = await audioCtx.decodeAudioData(arr);
          const data = decoded.getChannelData(0);

          ctx.clearRect(0, 0, width, height);
          ctx.fillStyle = '#0f141f';
          ctx.fillRect(0, 0, width, height);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;

          const step = Math.ceil(data.length / width);
          const amp = height / 2;

          for (let x = 0; x < width; x++) {
            let min = 1.0;
            let max = -1.0;
            for (let i = 0; i < step; i++) {
              const datum = data[x * step + i] || 0;
              if (datum < min) min = datum;
              if (datum > max) max = datum;
            }
            ctx.beginPath();
            ctx.moveTo(x, (1 + min) * amp);
            ctx.lineTo(x, (1 + max) * amp);
            ctx.stroke();
          }
          audioCtx.close();
        } catch (e) {
          console.error('waveform decode failed', e);
        }
      }
    };

    const UploadManager = {
      init() {
        this.bindDrop(document.getElementById('voice-drop'), document.getElementById('voice-file-input'), this.uploadVoice.bind(this));
      },

      bindDrop(zone, input, onFile) {
        zone.addEventListener('click', () => input.click());
        input.addEventListener('change', () => {
          if (input.files[0]) onFile(input.files[0]);
        });

        ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => {
          e.preventDefault();
          zone.classList.add('drag');
        }));

        ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => {
          e.preventDefault();
          zone.classList.remove('drag');
        }));

        zone.addEventListener('drop', (e) => {
          const file = e.dataTransfer.files[0];
          if (file) onFile(file);
        });
      },

      async uploadVoice(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (!['wav', 'mp3'].includes(ext)) {
          showToast('error', 'Only WAV or MP3 files are allowed');
          return;
        }

        showToast('info', 'Uploading voice...');
        const form = new FormData();
        form.append('voice_file', file);

        const data = await apiFetch('/upload-voice', { method: 'POST', body: form });
        AppState.currentVoiceId = data.voice_id;
        AppState.voiceFileInfo = data;

        const meta = document.getElementById('voice-meta');
        const pills = document.getElementById('voice-pills');
        meta.textContent = `${data.filename}`;
        pills.innerHTML = '';
        pills.appendChild(makePill(`${data.duration_seconds}s`));
        pills.appendChild(makePill(`${data.sample_rate} Hz`));

        const audioEl = document.getElementById('voice-audio');
        audioEl.style.display = 'block';
        audioEl.src = URL.createObjectURL(file);

        const canvas = document.getElementById('voice-wave');
        canvas.style.display = 'block';
        PlayerManager.drawWaveformFromUrl(audioEl.src, canvas, '#6c63ff');

        document.getElementById('voice-check').classList.add('visible');
        GenerationManager.updateGenerateState();
        showToast('success', 'Voice uploaded successfully');
      }
    };

    const TextManager = {
      init() {
        const tabType = document.getElementById('tab-type');
        const tabUpload = document.getElementById('tab-upload');
        const panelType = document.getElementById('panel-type');
        const panelUpload = document.getElementById('panel-upload');
        const textInput = document.getElementById('text-input');

        tabType.addEventListener('click', () => {
          AppState.activeTextTab = 'type';
          tabType.classList.add('active');
          tabUpload.classList.remove('active');
          panelType.classList.add('active');
          panelUpload.classList.remove('active');
          this.syncTextState();
        });

        tabUpload.addEventListener('click', () => {
          AppState.activeTextTab = 'upload';
          tabUpload.classList.add('active');
          tabType.classList.remove('active');
          panelUpload.classList.add('active');
          panelType.classList.remove('active');
          this.syncTextState();
        });

        textInput.addEventListener('input', () => {
          AppState.currentText = textInput.value;
          document.getElementById('char-count').textContent = `${textInput.value.length} characters`;
          this.syncTextState();
        });

        UploadManager.bindDrop(document.getElementById('text-drop'), document.getElementById('text-file-input'), this.uploadText.bind(this));
      },

      async uploadText(file) {
        if (!file.name.toLowerCase().endsWith('.txt')) {
          showToast('error', 'Only .txt files are allowed');
          return;
        }

        showToast('info', 'Uploading text file...');
        const form = new FormData();
        form.append('text_file', file);

        const data = await apiFetch('/upload-text', { method: 'POST', body: form });
        AppState.currentTextId = data.text_id;
        AppState.currentText = data.content;

        const preview = data.content.slice(0, 300);
        const words = data.word_count;
        const previewText = data.content.length > 300 ? `${preview}... (${words} total words)` : preview;
        document.getElementById('text-preview').textContent = previewText;

        const pills = document.getElementById('text-pills');
        pills.innerHTML = '';
        pills.appendChild(makePill(`${words} words`));

        document.getElementById('text-check').classList.add('visible');
        this.syncTextState();
        showToast('success', 'Text file uploaded successfully');
      },

      syncTextState() {
        const hasText = this.hasText();
        document.getElementById('text-check').classList.toggle('visible', hasText);
        GenerationManager.updateGenerateState();
      },

      hasText() {
        if (AppState.activeTextTab === 'upload') {
          return Boolean(AppState.currentTextId);
        }
        return Boolean((AppState.currentText || '').trim());
      }
    };

    const GenerationManager = {
      init() {
        const toggle = document.getElementById('options-toggle');
        const body = document.getElementById('options-body');

        toggle.addEventListener('click', () => {
          const open = body.classList.toggle('open');
          toggle.textContent = open ? 'Hide options' : 'Show options';
        });

        document.querySelectorAll('.mode-card').forEach((card) => {
          card.addEventListener('click', () => {
            document.querySelectorAll('.mode-card').forEach((c) => c.classList.remove('active'));
            card.classList.add('active');
            AppState.selectedMode = card.dataset.mode;
          });
        });

        const speed = document.getElementById('speed');
        const temp = document.getElementById('temperature');
        speed.addEventListener('input', () => {
          AppState.generationConfig.speed = Number(speed.value);
          document.getElementById('speed-val').textContent = `${Number(speed.value).toFixed(1)}x`;
        });
        temp.addEventListener('input', () => {
          AppState.generationConfig.temperature = Number(temp.value);
          document.getElementById('temp-val').textContent = Number(temp.value).toFixed(2);
        });

        document.getElementById('generate-btn').addEventListener('click', this.generate.bind(this));
        this.updateGenerateState();
      },

      canGenerate() {
        return Boolean(AppState.health?.model_loaded) && Boolean(AppState.currentVoiceId) && TextManager.hasText();
      },

      updateGenerateState() {
        const btn = document.getElementById('generate-btn');
        const can = this.canGenerate();
        btn.disabled = AppState.isGenerating || !can;

        if (AppState.health && !AppState.health.model_loaded) {
          btn.title = 'Voice generation is not enabled on this deployment';
        } else if (!AppState.currentVoiceId && !TextManager.hasText()) {
          btn.title = 'Upload reference voice and provide text first';
        } else if (!AppState.currentVoiceId) {
          btn.title = 'Upload reference voice first';
        } else if (!TextManager.hasText()) {
          btn.title = 'Provide text first';
        } else {
          btn.title = 'Generate voice';
        }
      },

      async generate() {
        if (!this.canGenerate()) return;

        setGeneratingState(true);
        showToast('info', 'Processing...');

        const payload = {
          voice_id: AppState.currentVoiceId,
          text: AppState.activeTextTab === 'type' ? AppState.currentText : null,
          text_id: AppState.activeTextTab === 'upload' ? AppState.currentTextId : null,
          mode: AppState.selectedMode,
          generation_config: AppState.generationConfig
        };

        try {
          const data = await apiFetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          AppState.currentJobId = data.job_id;
          AppState.lastOutputs = data.outputs || [];
          this.renderResults(AppState.lastOutputs);
          showToast('success', 'Voice generated successfully');
        } finally {
          setGeneratingState(false);
          this.updateGenerateState();
        }
      },

      async renderResults(outputs) {
        const panel = document.getElementById('results-panel');
        const grid = document.getElementById('results-grid');

        panel.style.display = outputs.length ? 'block' : 'none';
        grid.innerHTML = '';
        grid.classList.toggle('two', outputs.length === 2);

        for (const out of outputs) {
          const card = document.createElement('div');
          card.className = 'result-card';
          const canvasId = `wave-${out.file_id}`;
          const sim = out.feature_similarity || null;
          let simHtml = '';
          if (sim && Array.isArray(sim.top_features) && sim.top_features.length) {
            const rows = sim.top_features.map((f) => {
              const similar = f.verdict === 'similar';
              return `
                <tr>
                  <td>${f.feature}</td>
                  <td>${Number(f.reference).toFixed(4)}</td>
                  <td>${Number(f.generated).toFixed(4)}</td>
                  <td>${Number(f.similarity * 100).toFixed(1)}%</td>
                  <td><span class="badge ${similar ? 'real' : 'fake'}">${similar ? 'Similar' : 'Dissimilar'}</span></td>
                </tr>
              `;
            }).join('');
            simHtml = `
              <div style="margin-top:12px;border-top:1px solid #2d3348;padding-top:10px;">
                <h5 style="margin:0 0 8px 0;">Top-5 Decision Tree Feature Similarity</h5>
                <div class="meta-row" style="margin:0 0 8px 0;">
                  <span class="pill">Overall: ${Number((sim.overall_similarity || 0) * 100).toFixed(1)}%</span>
                  <span class="pill">Similar: ${sim.similar_count || 0}/${sim.total_features || 0}</span>
                </div>
                <div style="overflow:auto;">
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Reference</th>
                        <th>Generated</th>
                        <th>Similarity</th>
                        <th>Verdict</th>
                      </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                  </table>
                </div>
              </div>
            `;
          }

          card.innerHTML = `
            <h4>${out.label}</h4>
            <audio controls preload="metadata" style="width:100%; margin-top:8px;" src="${out.stream_url}"></audio>
            <canvas id="${canvasId}" width="900" height="80"></canvas>
            <div class="meta-row">
              <span class="pill">${Number(out.duration_seconds || 0).toFixed(2)}s</span>
              <span class="pill">${Math.round((out.file_size_bytes || 0) / 1024)} KB</span>
            </div>
            <div class="result-actions">
              <a class="download-btn" href="${out.download_url}">Download</a>
              <span class="muted">${out.filename}</span>
            </div>
            ${simHtml}
          `;
          grid.appendChild(card);
          const canvas = card.querySelector(`#${CSS.escape(canvasId)}`);
          PlayerManager.drawWaveformFromUrl(out.stream_url, canvas, '#22c55e');
        }
      }
    };

    const HistoryManager = {
      init() {
        document.getElementById('clear-history').addEventListener('click', async () => {
          await apiFetch('/cleanup', { method: 'DELETE' });
          showToast('info', 'Cleanup complete');
          this.loadJobs();
        });
      },

      async loadJobs() {
        const jobs = await apiFetch('/jobs');
        AppState.jobs = jobs;

        const tbody = document.getElementById('jobs-body');
        tbody.innerHTML = '';

        jobs.forEach((job) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${new Date(job.created_at).toLocaleString()}</td>
            <td>${job.voice_filename || '-'}</td>
            <td>${(job.text_preview || '').slice(0, 50)}</td>
            <td>${job.output_count || 0}</td>
            <td>${job.status}</td>
          `;

          const details = document.createElement('tr');
          details.style.display = 'none';
          const td = document.createElement('td');
          td.colSpan = 5;

          if (Array.isArray(job.outputs) && job.outputs.length > 0) {
            td.innerHTML = job.outputs.map((o) => `
              <div style="padding:8px 0; border-top:1px solid #2d3348;">
                <strong>${o.label}</strong>
                <audio controls style="width:100%; margin-top:6px;" src="${o.stream_url}"></audio>
              </div>
            `).join('');
          } else {
            td.innerHTML = '<span class="muted">No outputs available for this job.</span>';
          }

          details.appendChild(td);
          tr.addEventListener('click', () => {
            details.style.display = details.style.display === 'none' ? 'table-row' : 'none';
          });

          tbody.appendChild(tr);
          tbody.appendChild(details);
        });
      }
    };

    const StatusManager = {
      async refresh() {
        try {
          const h = await apiFetch('/health');
          AppState.health = h;
          this.render(h);
        } catch (_) {
          this.renderDown();
        }
      },

      render(h) {
        const used = Number(h.gpu_memory_used_mb || 0);
        const total = Number(h.gpu_memory_total_mb || 0);
        const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;

        document.getElementById('status-model').textContent = h.model_loaded ? 'Loaded' : 'Not loaded';
        GenerationManager.updateGenerateState();
        document.getElementById('gpu-text').textContent = `${used.toFixed(2)} / ${total.toFixed(2)} MB`;
        document.getElementById('gpu-bar').style.width = `${pct.toFixed(1)}%`;
        document.getElementById('status-uptime').textContent = formatSeconds(Number(h.uptime_seconds || 0));

        const dot = document.getElementById('sidebar-dot');
        const status = document.getElementById('sidebar-status');
        const uptime = document.getElementById('sidebar-uptime');
        dot.classList.add('ok');
        status.textContent = 'Server online';
        uptime.textContent = `Uptime: ${formatSeconds(Number(h.uptime_seconds || 0))}`;
      },

      renderDown() {
        const dot = document.getElementById('sidebar-dot');
        dot.classList.remove('ok');
        document.getElementById('sidebar-status').textContent = 'Server unavailable';
      }
    };

    const ClassifierManager = {
      selectedFile: null,

      init() {
        UploadManager.bindDrop(
          document.getElementById('classify-drop'),
          document.getElementById('classify-file-input'),
          this.pickFile.bind(this)
        );
        document.getElementById('classify-btn').addEventListener('click', this.classify.bind(this));
      },

      pickFile(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (!['wav', 'mp3'].includes(ext)) {
          showToast('error', 'Only WAV or MP3 files are allowed');
          return;
        }

        this.selectedFile = file;
        document.getElementById('classify-meta').textContent = file.name;
        document.getElementById('classify-btn').disabled = false;

        const audioEl = document.getElementById('classify-audio');
        audioEl.style.display = 'block';
        audioEl.src = URL.createObjectURL(file);

        const canvas = document.getElementById('classify-wave');
        canvas.style.display = 'block';
        PlayerManager.drawWaveformFromUrl(audioEl.src, canvas, '#6c63ff');
      },

      async classify() {
        if (!this.selectedFile) return;
        const btn = document.getElementById('classify-btn');
        const lbl = document.getElementById('classify-btn-label');
        btn.disabled = true;
        lbl.textContent = 'Classifying...';

        try {
          const form = new FormData();
          form.append('voice_file', this.selectedFile);
          const data = await apiFetch('/classify', { method: 'POST', body: form });
          this.renderResult(data);
          showToast('success', 'Classification complete');
          await this.loadHistory();
        } finally {
          btn.disabled = false;
          lbl.textContent = 'Classify Voice';
        }
      },

      renderResult(data) {
        const card = document.getElementById('classify-result-card');
        const body = document.getElementById('classify-result-body');
        const a = data.results.model_a;
        const b = data.results.model_b;
        const aFake = a.label === 'fake';
        const bFake = b.label === 'fake';
        const aScore = safeNumber(a.spoof_score, 0);
        const bScore = safeNumber(b.spoof_score, 0);
        const aThr = safeNumber(a.threshold, 0.42);
        const bThr = safeNumber(b.threshold, 0.42);
        const aConf = safeNumber(a.confidence, 0);
        const bConf = safeNumber(b.confidence, 0);
        const elapsed = safeNumber(data.processing_time_seconds, 0);

        card.style.display = 'block';
        body.innerHTML = `
          <div class="results-grid two">
            <div class="result-card">
              <h4>Model A</h4>
              <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                <span class="badge ${aFake ? 'fake' : 'real'}">${aFake ? 'Fake / Spoof' : 'Real / Bonafide'}</span>
                <span class="pill">Score: ${aScore.toFixed(4)}</span>
                <span class="pill">Threshold: ${aThr.toFixed(3)}</span>
                <span class="pill">Confidence: ${(aConf * 100).toFixed(1)}%</span>
              </div>
            </div>
            <div class="result-card">
              <h4>Model B</h4>
              <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                <span class="badge ${bFake ? 'fake' : 'real'}">${bFake ? 'Fake / Spoof' : 'Real / Bonafide'}</span>
                <span class="pill">Score: ${bScore.toFixed(4)}</span>
                <span class="pill">Threshold: ${bThr.toFixed(3)}</span>
                <span class="pill">Confidence: ${(bConf * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <p class="muted" style="margin-top:10px;">Processed in ${elapsed.toFixed(3)}s</p>
        `;
      },

      async loadHistory() {
        const rows = await apiFetch('/classify-history');
        const tbody = document.getElementById('classify-history-body');
        tbody.innerHTML = '';

        rows.forEach((r) => {
          const tr = document.createElement('tr');
          const a = r.results?.model_a || {};
          const b = r.results?.model_b || {};
          const aFake = a.label === 'fake';
          const bFake = b.label === 'fake';
          const aScore = safeNumber(a.spoof_score, 0);
          const bScore = safeNumber(b.spoof_score, 0);
          const aConf = safeNumber(a.confidence, 0);
          const bConf = safeNumber(b.confidence, 0);
          tr.innerHTML = `
            <td>${new Date(r.created_at).toLocaleString()}</td>
            <td>${r.filename}</td>
            <td><span class="badge ${aFake ? 'fake' : 'real'}">${aFake ? 'Fake' : 'Bonafide'}</span></td>
            <td><span class="badge ${bFake ? 'fake' : 'real'}">${bFake ? 'Fake' : 'Bonafide'}</span></td>
            <td>${aScore.toFixed(4)} / ${(aConf * 100).toFixed(1)}%</td>
            <td>${bScore.toFixed(4)} / ${(bConf * 100).toFixed(1)}%</td>
          `;
          tbody.appendChild(tr);
        });
      },
    };

    function makePill(text) {
      const el = document.createElement('span');
      el.className = 'pill';
      el.textContent = text;
      return el;
    }

    function safeNumber(value, fallback = 0) {
      const n = Number(value);
      return Number.isFinite(n) ? n : fallback;
    }

    function initializeNavigation() {
      UI.navButtons.forEach((btn) => {
        btn.addEventListener('click', () => switchView(btn.dataset.view));
      });
    }

    document.addEventListener('DOMContentLoaded', () => {
      initializeNavigation();
      UploadManager.init();
      TextManager.init();
      GenerationManager.init();
      ClassifierManager.init();
      HistoryManager.init();
      StatusManager.refresh();

      setInterval(() => {
        StatusManager.refresh();
      }, 10000);

      if (window.lucide && window.lucide.createIcons) {
        window.lucide.createIcons();
      }
    });
