class DiffusionWebSocketClient {
  constructor(options = {}) {
    this.options = options;
    this.ws = null;
    this.isConnected = false;
    this.reconnectTimer = null;
    this.connect();
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/generate`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnected = true;
        console.log('[WS] Connected to progressive diffusion stream.');
        if (this.options.onConnect) this.options.onConnect();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        console.warn('[WS] Stream closed. Attempting reconnect in 3s...');
        if (this.options.onDisconnect) this.options.onDisconnect();
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error('[WS] Stream error:', err);
      };
    } catch (e) {
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), 3000);
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    return false;
  }

  generate(params) {
    return this.send({
      action: 'generate',
      ...params
    });
  }

  cancel() {
    return this.send({ action: 'cancel' });
  }

  handleMessage(data) {
    if (this.options.onMessage) {
      this.options.onMessage(data);
    }
  }
}
