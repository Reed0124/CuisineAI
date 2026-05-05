/**
 * CuisineAI — API Layer
 * Communicates with the LangGraph Platform REST API.
 */

const API_BASE = 'http://localhost:2024';
const ASSISTANT_ID = 'main_agent';

// ─── Thread Management ───────────────────────────────────

async function createThread() {
  const res = await fetch(`${API_BASE}/threads`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  const data = await res.json();
  return data.thread_id;
}

async function getThreadState(threadId) {
  const res = await fetch(`${API_BASE}/threads/${threadId}/state`);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json();
}

// ─── Running the Agent ───────────────────────────────────

/**
 * Build the LangChain message input payload for the agent.
 *
 * Text-only messages use a simple string content.
 * Messages with images use the multimodal content array format
 * that Qwen 3.5-Omni (and most vision LLMs) expect.
 */
function buildMessageInput(text, imagesBase64) {
  if (!imagesBase64 || imagesBase64.length === 0) {
    return { messages: [{ type: 'human', content: text }] };
  }

  const content = [{ type: 'text', text }];

  for (const b64 of imagesBase64) {
    // Determine MIME type from the data URL prefix, default to jpeg
    content.push({
      type: 'image_url',
      image_url: { url: b64 },
    });
  }

  return { messages: [{ type: 'human', content }] };
}

/**
 * Build the request body for the LangGraph stream endpoint.
 */
function buildStreamBody(threadId, input) {
  return {
    assistant_id: ASSISTANT_ID,
    input: input.input,
    stream_mode: ['messages'],
    if_not_exists: 'create',
  };
}

/**
 * Async generator that yields parsed SSE events from the
 * LangGraph Platform /runs/stream endpoint (v1 format).
 *
 * Each yielded value is { event, data } where:
 *   - event: "metadata", "messages/partial", "messages/complete", "values", "end", "error"
 *   - data:  parsed JSON payload
 */
async function* streamRun(threadId, input) {
  const url = `${API_BASE}/threads/${threadId}/runs/stream`;
  const body = buildStreamBody(threadId, input);

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on double-newline (SSE event boundary)
    const events = buffer.split('\n\n');
    buffer = events.pop(); // keep incomplete event in buffer

    for (const raw of events) {
      const parsed = parseSSEEvent(raw);
      if (parsed) yield parsed;
    }
  }

  // Flush remaining buffer
  if (buffer.trim()) {
    const parsed = parseSSEEvent(buffer);
    if (parsed) yield parsed;
  }
}

/**
 * Parse a single SSE event string into { event, data }.
 */
function parseSSEEvent(raw) {
  const lines = raw.split('\n');
  let eventType = '';
  let dataStr = '';

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      dataStr += line.slice(6);
    }
  }

  if (!dataStr) return null;

  try {
    const data = JSON.parse(dataStr);
    return { event: eventType, data };
  } catch {
    // Non-JSON data (rare); return raw string
    return { event: eventType, data: dataStr };
  }
}

// ─── Image Encoding ──────────────────────────────────────

/**
 * Encode a File object to a base64 data URL.
 * Automatically resizes images larger than 2048px on the longest side.
 */
function encodeImageToBase64(file) {
  return new Promise((resolve, reject) => {
    // Check file size early
    if (file.size > 10 * 1024 * 1024) {
      reject(new Error(`Image "${file.name}" is too large (max 10MB)`));
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const { width, height } = img;
        const maxDim = 2048;

        if (width <= maxDim && height <= maxDim) {
          resolve(reader.result);
          return;
        }

        // Resize
        const ratio = Math.min(maxDim / width, maxDim / height);
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.85));
      };
      img.onerror = () => reject(new Error('Failed to decode image'));
      img.src = reader.result;
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

// ─── Error Class ─────────────────────────────────────────

class ApiError extends Error {
  constructor(status, body) {
    let message = `API error (${status})`;
    try {
      const parsed = JSON.parse(body);
      message = parsed.detail || parsed.message || parsed.error || message;
    } catch {
      message = body || message;
    }
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}
