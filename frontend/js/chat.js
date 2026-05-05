/**
 * CuisineAI — Chat Controller
 * Handles message rendering, input events, image upload, and streaming display.
 */

// ─── State ───────────────────────────────────────────────

let pendingImages = [];           // File[] — images not yet sent
let streamingAbortController = null;
let isStreaming = false;

// ─── DOM References ──────────────────────────────────────

const chatMessages = document.getElementById('chat-messages');
const welcomeScreen = document.getElementById('welcome-screen');
const messageInput = document.getElementById('message-input');
const btnSend = document.getElementById('btn-send');
const btnStop = document.getElementById('btn-stop');
const btnAttach = document.getElementById('btn-attach');
const fileInput = document.getElementById('file-input');
const imagePreviewBar = document.getElementById('image-preview-bar');
const imagePreviewList = document.getElementById('image-preview-list');
const errorToast = document.getElementById('error-toast');
const dropOverlay = document.getElementById('drop-overlay');
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxClose = document.getElementById('lightbox-close');

// ─── Image Upload ────────────────────────────────────────

btnAttach.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    addImagesToPreview(fileInput.files);
    fileInput.value = '';
  }
});

function addImagesToPreview(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith('image/')) continue;
    if (file.size > 10 * 1024 * 1024) {
      showError(`Image "${file.name}" is too large (max 10MB)`);
      continue;
    }
    pendingImages.push(file);

    const item = document.createElement('div');
    item.className = 'preview-item';
    item.dataset.fileName = file.name;

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    item.appendChild(img);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'preview-remove';
    removeBtn.innerHTML = '&times;';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      pendingImages = pendingImages.filter((f) => f.name !== file.name);
      item.remove();
      if (pendingImages.length === 0) imagePreviewBar.classList.add('hidden');
    });
    item.appendChild(removeBtn);

    imagePreviewList.appendChild(item);
  }
  imagePreviewBar.classList.toggle('hidden', pendingImages.length === 0);
}

// ─── Drag & Drop ─────────────────────────────────────────

let dragCounter = 0;

document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  if (dragCounter === 1) dropOverlay.classList.remove('hidden');
});

document.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter === 0) dropOverlay.classList.add('hidden');
});

document.addEventListener('dragover', (e) => {
  e.preventDefault();
});

document.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.add('hidden');
  if (e.dataTransfer.files.length) {
    addImagesToPreview(e.dataTransfer.files);
  }
});

// ─── Clipboard Paste ─────────────────────────────────────

document.addEventListener('paste', (e) => {
  if (document.activeElement !== messageInput && document.activeElement !== document.body) return;
  const files = e.clipboardData?.files;
  if (files && files.length) {
    const images = Array.from(files).filter((f) => f.type.startsWith('image/'));
    if (images.length) {
      e.preventDefault();
      addImagesToPreview(images);
    }
  }
});

// ─── Message Rendering ───────────────────────────────────

function renderUserMessage(text, imagesBase64) {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'message-row user';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  // Text
  if (text) {
    const p = document.createElement('p');
    p.textContent = text;
    bubble.appendChild(p);
  }

  // Images
  if (imagesBase64 && imagesBase64.length) {
    for (const b64 of imagesBase64) {
      const img = document.createElement('img');
      img.src = b64;
      img.addEventListener('click', () => openLightbox(b64));
      bubble.appendChild(img);
    }
  }

  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
}

function createAgentBubble() {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'message-row agent';

  const avatar = document.createElement('div');
  avatar.className = 'agent-avatar';
  avatar.textContent = '👨‍🍳';
  row.appendChild(avatar);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble streaming-cursor';
  row.appendChild(bubble);

  chatMessages.appendChild(row);
  return { row, bubble };
}

function showStatus(text) {
  hideWelcome();
  // Remove any existing status
  const existing = document.querySelector('.status-bubble');
  if (existing) existing.remove();

  const el = document.createElement('div');
  el.className = 'status-bubble';
  el.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(text)}</span>`;
  chatMessages.appendChild(el);
  scrollToBottom();
  return el;
}

function renderTypingIndicator() {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'message-row agent';

  const avatar = document.createElement('div');
  avatar.className = 'agent-avatar';
  avatar.textContent = '👨‍🍳';
  row.appendChild(avatar);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.innerHTML = '<div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
  row.appendChild(bubble);

  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

// ─── Markdown Rendering ──────────────────────────────────

function renderMarkdown(container, rawText) {
  if (typeof marked !== 'undefined') {
    // Configure marked for safe rendering
    marked.setOptions({ breaks: true, gfm: true });
    container.innerHTML = marked.parse(rawText);
  } else {
    // Fallback: plain text
    container.textContent = rawText;
    container.style.whiteSpace = 'pre-wrap';
  }

  // Enhance recipe-specific patterns
  enhanceRecipeOutput(container);
}

/**
 * Post-process rendered markdown to add recipe card styling:
 * - Bold number patterns get score badges
 * - Improves visual hierarchy for recipe lists
 */
function enhanceRecipeOutput(container) {
  // Wrap score-like patterns in spans
  // Matches patterns like "营养价值：8分" or "**8分**" or "评分：8/10"
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    const parent = node.parentElement;
    if (parent.tagName === 'CODE' || parent.tagName === 'PRE') continue;

    let html = node.textContent;

    // Highlight scores like "X分" or "X/10"
    html = html.replace(/(\d+(?:\.\d+)?)\s*[分分数]\b/g, (match, score) => {
      const n = parseFloat(score);
      const cls = n >= 8 ? 'score-high' : n >= 6 ? 'score-mid' : 'score-low';
      return `<span class="recipe-score ${cls}">${match}</span>`;
    });

    if (html !== node.textContent) {
      const span = document.createElement('span');
      span.innerHTML = html;
      node.replaceWith(span);
    }
  }

  // Make images inside agent messages clickable for lightbox
  container.querySelectorAll('img').forEach((img) => {
    img.addEventListener('click', () => openLightbox(img.src));
  });
}

// ─── Streaming ───────────────────────────────────────────

async function sendMessage() {
  if (isStreaming) return;

  const text = messageInput.value.trim();
  if (!text && pendingImages.length === 0) return;

  // Encode images first
  let imagesBase64 = [];
  if (pendingImages.length) {
    try {
      imagesBase64 = await Promise.all(
        pendingImages.map((f) => encodeImageToBase64(f))
      );
    } catch (err) {
      showError(err.message);
      return;
    }
  }

  // Render user message
  renderUserMessage(text || '', imagesBase64);

  // Clear input
  messageInput.value = '';
  pendingImages = [];
  imagePreviewList.innerHTML = '';
  imagePreviewBar.classList.add('hidden');
  autoResizeTextarea();

  // Build input
  const input = buildMessageInput(text, imagesBase64);

  // Start streaming
  setStreaming(true);
  const typingRow = renderTypingIndicator();

  let streamingEl = null;
  let streamingContent = '';
  let statusEl = null;

  try {
    const threadId = getThreadId();

    for await (const part of streamRun(threadId, input)) {
      switch (part.event) {
        case 'metadata':
          // Extract node name for status
          if (part.data?.langgraph_node) {
            const nodeNames = {
              model: 'Thinking...',
              tools: 'Searching for recipes...',
            };
            const label = nodeNames[part.data.langgraph_node] ||
              `${part.data.langgraph_node}...`;
            if (statusEl) statusEl.remove();
            statusEl = showStatus(label);
          }
          break;

        case 'messages/partial':
          // Remove typing indicator on first content
          if (typingRow.parentNode) typingRow.remove();
          if (statusEl) { statusEl.remove(); statusEl = null; }

          for (const msg of part.data) {
            if (msg.content) {
              streamingContent += msg.content;
              if (!streamingEl) {
                const result = createAgentBubble();
                streamingEl = { row: result.row, bubble: result.bubble };
              }
              renderMarkdown(streamingEl.bubble, streamingContent);
              streamingEl.bubble.classList.add('streaming-cursor');
            }
          }
          break;

        case 'messages/complete':
          if (typingRow.parentNode) typingRow.remove();
          if (statusEl) { statusEl.remove(); statusEl = null; }

          for (const msg of part.data) {
            if (msg.content) {
              streamingContent = msg.content;
              if (!streamingEl) {
                const result = createAgentBubble();
                streamingEl = { row: result.row, bubble: result.bubble };
              }
              renderMarkdown(streamingEl.bubble, streamingContent);
              streamingEl.bubble.classList.remove('streaming-cursor');
            }
          }
          streamingEl = null;
          streamingContent = '';
          break;

        case 'values':
          // Full state update — can be used to sync messages but we trust streaming
          break;

        case 'end':
          break;

        case 'error':
          showError(part.data?.message || 'An error occurred during generation');
          break;
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // User cancelled
      if (streamingEl) {
        streamingEl.bubble.classList.remove('streaming-cursor');
        if (streamingContent) {
          renderMarkdown(streamingEl.bubble, streamingContent + '\n\n*[Generation stopped]*');
        }
      }
    } else if (err instanceof ApiError) {
      showError(err.message);
    } else {
      showError(err.message || 'Connection failed. Is the server running?');
    }
  } finally {
    if (typingRow.parentNode) typingRow.remove();
    if (streamingEl) streamingEl.bubble.classList.remove('streaming-cursor');
    if (statusEl) statusEl.remove();
    setStreaming(false);
    streamingEl = null;
    streamingContent = '';
    messageInput.focus();
  }
}

// ─── Streaming State ─────────────────────────────────────

function setStreaming(active) {
  isStreaming = active;
  btnSend.classList.toggle('hidden', active);
  btnStop.classList.toggle('hidden', !active);
  messageInput.disabled = active;
  btnAttach.disabled = active;
}

btnStop.addEventListener('click', () => {
  // Reloading the page is the simplest way to abort a streaming fetch
  // since AbortController cancellation of ReadableStream is not universally supported.
  // We preserve the thread in localStorage so the user can continue.
  if (isStreaming) {
    // Force the fetch to abort by navigating
    window.stop();
    setStreaming(false);
    messageInput.focus();
  }
});

// ─── Message Input ───────────────────────────────────────

messageInput.addEventListener('input', autoResizeTextarea);

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function autoResizeTextarea() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// ─── Send Button ─────────────────────────────────────────

btnSend.addEventListener('click', sendMessage);

// ─── Welcome Hints ───────────────────────────────────────

document.addEventListener('click', (e) => {
  const chip = e.target.closest('.hint-chip');
  if (chip) {
    messageInput.value = chip.dataset.hint;
    sendMessage();
  }
});

// ─── Lightbox ────────────────────────────────────────────

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.remove('hidden');
}

lightbox.addEventListener('click', (e) => {
  if (e.target === lightbox || e.target === lightboxClose) {
    lightbox.classList.add('hidden');
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) {
    lightbox.classList.add('hidden');
  }
});

// ─── Utilities ───────────────────────────────────────────

function hideWelcome() {
  if (welcomeScreen) welcomeScreen.remove();
}

function scrollToBottom() {
  const container = document.getElementById('chat-container');
  container.scrollTop = container.scrollHeight;
}

function showError(message, duration = 5000) {
  errorToast.textContent = message;
  errorToast.classList.remove('hidden');
  clearTimeout(errorToast._timeout);
  errorToast._timeout = setTimeout(() => {
    errorToast.classList.add('hidden');
  }, duration);
}

errorToast.addEventListener('click', () => {
  errorToast.classList.add('hidden');
});

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
