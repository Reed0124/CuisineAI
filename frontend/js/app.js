/**
 * CuisineAI — Application Entry Point
 * Initializes the app, manages threads, and wires up top-level events.
 */

let currentThreadId = null;

// ─── Thread Management ───────────────────────────────────

function getThreadId() {
  return currentThreadId;
}

async function initThread() {
  const saved = localStorage.getItem('cuisineai_thread_id');

  if (saved) {
    try {
      // Verify the thread still exists
      const state = await getThreadState(saved);
      currentThreadId = saved;
      updateThreadDisplay();
      await restoreMessages(state);
      return;
    } catch (err) {
      // Thread expired or server restarted — create a new one
      console.warn('Saved thread unavailable, creating new one:', err.message);
      localStorage.removeItem('cuisineai_thread_id');
    }
  }

  await newThread();
}

async function newThread() {
  try {
    currentThreadId = await createThread();
    localStorage.setItem('cuisineai_thread_id', currentThreadId);
    updateThreadDisplay();
  } catch (err) {
    showError('Failed to connect to server. Is LangGraph running on port 2024?');
    throw err;
  }
}

function updateThreadDisplay() {
  const el = document.getElementById('thread-id-display');
  if (currentThreadId) {
    el.textContent = currentThreadId.slice(0, 8) + '...';
    el.title = currentThreadId;
  }
}

// ─── Message Restoration ─────────────────────────────────

function restoreMessages(state) {
  const messages = state?.values?.messages || [];
  if (messages.length === 0) return;

  for (const msg of messages) {
    // Skip system and tool messages in the chat display
    if (msg.type === 'system') continue;

    if (msg.type === 'human') {
      restoreHumanMessage(msg);
    } else if (msg.type === 'ai') {
      restoreAIMessage(msg);
    }
    // Skip ToolMessage — they're internal
  }
}

function restoreHumanMessage(msg) {
  let text = '';
  let images = [];

  if (typeof msg.content === 'string') {
    text = msg.content;
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block.type === 'text') {
        text += block.text;
      } else if (block.type === 'image_url') {
        images.push(block.image_url?.url || block.url || '');
      }
    }
  }

  const row = document.createElement('div');
  row.className = 'message-row user';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  if (text) {
    const p = document.createElement('p');
    p.textContent = text;
    bubble.appendChild(p);
  }

  for (const src of images) {
    if (src) {
      const img = document.createElement('img');
      img.src = src;
      img.addEventListener('click', () => openLightbox(src));
      bubble.appendChild(img);
    }
  }

  row.appendChild(bubble);
  chatMessages.appendChild(row);
}

function restoreAIMessage(msg) {
  const content = typeof msg.content === 'string'
    ? msg.content
    : (Array.isArray(msg.content)
      ? msg.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
      : '');

  if (!content) return;

  const row = document.createElement('div');
  row.className = 'message-row agent';

  const avatar = document.createElement('div');
  avatar.className = 'agent-avatar';
  avatar.textContent = '👨‍🍳';
  row.appendChild(avatar);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  renderMarkdown(bubble, content);

  row.appendChild(bubble);
  chatMessages.appendChild(row);
}

// ─── New Chat ────────────────────────────────────────────

document.getElementById('btn-new-chat').addEventListener('click', async () => {
  const container = document.querySelector('#chat-messages');
  // Remove all message rows, keep welcome screen
  container.querySelectorAll('.message-row, .status-bubble').forEach((el) => el.remove());

  // Restore welcome screen if it doesn't exist
  if (!document.getElementById('welcome-screen')) {
    const welcome = document.createElement('div');
    welcome.className = 'welcome-screen';
    welcome.id = 'welcome-screen';
    welcome.innerHTML = `
      <div class="welcome-icon">🥗</div>
      <h2>Welcome to CuisineAI</h2>
      <p>Send me photos of your ingredients or tell me what you have — I'll suggest the best recipes for you!</p>
      <div class="welcome-hints">
        <button class="hint-chip" data-hint="I have tomatoes, eggs, and green onions. What can I make?">🍅 Tomatoes + Eggs + Onions</button>
        <button class="hint-chip" data-hint="I have chicken breast, broccoli, and garlic. Suggest some healthy recipes.">🥦 Chicken + Broccoli</button>
        <button class="hint-chip" data-hint="What can I cook with tofu, mushrooms, and bell peppers?">🍄 Tofu + Mushrooms</button>
        <button class="hint-chip" data-hint="I have salmon, lemon, and asparagus. What recipes do you recommend?">🐟 Salmon + Asparagus</button>
      </div>
    `;
    container.insertBefore(welcome, container.firstChild);
  }

  await newThread();
  messageInput.focus();
});

// ─── Boot ────────────────────────────────────────────────

async function boot() {
  try {
    await initThread();
  } catch (err) {
    console.error('Boot failed:', err);
    // Show welcome screen anyway — user can type and we'll retry on send
  }
}

boot();
