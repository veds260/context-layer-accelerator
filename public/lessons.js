const BADGES = {
  'vector-embeddings': '🗺️',
  'semantic-search': '🔮',
  'clustering-patterns': '🕵️',
  'recommendation-systems': '🎯',
  'rag-basics': '✂️',
  'mcp-integration': '🔌',
  'production-scale': '🏗️'
};

let lessons = [];
let progress = { completedLessons: [], xp: 0, streak: 0 };
let currentLessonId = null;

async function init() {
  try {
    const [lessonsRes, progressRes] = await Promise.all([
      fetch('/api/lessons'),
      fetch('/api/progress')
    ]);
    lessons = await lessonsRes.json();
    progress = await progressRes.json();
    updateStats();
    renderWhatsNext();
    renderLessons();
  } catch (err) {
    console.error('Failed to init:', err);
  }
}

function updateStats() {
  document.getElementById('xp').textContent = progress.xp;
  document.getElementById('streak').textContent = progress.streak;
}

function renderWhatsNext() {
  const container = document.getElementById('whatsNext');
  const next = lessons.find(l => !progress.completedLessons.includes(l.id));

  if (next) {
    container.innerHTML = `
      <div class="whats-next-banner">
        <div class="whats-next-label">Up next</div>
        <div class="whats-next-title">Lesson ${next.level}: ${next.emoji} ${next.title}</div>
        <button class="btn btn-primary" onclick="openLesson('${next.id}')">Open lesson →</button>
      </div>
    `;
  } else {
    container.innerHTML = `
      <div class="whats-next-banner all-done">
        <div class="whats-next-label">All done</div>
        <p class="whats-next-done-text">You finished all 7 lessons. Time to build the real thing.</p>
      </div>
    `;
  }
}

function renderLessons() {
  const container = document.getElementById('lessonList');
  container.innerHTML = '';

  lessons.forEach((lesson, index) => {
    const isCompleted = progress.completedLessons.includes(lesson.id);
    const isCurrent = !isCompleted && (index === 0 || progress.completedLessons.includes(lessons[index - 1].id));
    const isLocked = !isCompleted && !isCurrent;

    const statusIcon = isCompleted ? '✓' : (isCurrent ? '▶' : '🔒');

    const card = document.createElement('div');
    card.className = `lesson-card${isCompleted ? ' completed' : ''}${isCurrent ? ' current' : ''}${isLocked ? ' locked' : ''}`;
    card.innerHTML = `
      <div class="lesson-card-emoji">${lesson.emoji}</div>
      <div class="lesson-card-body">
        <div class="lesson-card-title">Lesson ${lesson.level}: ${lesson.title}</div>
        <div class="lesson-card-sub">${lesson.subtitle}</div>
      </div>
      <div class="lesson-card-status">${statusIcon}</div>
    `;

    if (!isLocked) {
      card.onclick = () => openLesson(lesson.id);
    }

    container.appendChild(card);
  });
}

async function openLesson(lessonId) {
  currentLessonId = lessonId;
  const lesson = lessons.find(l => l.id === lessonId);
  const isCompleted = progress.completedLessons.includes(lessonId);

  const content = document.getElementById('lessonContent');
  content.innerHTML = `
    <div class="lesson-header">
      <span class="lesson-header-emoji">${lesson.emoji}</span>
      <h1>Lesson ${lesson.level}: ${lesson.title}</h1>
      <p class="subtitle">${lesson.subtitle}</p>
    </div>

    <div class="content-section">
      <div class="content-section-title">The story</div>
      <div class="story-box">${lesson.story}</div>
    </div>

    <div class="content-section">
      <p class="hook-text">"${lesson.hook}"</p>
    </div>

    <div class="content-section">
      <div class="content-section-title">The concept</div>
      <p class="concept-text">${lesson.concept}</p>
    </div>

    <div class="content-section">
      <div class="content-section-title">Analogy</div>
      <p class="analogy-text">${lesson.analogy}</p>
    </div>

    <div class="content-section">
      <div class="content-section-title">Visual</div>
      <pre class="visual-pre">${lesson.visual}</pre>
    </div>

    <div class="content-section">
      <div class="content-section-title">Code examples</div>
      ${lesson.interactive.map(ex => `
        <div class="code-example">
          <div class="code-example-header">
            <h4>${ex.title}</h4>
            <div class="code-example-desc">${ex.description}</div>
          </div>
          <pre><code>${escapeHtml(ex.code)}</code></pre>
          <div class="code-example-note">${ex.explanation}</div>
        </div>
      `).join('')}
    </div>

    <div class="content-section">
      <div class="content-section-title">Key takeaways</div>
      <ul class="key-list">
        ${lesson.keyPoints.map(p => `<li>${p}</li>`).join('')}
      </ul>
    </div>

    <div class="content-section">
      <div class="content-section-title">Real-world applications</div>
      <ul class="key-list rw-list">
        ${lesson.realWorld.map(r => `<li>${r}</li>`).join('')}
      </ul>
    </div>

    <div class="content-section">
      <div class="easter-egg-box">
        <strong>🥚 Easter egg</strong>
        ${lesson.easterEgg}
      </div>
    </div>

    <div class="content-section">
      <div class="challenge-box">
        <h4>🏗️ Build challenge</h4>
        <p>${lesson.challenge.preview}</p>
        <div class="challenge-xp">Reward: ${lesson.challenge.xp} XP</div>
      </div>
    </div>

    ${lesson.bonusSection ? `
    <div class="content-section">
      <div class="content-section-title">${lesson.bonusSection.title}</div>
      <div class="bonus-table">
        ${lesson.bonusSection.content.map(row => `
          <div class="bonus-row">
            <div class="bonus-scenario">${row.scenario}</div>
            <div class="bonus-rec">${row.recommendation}</div>
          </div>
        `).join('')}
      </div>
    </div>
    ` : ''}

    <button class="complete-btn" onclick="completeLesson()" ${isCompleted ? 'disabled' : ''}>
      ${isCompleted ? '✓ Completed' : 'Mark as complete'}
    </button>
  `;

  document.getElementById('lessonModal').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeLesson() {
  document.getElementById('lessonModal').classList.add('hidden');
  document.body.style.overflow = '';
  currentLessonId = null;
}

async function completeLesson() {
  if (!currentLessonId) return;

  try {
    const res = await fetch(`/api/lessons/${currentLessonId}/complete`, { method: 'POST' });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    if (data.alreadyCompleted) {
      closeLesson();
      return;
    }

    progress.xp = data.xp;
    progress.streak = data.streak;
    progress.completedLessons.push(currentLessonId);

    document.getElementById('xpGained').textContent = data.xpGained;
    document.getElementById('badgeEarned').textContent = BADGES[currentLessonId] || '🏆';
    document.getElementById('lessonModal').classList.add('hidden');
    document.getElementById('completionModal').classList.remove('hidden');

  } catch (err) {
    console.error('Failed to complete lesson:', err);
    alert('Failed to save progress');
  }
}

function closeCompletion() {
  document.getElementById('completionModal').classList.add('hidden');
  document.body.style.overflow = '';
  updateStats();
  renderLessons();
  renderWhatsNext();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeLesson();
    closeCompletion();
  }
});

init();
