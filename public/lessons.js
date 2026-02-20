// Badges for each lesson
const BADGES = {
  'vector-embeddings': '🗺️',
  'semantic-search': '🔮',
  'clustering-patterns': '🕵️',
  'recommendation-systems': '🎯',
  'rag-basics': '✂️',
  'mcp-integration': '🔌'
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
    renderLessons();
    renderWhatsNext();
  } catch (err) {
    console.error('Failed to init:', err);
  }
}

function updateStats() {
  document.getElementById('xp').textContent = progress.xp;
  document.getElementById('streak').textContent = `🔥 ${progress.streak}`;
}

function renderWhatsNext() {
  const container = document.getElementById('whatsNext');
  const nextLesson = lessons.find(l => !progress.completedLessons.includes(l.id));
  
  if (nextLesson) {
    container.innerHTML = `
      <h3>📍 What's Next</h3>
      <div class="lesson-preview">
        <strong>Lesson ${nextLesson.level}:</strong> ${nextLesson.emoji} ${nextLesson.title}
      </div>
      <button class="btn btn-primary" onclick="openLesson('${nextLesson.id}')">
        Start Lesson →
      </button>
    `;
  } else {
    container.innerHTML = `
      <h3>🎉 All Lessons Complete!</h3>
      <p>You've mastered the foundations. Time to build your context layer!</p>
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
    
    const card = document.createElement('div');
    card.className = `lesson-card ${isCompleted ? 'completed' : ''} ${isCurrent ? 'current' : ''} ${isLocked ? 'locked' : ''}`;
    
    let status = isLocked ? '🔒' : (isCompleted ? '✅' : '▶️');
    
    card.innerHTML = `
      <div class="lesson-emoji">${lesson.emoji}</div>
      <div class="lesson-info">
        <h3>Lesson ${lesson.level}: ${lesson.title}</h3>
        <p>${lesson.subtitle}</p>
      </div>
      <div class="lesson-status">${status}</div>
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
      <div class="emoji">${lesson.emoji}</div>
      <h1>Lesson ${lesson.level}: ${lesson.title}</h1>
      <p class="subtitle">${lesson.subtitle}</p>
    </div>
    
    <div class="lesson-section story">
      <h2>📖 The Story</h2>
      <div class="story-box">${lesson.story}</div>
    </div>
    
    <div class="lesson-section">
      <p class="hook">"${lesson.hook}"</p>
    </div>
    
    <div class="lesson-section concept">
      <h2>🧠 The Concept</h2>
      <p class="concept-text">${lesson.concept}</p>
    </div>
    
    <div class="lesson-section analogy">
      <h2>💡 Analogy</h2>
      <p class="analogy-text">${lesson.analogy}</p>
    </div>
    
    <div class="lesson-section visual">
      <h2>📊 Visual</h2>
      <pre class="visual-diagram">${lesson.visual}</pre>
    </div>
    
    <div class="lesson-section interactive">
      <h2>💻 Interactive Examples</h2>
      ${lesson.interactive.map(ex => `
        <div class="code-example">
          <h4>${ex.title}</h4>
          <p class="description">${ex.description}</p>
          <pre><code>${escapeHtml(ex.code)}</code></pre>
          <p class="explanation">💡 ${ex.explanation}</p>
        </div>
      `).join('')}
    </div>
    
    <div class="lesson-section key-points">
      <h2>✅ Key Takeaways</h2>
      <ul>
        ${lesson.keyPoints.map(p => `<li>${p}</li>`).join('')}
      </ul>
    </div>
    
    <div class="lesson-section real-world">
      <h2>🌍 Real-World Applications</h2>
      <ul>
        ${lesson.realWorld.map(r => `<li>${r}</li>`).join('')}
      </ul>
    </div>
    
    <div class="lesson-section">
      <div class="easter-egg">
        <h3>🥚 Easter Egg</h3>
        <p>${lesson.easterEgg}</p>
      </div>
    </div>
    
    <div class="lesson-section challenge">
      <div class="challenge-box">
        <h3>🏗️ Build Challenge</h3>
        <p>${lesson.challenge.preview}</p>
        <p class="xp-reward">Reward: ${lesson.challenge.xp} XP</p>
      </div>
    </div>
    
    <button class="complete-btn" onclick="completeLesson()" ${isCompleted ? 'disabled' : ''}>
      ${isCompleted ? '✅ Completed' : '✨ Mark as Complete'}
    </button>
  `;
  
  document.getElementById('lessonModal').classList.remove('hidden');
}

function closeLesson() {
  document.getElementById('lessonModal').classList.add('hidden');
  currentLessonId = null;
}

async function completeLesson() {
  if (!currentLessonId) return;
  
  try {
    const res = await fetch(`/api/lessons/${currentLessonId}/complete`, {
      method: 'POST'
    });
    
    const data = await res.json();
    
    if (data.error) {
      alert(data.error);
      return;
    }
    
    if (data.alreadyCompleted) {
      closeLesson();
      return;
    }
    
    // Update local progress
    progress.xp = data.xp;
    progress.streak = data.streak;
    progress.completedLessons.push(currentLessonId);
    
    // Show completion modal
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
  updateStats();
  renderLessons();
  renderWhatsNext();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Close modals on escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeLesson();
    closeCompletion();
  }
});

init();
