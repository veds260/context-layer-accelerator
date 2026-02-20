const express = require('express');
const fs = require('fs');
const path = require('path');

// ─── Catch any crash before server even starts ───────────────────────────────
process.on('uncaughtException', (err) => {
  console.error('[FATAL] uncaughtException:', err.message);
  console.error(err.stack);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] unhandledRejection:', reason);
  process.exit(1);
});

// ─── Startup diagnostics ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
console.log('[startup] NODE_VERSION:', process.version);
console.log('[startup] PORT:', PORT);
console.log('[startup] CWD:', process.cwd());
console.log('[startup] __dirname:', __dirname);

const DATA_DIR = path.join(__dirname, 'data');
const LESSONS_FILE = path.join(__dirname, 'lesson-content.json'); // root, NOT in data/ volume
const PROGRESS_FILE = path.join(DATA_DIR, 'progress.json');
const PUBLIC_DIR = path.join(__dirname, 'public');

// Verify critical files exist before starting
console.log('[startup] Checking paths...');
console.log('[startup]   data dir exists:', fs.existsSync(DATA_DIR));
console.log('[startup]   lessons file exists:', fs.existsSync(LESSONS_FILE));
console.log('[startup]   public dir exists:', fs.existsSync(PUBLIC_DIR));

if (!fs.existsSync(DATA_DIR)) {
  console.error('[FATAL] data/ directory missing');
  process.exit(1);
}
if (!fs.existsSync(LESSONS_FILE)) {
  console.error('[FATAL] data/lesson-content.json missing');
  process.exit(1);
}
if (!fs.existsSync(PUBLIC_DIR)) {
  console.error('[FATAL] public/ directory missing');
  process.exit(1);
}

// ─── Progress helpers ─────────────────────────────────────────────────────────
function getProgress() {
  try {
    return JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
  } catch {
    const initial = {
      xp: 0,
      streak: 0,
      lastActive: null,
      completedLessons: [],
      completedChallenges: []
    };
    try {
      fs.writeFileSync(PROGRESS_FILE, JSON.stringify(initial, null, 2));
    } catch (writeErr) {
      console.error('[warn] Could not write progress file:', writeErr.message);
    }
    return initial;
  }
}

function saveProgress(progress) {
  try {
    fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
  } catch (err) {
    console.error('[error] saveProgress failed:', err.message);
    throw err;
  }
}

// ─── App ──────────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.static(PUBLIC_DIR));

// Health check (Railway pings this)
app.get('/health', (req, res) => {
  res.json({ status: 'ok', port: PORT, uptime: process.uptime() });
});

// Get all lessons
app.get('/api/lessons', (req, res) => {
  try {
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    res.json(lessons);
  } catch (err) {
    console.error('[error] GET /api/lessons:', err.message);
    res.status(500).json({ error: 'Failed to load lessons' });
  }
});

// Get specific lesson
app.get('/api/lessons/:id', (req, res) => {
  try {
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    const lesson = lessons.find(l => l.id === req.params.id);
    if (!lesson) return res.status(404).json({ error: 'Lesson not found' });
    res.json(lesson);
  } catch (err) {
    console.error('[error] GET /api/lessons/:id:', err.message);
    res.status(500).json({ error: 'Failed to load lesson' });
  }
});

// Complete a lesson
app.post('/api/lessons/:id/complete', (req, res) => {
  try {
    const progress = getProgress();
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    const lesson = lessons.find(l => l.id === req.params.id);

    if (!lesson) return res.status(404).json({ error: 'Lesson not found' });

    if (progress.completedLessons.includes(req.params.id)) {
      return res.json({ xp: progress.xp, xpGained: 0, streak: progress.streak,
        completedLessons: progress.completedLessons.length, alreadyCompleted: true });
    }

    const lessonIndex = lessons.findIndex(l => l.id === req.params.id);
    if (lessonIndex > 0) {
      const prevLesson = lessons[lessonIndex - 1];
      if (!progress.completedLessons.includes(prevLesson.id)) {
        return res.status(400).json({ error: 'Must complete previous lesson first', required: prevLesson.id });
      }
    }

    const xpGained = 100;
    progress.xp += xpGained;
    progress.completedLessons.push(req.params.id);

    const today = new Date().toISOString().split('T')[0];
    if (progress.lastActive !== today) {
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      progress.streak = progress.lastActive === yesterday ? progress.streak + 1 : 1;
    }
    progress.lastActive = today;

    saveProgress(progress);
    res.json({ xp: progress.xp, xpGained, streak: progress.streak,
      completedLessons: progress.completedLessons.length });
  } catch (err) {
    console.error('[error] POST /api/lessons/:id/complete:', err.message);
    res.status(500).json({ error: 'Failed to complete lesson' });
  }
});

// Get progress
app.get('/api/progress', (req, res) => {
  try {
    res.json(getProgress());
  } catch (err) {
    console.error('[error] GET /api/progress:', err.message);
    res.status(500).json({ error: 'Failed to load progress' });
  }
});

// Get dashboard data
app.get('/api/dashboard', (req, res) => {
  try {
    const progress = getProgress();
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));

    let nextLesson = null;
    for (const lesson of lessons) {
      if (!progress.completedLessons.includes(lesson.id)) {
        nextLesson = { id: lesson.id, level: lesson.level, title: lesson.title, emoji: lesson.emoji };
        break;
      }
    }

    res.json({ xp: progress.xp, streak: progress.streak,
      completedLessons: progress.completedLessons.length,
      totalLessons: lessons.length, nextLesson });
  } catch (err) {
    console.error('[error] GET /api/dashboard:', err.message);
    res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

// Reset progress
app.post('/api/reset', (req, res) => {
  try {
    const initial = { xp: 0, streak: 0, lastActive: null, completedLessons: [], completedChallenges: [] };
    saveProgress(initial);
    res.json({ success: true });
  } catch (err) {
    console.error('[error] POST /api/reset:', err.message);
    res.status(500).json({ error: 'Failed to reset' });
  }
});

// ─── Global error handler ─────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('[error] unhandled express error:', err.message, err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

// ─── Start ────────────────────────────────────────────────────────────────────
const server = app.listen(PORT, '0.0.0.0', () => {
  const addr = server.address();
  console.log(`[startup] Server listening on ${addr.address}:${addr.port}`);
  console.log('[startup] Ready to accept connections');
});

server.on('error', (err) => {
  console.error('[FATAL] Server failed to start:', err.message);
  process.exit(1);
});
