const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(express.json());
app.use(express.static('public'));

// Data paths
const DATA_DIR = path.join(__dirname, 'data');
const LESSONS_FILE = path.join(DATA_DIR, 'lesson-content.json');
const PROGRESS_FILE = path.join(DATA_DIR, 'progress.json');

// Initialize progress if needed
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
    fs.writeFileSync(PROGRESS_FILE, JSON.stringify(initial, null, 2));
    return initial;
  }
}

function saveProgress(progress) {
  fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
}

// Get all lessons
app.get('/api/lessons', (req, res) => {
  try {
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    res.json(lessons);
  } catch (err) {
    res.status(500).json({ error: 'Failed to load lessons' });
  }
});

// Get specific lesson
app.get('/api/lessons/:id', (req, res) => {
  try {
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    const lesson = lessons.find(l => l.id === req.params.id);
    if (!lesson) {
      return res.status(404).json({ error: 'Lesson not found' });
    }
    res.json(lesson);
  } catch (err) {
    res.status(500).json({ error: 'Failed to load lesson' });
  }
});

// Complete a lesson
app.post('/api/lessons/:id/complete', (req, res) => {
  try {
    const progress = getProgress();
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    const lesson = lessons.find(l => l.id === req.params.id);
    
    if (!lesson) {
      return res.status(404).json({ error: 'Lesson not found' });
    }
    
    // Check if already completed
    if (progress.completedLessons.includes(req.params.id)) {
      return res.json({
        xp: progress.xp,
        xpGained: 0,
        streak: progress.streak,
        completedLessons: progress.completedLessons.length,
        alreadyCompleted: true
      });
    }
    
    // Check prerequisites
    const lessonIndex = lessons.findIndex(l => l.id === req.params.id);
    if (lessonIndex > 0) {
      const prevLesson = lessons[lessonIndex - 1];
      if (!progress.completedLessons.includes(prevLesson.id)) {
        return res.status(400).json({ 
          error: 'Must complete previous lesson first',
          required: prevLesson.id
        });
      }
    }
    
    // Award XP and mark complete
    const xpGained = 100;
    progress.xp += xpGained;
    progress.completedLessons.push(req.params.id);
    
    // Update streak
    const today = new Date().toISOString().split('T')[0];
    if (progress.lastActive !== today) {
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      progress.streak = progress.lastActive === yesterday ? progress.streak + 1 : 1;
    }
    progress.lastActive = today;
    
    saveProgress(progress);
    
    res.json({
      xp: progress.xp,
      xpGained,
      streak: progress.streak,
      completedLessons: progress.completedLessons.length
    });
  } catch (err) {
    res.status(500).json({ error: 'Failed to complete lesson' });
  }
});

// Get progress
app.get('/api/progress', (req, res) => {
  res.json(getProgress());
});

// Get dashboard data
app.get('/api/dashboard', (req, res) => {
  try {
    const progress = getProgress();
    const lessons = JSON.parse(fs.readFileSync(LESSONS_FILE, 'utf8'));
    
    // Find next lesson
    let nextLesson = null;
    for (const lesson of lessons) {
      if (!progress.completedLessons.includes(lesson.id)) {
        nextLesson = {
          id: lesson.id,
          level: lesson.level,
          title: lesson.title,
          emoji: lesson.emoji
        };
        break;
      }
    }
    
    res.json({
      xp: progress.xp,
      streak: progress.streak,
      completedLessons: progress.completedLessons.length,
      totalLessons: lessons.length,
      nextLesson
    });
  } catch (err) {
    res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

// Reset progress
app.post('/api/reset', (req, res) => {
  const initial = {
    xp: 0,
    streak: 0,
    lastActive: null,
    completedLessons: [],
    completedChallenges: []
  };
  saveProgress(initial);
  res.json({ success: true });
});

app.listen(PORT, () => {
  console.log(`🚀 Context Layer Accelerator running at http://localhost:${PORT}`);
});
