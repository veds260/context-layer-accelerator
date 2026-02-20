// Dashboard logic
async function loadDashboard() {
  try {
    const res = await fetch('/api/dashboard');
    const data = await res.json();
    
    // Update stats
    document.getElementById('xp').textContent = data.xp;
    document.getElementById('streak').textContent = `🔥 ${data.streak}`;
    document.getElementById('progress').textContent = `${data.completedLessons}/${data.totalLessons}`;
    
    // Update next lesson
    const nextLessonEl = document.getElementById('nextLesson');
    if (data.nextLesson) {
      nextLessonEl.innerHTML = `
        <strong>Lesson ${data.nextLesson.level}:</strong> 
        ${data.nextLesson.emoji} ${data.nextLesson.title}
      `;
    } else {
      nextLessonEl.innerHTML = `
        <strong>🎉 All lessons complete!</strong><br>
        Time to build your context layer for real.
      `;
      document.body.classList.add('completed');
    }
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  }
}

loadDashboard();
