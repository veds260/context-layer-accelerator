async function loadDashboard() {
  try {
    const res = await fetch('/api/dashboard');
    const data = await res.json();

    document.getElementById('xp').textContent = data.xp;
    document.getElementById('streak').textContent = `🔥 ${data.streak}`;
    document.getElementById('progress').textContent = `${data.completedLessons}/${data.totalLessons}`;
    document.getElementById('progressFraction').textContent = `${data.completedLessons} of ${data.totalLessons} lessons`;

    const pct = data.totalLessons > 0 ? (data.completedLessons / data.totalLessons) * 100 : 0;
    document.getElementById('progressBar').style.width = `${pct}%`;

    const nextCard = document.getElementById('nextCard');
    const nextLessonEl = document.getElementById('nextLesson');

    if (data.nextLesson) {
      nextLessonEl.textContent = `Lesson ${data.nextLesson.level}: ${data.nextLesson.emoji} ${data.nextLesson.title}`;
    } else {
      nextCard.classList.add('all-done');
      nextCard.querySelector('.next-card-label').textContent = 'All done';
      nextLessonEl.textContent = 'You finished all 6 lessons. Time to build your context layer for real.';
    }
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  }
}

loadDashboard();
