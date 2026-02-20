# 🧠 Context Layer Accelerator

Build a personal context layer for AI — gamified, story-driven learning.

## What You'll Build

A system that:
1. **Saves anything** (tweets, articles, images) with one click
2. **Auto-organizes** using ML pattern detection
3. **Injects context** into any AI without filling the window
4. **Connects via MCP** to Claude and other AI platforms

## Quick Start

```bash
cd context-layer-accelerator
npm install
npm start
```

Open http://localhost:3001

## Learning Path

### Foundation Tier (Lessons 1-5)
| Lesson | Topic | You'll Learn |
|--------|-------|--------------|
| 1 | 🗺️ Vector Embeddings | Convert content to meaning-coordinates |
| 2 | 🔮 Semantic Search | Find content by meaning, not keywords |
| 3 | 🕵️ Clustering | Auto-discover themes in your saves |
| 4 | 🎯 Recommendations | Surface relevant content proactively |
| 5 | ✂️ RAG | Inject only relevant context into AI |

### Integration Tier (Lesson 6)
| Lesson | Topic | You'll Learn |
|--------|-------|--------------|
| 6 | 🔌 MCP Integration | Connect your context layer to Claude |

## Time Investment

**~10 hours total** at 10 hrs/week:
- 6 lessons × 30 min = 3 hours learning
- 6 challenges × 1 hour = 6 hours building
- 1 hour buffer

## Lesson Structure

Each lesson follows the same pattern:
1. **Story** — Anime-style narrative hook
2. **Hook** — One memorable question
3. **Concept** — Plain English explanation
4. **Analogy** — Real-world comparison
5. **Visual** — ASCII diagram
6. **Interactive** — Code examples with explanations
7. **Key Points** — 5 takeaways
8. **Real-World** — Production examples
9. **Easter Egg** — Surprising deep fact
10. **Challenge** — What you'll build

## Tech Stack (What You'll Use)

- **Embeddings**: OpenAI `text-embedding-3-small` 
- **Vector Store**: SQLite (small) → Pinecone/Chroma (scale)
- **ML**: scikit-learn for clustering
- **AI Integration**: MCP for Claude, OpenAI for RAG
- **Frontend**: Your existing Chrome extension

## End Goal

When complete, you'll have:
- ✅ Chrome extension that saves content with embeddings
- ✅ SQLite DB with vector search
- ✅ Auto-clustering that names its own categories
- ✅ Recommendation engine ("Based on your recent focus...")
- ✅ RAG endpoint for AI context injection
- ✅ MCP server for direct Claude integration

## Philosophy

This isn't about watching videos. It's about **building**.

Each lesson teaches one concept, then you implement it. By the end, you have a working product — not just knowledge.

## Credits

Built for Ved's context layer product vision.
Inspired by the learning-accelerator format.
