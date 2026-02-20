const express = require('express');
const fs = require('fs');
const path = require('path');

process.on('uncaughtException', (err) => {
  console.error('[FATAL] uncaughtException:', err.message);
  console.error(err.stack);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] unhandledRejection:', reason);
  process.exit(1);
});

const PORT = process.env.PORT || 3001;
console.log('[startup] NODE_VERSION:', process.version);
console.log('[startup] PORT:', PORT);

// Lessons hardcoded in the image — immune to volume mounts
const LESSONS = [
  {
    "id": "vector-embeddings",
    "level": 1,
    "title": "The Meaning Mapper",
    "subtitle": "Turn any content into coordinates in meaning-space",
    "emoji": "🗺️",
    "story": "Imagine you're building a massive library, but instead of organizing books by author or title, you want to organize them by *vibe*. A book about heartbreak should sit near other sad stories, regardless of genre. A tweet about startup hustle should be near LinkedIn posts about grinding. You need a way to convert any piece of content—text, images, anything—into coordinates that capture what it *means*. Welcome to embeddings.",
    "hook": "How do you measure the distance between ideas?",
    "concept": "An embedding is just a list of numbers (a vector) that represents the meaning of content. Similar content = similar numbers = close together in 'meaning space'. When you save a tweet about AI, it gets converted to something like [0.23, -0.15, 0.87, ...] (hundreds of dimensions). Another tweet about machine learning would have similar numbers. A post about cooking? Totally different numbers.\n\nThe magic: these numbers aren't random. They're learned by AI models that have seen billions of examples. The model learns that 'king - man + woman = queen' actually works mathematically with embeddings.",
    "analogy": "Think of GPS coordinates. San Francisco is at (37.7749, -122.4194). New York is at (40.7128, -74.0060). You can calculate the distance between them. Embeddings are the same, but instead of latitude/longitude, you have hundreds of dimensions representing different aspects of meaning: formality, topic, sentiment, complexity, etc.",
    "visual": "        MEANING SPACE (simplified to 2D)\n        \n        Formal ↑\n              │    📄 Research Paper\n              │         📊 LinkedIn Post\n              │    \n              │              📧 Work Email\n              │    \n        ──────┼────────────────────────→ Technical\n              │    \n              │         🐦 Tech Tweet\n              │    \n              │    💬 Slack Message\n              │              😂 Meme\n        Casual",
    "interactive": [
      {
        "type": "code",
        "title": "Generate Your First Embedding",
        "description": "Using OpenAI's embedding model to convert text to vectors",
        "code": "from openai import OpenAI\nclient = OpenAI()\n\n# Your saved content\ntweet = \"Just shipped a new feature. 14 hour day but worth it.\"\n\n# Convert to embedding (1536 dimensions!)\nresponse = client.embeddings.create(\n    model=\"text-embedding-3-small\",\n    input=tweet\n)\n\nembedding = response.data[0].embedding\nprint(f\"Dimensions: {len(embedding)}\")\nprint(f\"First 5 values: {embedding[:5]}\")\n# Output: Dimensions: 1536\n# First 5 values: [0.023, -0.156, 0.087, 0.234, -0.045]",
        "explanation": "That tweet is now a point in 1536-dimensional space. Any similar content will be nearby."
      },
      {
        "type": "code",
        "title": "Measure Similarity Between Content",
        "description": "Calculate how 'close' two pieces of content are",
        "code": "import numpy as np\n\ndef cosine_similarity(a, b):\n    \"\"\"How similar are two embeddings? (0 to 1)\"\"\"\n    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\n\n# Your saved items\ntweet1 = \"Building in public. Day 47 of my startup journey.\"\ntweet2 = \"Launched my side project today. Scary but exciting.\"\ntweet3 = \"Best pasta recipe: use pasta water in the sauce.\"\n\n# Get embeddings for each\nemb1 = get_embedding(tweet1)  # startup content\nemb2 = get_embedding(tweet2)  # also startup content  \nemb3 = get_embedding(tweet3)  # cooking content\n\nprint(cosine_similarity(emb1, emb2))  # ~0.89 (very similar!)\nprint(cosine_similarity(emb1, emb3))  # ~0.34 (not similar)",
        "explanation": "The startup tweets cluster together. The pasta tweet is far away. This is how your context layer knows what's related."
      }
    ],
    "keyPoints": [
      "Embeddings convert content into numbers that capture meaning",
      "Similar content = similar numbers = close in vector space",
      "Modern embedding models use 256-3072 dimensions",
      "Cosine similarity measures how close two embeddings are (0-1)",
      "This is the foundation for pattern detection, search, and recommendations"
    ],
    "realWorld": [
      "Your context layer: When you save an Instagram post about design, it automatically groups with your other design saves",
      "ChatGPT: Uses embeddings to find relevant context from your uploaded files",
      "Spotify: Song embeddings power 'listeners also liked'",
      "Google: Search embeddings match queries to results by meaning, not just keywords"
    ],
    "easterEgg": "OpenAI's text-embedding-3-large has 3072 dimensions. That means each piece of content becomes a point in a space with 3072 axes. Humans can only visualize 3 dimensions—these models operate in spaces we literally cannot imagine.",
    "challenge": {
      "unlocks": "embedding-pipeline",
      "preview": "Build a function that takes any content (tweet, article, image caption) and stores it with its embedding in your SQLite database. This is the foundation of your entire platform.",
      "xp": 150
    }
  },
  {
    "id": "semantic-search",
    "level": 2,
    "title": "The Mind Reader",
    "subtitle": "Find content by meaning, not keywords",
    "emoji": "🔮",
    "story": "You saved 500 pieces of content over the past month. Now you're working on a pitch deck and think 'I remember seeing something about viral growth tactics...' You search 'viral' but nothing comes up—because the post you're thinking of said 'exponential user acquisition.' Traditional search fails. You need a system that understands what you *mean*, not just what you *type*. You need semantic search.",
    "hook": "What if search could read your mind?",
    "concept": "Semantic search converts your query into an embedding, then finds the saved content with the closest embeddings. It doesn't match keywords—it matches meaning.\n\nQuery: 'viral growth' → embedding → finds content about 'exponential user acquisition', 'hockey stick metrics', 'product-led growth'\n\nThis is why your context layer can surface relevant saves even when you don't remember the exact words.",
    "analogy": "Traditional search is like finding a book by its exact title. Semantic search is like asking a librarian who's read every book: 'I need something about overcoming failure'—and they hand you books about resilience, grit, comeback stories, even if none have 'failure' in the title.",
    "visual": "    QUERY: \"startup fundraising tips\"\n           ↓\n    [Convert to embedding]\n           ↓\n    🎯 Query vector in meaning-space\n           \n    Find nearest neighbors:\n    ├─ 📄 \"How I raised my seed round\" (0.91)\n    ├─ 📄 \"VC pitch deck template\" (0.87)\n    ├─ 📄 \"Angel investor red flags\" (0.84)\n    └─ 📄 \"Term sheet negotiation\" (0.79)\n    \n    ❌ Ignored (far away):\n    └─ 📄 \"My startup journey\" (0.45) - too vague",
    "interactive": [
      {
        "type": "code",
        "title": "Build Semantic Search",
        "description": "Search your saved content by meaning",
        "code": "import numpy as np\nfrom openai import OpenAI\n\nclient = OpenAI()\n\ndef semantic_search(query, saved_items, top_k=5):\n    \"\"\"Find most relevant saved content\"\"\"\n    \n    # Convert query to embedding\n    query_emb = client.embeddings.create(\n        model=\"text-embedding-3-small\",\n        input=query\n    ).data[0].embedding\n    \n    # Calculate similarity to all saved items\n    results = []\n    for item in saved_items:\n        similarity = cosine_similarity(query_emb, item['embedding'])\n        results.append((item, similarity))\n    \n    # Return top matches\n    results.sort(key=lambda x: x[1], reverse=True)\n    return results[:top_k]\n\n# Usage\nquery = \"content that went viral\"\nmatches = semantic_search(query, my_saved_content)\n\nfor item, score in matches:\n    print(f\"{score:.2f}: {item['title']}\")",
        "explanation": "This searches by meaning. 'Content that went viral' will find posts about 'blew up on Twitter', 'hit the front page', 'million views overnight'—even without the word 'viral'."
      },
      {
        "type": "code",
        "title": "Add to SQLite",
        "description": "Store embeddings for fast retrieval",
        "code": "import sqlite3\nimport json\n\ndef save_content(content, source, embedding):\n    \"\"\"Save content with its embedding\"\"\"\n    conn = sqlite3.connect('context.db')\n    c = conn.cursor()\n    \n    c.execute('''\n        INSERT INTO saved_content \n        (content, source, embedding, created_at)\n        VALUES (?, ?, ?, datetime('now'))\n    ''', (content, source, json.dumps(embedding)))\n    \n    conn.commit()\n    conn.close()\n\ndef search_content(query_embedding, limit=10):\n    \"\"\"Retrieve and rank by similarity\"\"\"\n    conn = sqlite3.connect('context.db')\n    c = conn.cursor()\n    \n    # Get all embeddings (in production, use vector DB)\n    c.execute('SELECT id, content, embedding FROM saved_content')\n    \n    results = []\n    for row in c.fetchall():\n        emb = json.loads(row[2])\n        sim = cosine_similarity(query_embedding, emb)\n        results.append({'id': row[0], 'content': row[1], 'score': sim})\n    \n    return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]",
        "explanation": "SQLite works for small collections (<10k items). For serious scale, you'll want a vector database like Pinecone, Weaviate, or Chroma."
      }
    ],
    "keyPoints": [
      "Semantic search matches meaning, not keywords",
      "Query → embedding → find nearest saved embeddings",
      "Cosine similarity ranks results (1.0 = identical meaning)",
      "Works across languages: search in English, find Spanish content",
      "This is how your context layer retrieves relevant saves"
    ],
    "realWorld": [
      "Your context layer: Search 'productivity hacks' and find that Japanese blog post about time management",
      "Claude/ChatGPT: When you ask a question, it searches your uploaded docs semantically",
      "Notion AI: Searches your workspace by meaning",
      "Perplexity: Combines semantic search with web results"
    ],
    "easterEgg": "Semantic search works across languages because embeddings capture meaning, not words. A query in English will find relevant content in Japanese, Spanish, or Arabic—if the meanings are similar, the embeddings are close.",
    "challenge": {
      "unlocks": "search-interface",
      "preview": "Build a search endpoint for your Chrome extension. When you type a query, it returns the top 10 most relevant saved items with similarity scores. Bonus: highlight why each result matched.",
      "xp": 150
    }
  },
  {
    "id": "clustering-patterns",
    "level": 3,
    "title": "The Pattern Detective",
    "subtitle": "Auto-discover themes in your saved content",
    "emoji": "🕵️",
    "story": "You've saved 1,000 items over three months. You didn't tag any of them. Now you realize: 40% are about AI tools, 25% are startup case studies, 20% are design inspiration, and 15% are random. But you never told the system this—it figured it out by watching your patterns. This is clustering: letting ML discover the hidden structure in your saves.",
    "hook": "What if your library organized itself?",
    "concept": "Clustering groups similar embeddings together automatically. The algorithm doesn't know what 'AI tools' or 'startup case studies' means—it just sees that certain embeddings clump together in vector space.\n\nK-means: 'Find K groups where items in each group are close together'\nHDBSCAN: 'Find natural clusters of any shape/size, ignore outliers'\n\nOnce clusters form, you can name them manually or use AI to generate labels.",
    "analogy": "Imagine dumping 1,000 photos on a table. Without knowing anything about them, you could group them: beach photos here, food photos there, selfies in another pile. You're clustering by visual similarity. Embeddings let you do this with ANY content—tweets, articles, images—all mixed together.",
    "visual": "    BEFORE: Random saves in vector space\n    \n    •  •     •   •  •\n      •   •  •     •\n    •    •      •   •\n      •     • •   •\n    \n    AFTER: Clusters discovered\n    \n    🔵 🔵     🟢 🟢 🟢    (🔵 = AI tools)\n      🔵 🔵  🟢     🟢    (🟢 = Startups)\n    🔵    🔵      🟡 🟡    (🟡 = Design)\n      🔵     🟡 🟡   🟡\n    \n    Auto-generated labels:\n    Cluster 1: \"AI Tools & Automation\"\n    Cluster 2: \"Startup Growth Stories\"\n    Cluster 3: \"Design Inspiration\"",
    "interactive": [
      {
        "type": "code",
        "title": "Discover Clusters with K-Means",
        "description": "Find K groups in your saved content",
        "code": "from sklearn.cluster import KMeans\nimport numpy as np\n\ndef discover_clusters(embeddings, n_clusters=5):\n    \"\"\"Group similar content together\"\"\"\n    \n    # Stack embeddings into matrix\n    X = np.array(embeddings)\n    \n    # Find clusters\n    kmeans = KMeans(n_clusters=n_clusters, random_state=42)\n    labels = kmeans.fit_predict(X)\n    \n    return labels  # Array of cluster IDs for each item\n\n# Usage\nall_embeddings = [item['embedding'] for item in saved_content]\ncluster_labels = discover_clusters(all_embeddings, n_clusters=6)\n\n# Group content by cluster\nfrom collections import defaultdict\nclusters = defaultdict(list)\nfor i, item in enumerate(saved_content):\n    clusters[cluster_labels[i]].append(item)\n\nprint(f\"Cluster 0: {len(clusters[0])} items\")\nprint(f\"Cluster 1: {len(clusters[1])} items\")",
        "explanation": "K-means needs you to specify how many clusters. Start with sqrt(n_items) as a rule of thumb, then adjust based on results."
      },
      {
        "type": "code",
        "title": "Auto-Label Clusters with AI",
        "description": "Generate human-readable names for each cluster",
        "code": "from openai import OpenAI\n\nclient = OpenAI()\n\ndef label_cluster(items, sample_size=10):\n    \"\"\"Generate a label for a cluster based on sample items\"\"\"\n    \n    # Sample items from cluster\n    sample = items[:sample_size]\n    content_list = \"\\n\".join([f\"- {item['content'][:200]}\" for item in sample])\n    \n    response = client.chat.completions.create(\n        model=\"gpt-4o-mini\",\n        messages=[{\n            \"role\": \"user\",\n            \"content\": f\"\"\"These items were grouped together by ML. \nGenerate a short, specific label (2-4 words) for this cluster:\n\n{content_list}\n\nLabel:\"\"\"\n        }]\n    )\n    \n    return response.choices[0].message.content.strip()\n\n# Auto-label all clusters\nfor cluster_id, items in clusters.items():\n    label = label_cluster(items)\n    print(f\"Cluster {cluster_id}: {label}\")\n    # Output: \n    # Cluster 0: AI Development Tools\n    # Cluster 1: Startup Growth Tactics\n    # Cluster 2: UI/UX Inspiration",
        "explanation": "Clustering finds the groups. AI names them. This is how your context layer auto-organizes without you lifting a finger."
      }
    ],
    "keyPoints": [
      "Clustering finds groups of similar embeddings automatically",
      "K-means: specify number of clusters; HDBSCAN: finds them naturally",
      "Use AI to generate human-readable labels for discovered clusters",
      "Re-cluster periodically as you save more content",
      "This is the 'Instagram algo' for your own content"
    ],
    "realWorld": [
      "Your context layer: Auto-create collections like 'AI Tools', 'Startup Playbooks', 'Design Refs' without manual tagging",
      "Spotify: Discovers playlist themes from your listening patterns",
      "Gmail: Groups emails into Primary, Social, Promotions",
      "Instagram Explore: Clusters your engagement into interest categories"
    ],
    "easterEgg": "HDBSCAN (Hierarchical Density-Based Spatial Clustering) can find clusters of any shape—even crescent moons or spirals. K-means only finds spherical clusters. For messy human behavior data, HDBSCAN usually wins.",
    "challenge": {
      "unlocks": "auto-collections",
      "preview": "Build a nightly job that re-clusters your saved content and updates collection labels. When you open the app, you see auto-organized folders that actually make sense.",
      "xp": 200
    }
  },
  {
    "id": "recommendation-systems",
    "level": 4,
    "title": "The Taste Predictor",
    "subtitle": "Surface content you'll love before you search",
    "emoji": "🎯",
    "story": "You open Instagram and the first reel is exactly what you wanted to see—even though you didn't search for it. The algorithm watched you: you paused on startup content, skipped fitness stuff, rewatched that design breakdown. Now it knows your taste better than you do. This is recommendation systems: predicting what you want before you ask.",
    "hook": "What if your library knew what you needed next?",
    "concept": "Recommendation systems use your behavior to predict relevance. Two main approaches:\n\n**Content-based**: 'You saved X, here's similar content Y' (uses embeddings)\n**Collaborative**: 'Users like you also saved Y' (requires multiple users)\n\nFor your context layer (single user), content-based is the play. Weight recent saves higher, track what you actually use vs. just save, learn your current focus.",
    "analogy": "A great barista remembers: 'Last week you ordered oat milk lattes, but this week you've been doing cold brew. Want your usual... cold brew?' They're not just remembering your orders—they're tracking patterns over time and predicting what you want now.",
    "visual": "    YOUR BEHAVIOR SIGNALS:\n    \n    🔥 High signal (recent, engaged)\n    ├─ Saved 3 MCP articles today\n    ├─ Searched 'Claude integration' twice\n    └─ Opened RAG doc 4 times\n    \n    💤 Low signal (old, ignored)\n    ├─ Saved design inspo 2 months ago\n    └─ Never opened those cooking saves\n    \n    RECOMMENDATION:\n    \"Based on your recent focus on AI integration,\n    you might want to revisit this saved thread\n    about LangChain context windows...\"",
    "interactive": [
      {
        "type": "code",
        "title": "Build a Simple Recommender",
        "description": "Surface relevant saves based on recent activity",
        "code": "import numpy as np\nfrom datetime import datetime, timedelta\n\ndef get_recommendations(saved_items, recent_activity, top_k=5):\n    \"\"\"Recommend saves based on recent behavior\"\"\"\n    \n    # Build user profile from recent activity (last 7 days)\n    recent_embeddings = []\n    for activity in recent_activity:\n        if activity['timestamp'] > datetime.now() - timedelta(days=7):\n            # Weight by recency and engagement\n            weight = activity.get('engagement_score', 1.0)\n            recent_embeddings.append(\n                np.array(activity['embedding']) * weight\n            )\n    \n    if not recent_embeddings:\n        return []  # No recent activity\n    \n    # Average = your current \"interest vector\"\n    user_profile = np.mean(recent_embeddings, axis=0)\n    \n    # Find saves closest to current interests\n    scores = []\n    for item in saved_items:\n        # Skip items you've seen recently\n        if item['id'] in [a['item_id'] for a in recent_activity[-20:]]:\n            continue\n            \n        sim = cosine_similarity(user_profile, item['embedding'])\n        \n        # Boost older saves (resurface forgotten content)\n        days_old = (datetime.now() - item['created_at']).days\n        recency_boost = min(days_old / 30, 2.0)  # Up to 2x for old saves\n        \n        scores.append((item, sim * (1 + recency_boost * 0.3)))\n    \n    scores.sort(key=lambda x: x[1], reverse=True)\n    return scores[:top_k]",
        "explanation": "This creates a 'profile vector' from your recent saves/searches, then finds old saves that match. Older content gets boosted—resurfacing forgotten gems."
      },
      {
        "type": "code",
        "title": "Track Engagement Signals",
        "description": "Learn what you actually care about",
        "code": "def record_engagement(item_id, action_type):\n    \"\"\"Track what users do with saved content\"\"\"\n    \n    engagement_weights = {\n        'save': 1.0,       # Saved it\n        'view': 0.5,       # Opened it\n        'search_click': 2.0,  # Found via search and clicked\n        'copy': 3.0,       # Copied text from it\n        'share': 4.0,      # Shared it\n        'use_in_ai': 5.0,  # Used as context in AI chat\n    }\n    \n    weight = engagement_weights.get(action_type, 1.0)\n    \n    # Store engagement event\n    conn.execute('''\n        INSERT INTO engagement_log \n        (item_id, action, weight, timestamp)\n        VALUES (?, ?, ?, datetime('now'))\n    ''', (item_id, action_type, weight))\n    \n    # Update item's total engagement score\n    conn.execute('''\n        UPDATE saved_content \n        SET engagement_score = engagement_score + ?\n        WHERE id = ?\n    ''', (weight, item_id))",
        "explanation": "Not all saves are equal. Content you copy, share, or use in AI is way more valuable than content you just saved and forgot."
      }
    ],
    "keyPoints": [
      "Content-based: recommend similar to what you've engaged with",
      "Create a 'profile vector' by averaging recent activity embeddings",
      "Weight by engagement: copy > view > save",
      "Resurface old content that matches current interests",
      "Track usage, not just saves—what you USE matters most"
    ],
    "realWorld": [
      "Your context layer: 'Based on your AI research this week, here are 5 saves you might want to revisit'",
      "YouTube: 'Recommended for you' sidebar",
      "TikTok: The entire For You page is recommendations",
      "Amazon: 'Customers who bought this also bought'"
    ],
    "easterEgg": "TikTok's algorithm is so good because they track micro-signals: how long you watch before scrolling, if you rewatch, if you tap the caption. The more signals you track, the better recommendations get. Your engagement_weights is a simple version of this.",
    "challenge": {
      "unlocks": "smart-home-feed",
      "preview": "Build a 'For You' feed for your context layer. When you open the app, show 5 resurfaces: old saves that match your current focus. Include the reason: 'Because you've been researching MCP this week...'",
      "xp": 200
    }
  },
  {
    "id": "rag-basics",
    "level": 5,
    "title": "The Context Surgeon",
    "subtitle": "Inject exactly the right context into AI—nothing more",
    "emoji": "✂️",
    "story": "You're in ChatGPT working on a pitch deck. You've saved 500 items about startups, fundraising, and pitching. You want the AI to reference YOUR knowledge—but dumping 500 items would explode the context window and cost $$$. RAG is the scalpel: retrieve only the 3-5 most relevant pieces, inject them surgically, get gold-quality responses.",
    "hook": "How do you give AI a photographic memory without bankruptcy?",
    "concept": "RAG = Retrieval Augmented Generation\n\n1. User asks a question\n2. System searches your saves semantically\n3. Top K relevant items retrieved\n4. Items injected into prompt as context\n5. AI generates response using YOUR knowledge\n\nThe magic: you get personalized responses without fine-tuning or massive prompts. The AI only sees what's relevant right now.",
    "analogy": "Imagine a lawyer with a library of 10,000 case files. When a client asks a question, the lawyer doesn't read all 10,000 files—they search for the 3-5 most relevant precedents and reference those. RAG is that librarian-brain for AI.",
    "visual": "    USER: \"What are best practices for cold outreach?\"\n                    │\n                    ▼\n    ┌─────────────────────────────────────┐\n    │  1. SEMANTIC SEARCH your saves      │\n    │     Query → embedding → search      │\n    └─────────────────────────────────────┘\n                    │\n                    ▼\n    ┌─────────────────────────────────────┐\n    │  2. RETRIEVE top 3-5 matches        │\n    │     • \"How I booked 20 calls/week\"  │\n    │     • \"Cold email template thread\"  │\n    │     • \"B2B outreach mistakes\"       │\n    └─────────────────────────────────────┘\n                    │\n                    ▼\n    ┌─────────────────────────────────────┐\n    │  3. INJECT into prompt              │\n    │     \"Using this context: [...]      │\n    │      Answer the user's question\"    │\n    └─────────────────────────────────────┘\n                    │\n                    ▼\n    🤖 AI response grounded in YOUR saves",
    "interactive": [
      {
        "type": "code",
        "title": "Build a RAG Pipeline",
        "description": "The core loop: retrieve → inject → generate",
        "code": "from openai import OpenAI\n\nclient = OpenAI()\n\ndef rag_query(user_question, saved_content, top_k=5):\n    \"\"\"Answer questions using your saved context\"\"\"\n    \n    # 1. RETRIEVE: Search for relevant saves\n    relevant = semantic_search(user_question, saved_content, top_k=top_k)\n    \n    # 2. FORMAT: Build context block\n    context_parts = []\n    for item, score in relevant:\n        context_parts.append(f\"\"\"---\nSource: {item['source']}\nRelevance: {score:.2f}\nContent: {item['content'][:500]}\n---\"\"\")\n    \n    context_block = \"\\n\\n\".join(context_parts)\n    \n    # 3. INJECT: Create augmented prompt\n    augmented_prompt = f\"\"\"Use the following saved content to answer the question.\nOnly use information from these sources. Cite which source you used.\n\nSAVED CONTENT:\n{context_block}\n\nQUESTION: {user_question}\n\nANSWER:\"\"\"\n    \n    # 4. GENERATE: Get AI response\n    response = client.chat.completions.create(\n        model=\"gpt-4o\",\n        messages=[{\"role\": \"user\", \"content\": augmented_prompt}]\n    )\n    \n    return {\n        'answer': response.choices[0].message.content,\n        'sources': relevant  # Show what was used\n    }",
        "explanation": "This is the core RAG loop. User question → search → inject relevant context → AI generates grounded response. Simple but powerful."
      },
      {
        "type": "code",
        "title": "Optimize Context Window Usage",
        "description": "Get more signal, less noise per token",
        "code": "def smart_context_injection(relevant_items, max_tokens=2000):\n    \"\"\"Pack maximum signal into limited tokens\"\"\"\n    \n    context_parts = []\n    current_tokens = 0\n    \n    for item, score in relevant_items:\n        # Estimate tokens (~4 chars per token)\n        item_tokens = len(item['content']) // 4\n        \n        if current_tokens + item_tokens > max_tokens:\n            # Truncate to fit remaining space\n            remaining = max_tokens - current_tokens\n            truncated = item['content'][:remaining * 4]\n            context_parts.append(f\"[{item['source']}] {truncated}...\")\n            break\n        \n        context_parts.append(f\"[{item['source']}] {item['content']}\")\n        current_tokens += item_tokens\n    \n    return \"\\n\\n\".join(context_parts)\n\n# Advanced: summarize low-relevance items\ndef compress_context(items, threshold=0.8):\n    \"\"\"Full content for high relevance, summaries for medium\"\"\"\n    context = []\n    for item, score in items:\n        if score > threshold:\n            # High relevance: include full\n            context.append(item['content'])\n        else:\n            # Medium relevance: summarize\n            summary = get_summary(item['content'])  # Use AI\n            context.append(f\"[Summary] {summary}\")\n    return context",
        "explanation": "Context window is expensive real estate. High-relevance content gets full space; medium gets summarized. Maximize signal per token."
      }
    ],
    "keyPoints": [
      "RAG = Retrieve relevant context → Inject into prompt → Generate",
      "Semantic search finds relevant saves without keyword matching",
      "Only inject what's relevant—don't dump everything",
      "Budget tokens carefully: full content for high relevance, summaries for medium",
      "Always show sources—builds trust and lets users verify"
    ],
    "realWorld": [
      "Your context layer: 'Ask anything about your saves' feature—questions answered by YOUR knowledge",
      "ChatGPT file uploads: Exactly this pattern",
      "Notion AI: Searches your workspace, injects relevant docs",
      "Perplexity: RAG over the entire web"
    ],
    "easterEgg": "The 'retrieval' in RAG is usually the bottleneck. A query takes 0.1s, but if you have 100k embeddings, searching all of them takes forever. Vector databases solve this with approximate nearest neighbor algorithms—trading tiny accuracy for 1000x speed.",
    "challenge": {
      "unlocks": "context-api",
      "preview": "Build an API endpoint: POST /ask with a question → returns answer + sources. This is the core of your 'context layer for AI' vision. Any AI platform can call this to get your personalized context.",
      "xp": 250
    }
  },
  {
    "id": "mcp-integration",
    "level": 6,
    "title": "The Universal Translator",
    "subtitle": "Connect your context layer to any AI",
    "emoji": "🔌",
    "story": "You've built it: one-click save, auto-clustering, semantic search, RAG. But now you have to copy-paste context into ChatGPT like an animal. What if Claude could just... access your saves directly? What if any AI could plug into your context layer? That's MCP: the USB-C of AI integrations.",
    "hook": "What if every AI could speak directly to your brain?",
    "concept": "MCP (Model Context Protocol) is Anthropic's open standard for connecting AI to external data. Instead of uploading files manually, the AI can query your context layer directly.\n\n**Without MCP**: You → copy saves → paste into ChatGPT → ask question\n**With MCP**: You ask question → Claude queries your saves → responds with context\n\nYou build one MCP server, and any MCP-compatible AI can use your context layer.",
    "analogy": "Before USB-C, every device had its own charger. MCP is USB-C for AI: a universal standard that lets any model plug into any data source. Your context layer becomes a power bank that works with every device.",
    "visual": "    WITHOUT MCP:\n    ┌─────────┐     📋 copy     ┌─────────┐\n    │ Context │ ───────────────▶ │ ChatGPT │\n    │  Layer  │     📋 paste    │         │\n    └─────────┘                  └─────────┘\n         You do all the work ^\n    \n    WITH MCP:\n    ┌─────────┐   🔌 MCP query  ┌─────────┐\n    │ Context │ ◀───────────── │ Claude  │\n    │  Layer  │   📦 returns    │         │\n    │ (Server)│ ───────────────▶│(Client) │\n    └─────────┘    context      └─────────┘\n         AI does the work ^",
    "interactive": [
      {
        "type": "code",
        "title": "Build an MCP Server",
        "description": "Expose your context layer to Claude",
        "code": "# mcp_server.py\nfrom mcp import MCPServer, Tool, Resource\n\nserver = MCPServer(\"context-layer\")\n\n@server.tool(\"search_saves\")\nasync def search_saves(query: str, limit: int = 5):\n    \"\"\"Search user's saved content by meaning\"\"\"\n    results = semantic_search(query, saved_content, top_k=limit)\n    return [{\n        'content': item['content'],\n        'source': item['source'],\n        'relevance': score\n    } for item, score in results]\n\n@server.tool(\"get_recent_saves\")\nasync def get_recent_saves(days: int = 7, limit: int = 10):\n    \"\"\"Get user's recent saves\"\"\"\n    recent = get_saves_since(days)\n    return recent[:limit]\n\n@server.tool(\"get_clusters\")\nasync def get_clusters():\n    \"\"\"Get auto-discovered content themes\"\"\"\n    return [{\n        'name': c['label'],\n        'count': len(c['items']),\n        'sample': c['items'][0]['content'][:100]\n    } for c in clusters]\n\n@server.resource(\"saves://{save_id}\")\nasync def get_save(save_id: str):\n    \"\"\"Get full content of a specific save\"\"\"\n    return get_save_by_id(save_id)\n\n# Run server\nif __name__ == \"__main__\":\n    server.run()",
        "explanation": "This MCP server exposes 3 tools (search, recent, clusters) and a resource pattern (individual saves). Claude can now query your context layer directly."
      },
      {
        "type": "code",
        "title": "Configure Claude Desktop",
        "description": "Connect Claude to your context layer",
        "code": "// ~/Library/Application Support/Claude/claude_desktop_config.json\n{\n  \"mcpServers\": {\n    \"context-layer\": {\n      \"command\": \"python\",\n      \"args\": [\"/path/to/mcp_server.py\"],\n      \"env\": {\n        \"DATABASE_PATH\": \"/path/to/context.db\"\n      }\n    }\n  }\n}\n\n// Now in Claude, you can ask:\n// \"Search my saves for content about fundraising\"\n// Claude will call search_saves(\"fundraising\") automatically\n// and respond with your actual saved content",
        "explanation": "Once configured, Claude sees your context layer as a native capability. 'Search my saves' just works."
      }
    ],
    "keyPoints": [
      "MCP is a universal protocol for connecting AI to external data",
      "You build an MCP server; any MCP client (Claude, etc.) can connect",
      "Tools = actions the AI can take (search, get clusters)",
      "Resources = data the AI can read (individual saves)",
      "One server → works with Claude Desktop, Claude Code, any MCP client"
    ],
    "realWorld": [
      "Your context layer: Claude directly searches your saves without copy-paste",
      "Notion MCP: Let Claude read/write to your Notion workspace",
      "GitHub MCP: Let Claude browse repos and create PRs",
      "Slack MCP: Let Claude search and send messages"
    ],
    "easterEgg": "MCP was released December 2024 and is still evolving. Anthropic made it open source hoping it becomes the standard—like HTTP for AI integrations. If it wins, your context layer MCP server will work with every future AI model that supports the protocol.",
    "challenge": {
      "unlocks": "production-mcp",
      "preview": "Deploy your MCP server and test it with Claude Desktop. Save a few tweets, then ask Claude 'What did I save about X?' without uploading anything. Magic moment when it just works.",
      "xp": 250
    }
  }
];

// Progress stored on disk (Railway volume or ephemeral)
const PROGRESS_FILE = path.join(__dirname, 'data', 'progress.json');

function getProgress() {
  try {
    return JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8'));
  } catch {
    return { xp: 0, streak: 0, lastActive: null, completedLessons: [], completedChallenges: [] };
  }
}

function saveProgress(progress) {
  try {
    const dir = path.dirname(PROGRESS_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
  } catch (err) {
    console.error('[warn] saveProgress failed:', err.message);
  }
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/api/lessons', (req, res) => res.json(LESSONS));

app.get('/api/lessons/:id', (req, res) => {
  const lesson = LESSONS.find(l => l.id === req.params.id);
  if (!lesson) return res.status(404).json({ error: 'Lesson not found' });
  res.json(lesson);
});

app.post('/api/lessons/:id/complete', (req, res) => {
  try {
    const progress = getProgress();
    const lesson = LESSONS.find(l => l.id === req.params.id);
    if (!lesson) return res.status(404).json({ error: 'Lesson not found' });

    if (progress.completedLessons.includes(req.params.id)) {
      return res.json({ xp: progress.xp, xpGained: 0, streak: progress.streak,
        completedLessons: progress.completedLessons.length, alreadyCompleted: true });
    }

    const lessonIndex = LESSONS.findIndex(l => l.id === req.params.id);
    if (lessonIndex > 0) {
      const prevLesson = LESSONS[lessonIndex - 1];
      if (!progress.completedLessons.includes(prevLesson.id)) {
        return res.status(400).json({ error: 'Must complete previous lesson first' });
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
    console.error('[error] complete lesson:', err.message);
    res.status(500).json({ error: 'Failed to complete lesson' });
  }
});

app.get('/api/progress', (req, res) => res.json(getProgress()));

app.get('/api/dashboard', (req, res) => {
  try {
    const progress = getProgress();
    let nextLesson = null;
    for (const lesson of LESSONS) {
      if (!progress.completedLessons.includes(lesson.id)) {
        nextLesson = { id: lesson.id, level: lesson.level, title: lesson.title, emoji: lesson.emoji };
        break;
      }
    }
    res.json({ xp: progress.xp, streak: progress.streak,
      completedLessons: progress.completedLessons.length,
      totalLessons: LESSONS.length, nextLesson });
  } catch (err) {
    res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

app.post('/api/reset', (req, res) => {
  const initial = { xp: 0, streak: 0, lastActive: null, completedLessons: [], completedChallenges: [] };
  saveProgress(initial);
  res.json({ success: true });
});

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log('[startup] Server listening on port', PORT);
});

server.on('error', (err) => {
  console.error('[FATAL] Server failed to start:', err.message);
  process.exit(1);
});
