# RepoMind - Developer Guide for Junior Developers

> **Welcome!** This guide will help you understand and contribute to the RepoMind project, even if you're new to some of the technologies we use. Everything is explained in plain English with examples.

---

## 📚 Table of Contents

1. [What is RepoMind?](#what-is-repomind)
2. [How to Read This Guide](#how-to-read-this-guide)
3. [Setting Up Your Development Environment](#setting-up-your-development-environment)
4. [Understanding the Big Picture](#understanding-the-big-picture)
5. [Key Concepts Explained Simply](#key-concepts-explained-simply)
6. [Project Structure Walkthrough](#project-structure-walkthrough)
7. [How Code Flows Through the System](#how-code-flows-through-the-system)
8. [Making Your First Contribution](#making-your-first-contribution)
9. [Common Tasks and How to Do Them](#common-tasks-and-how-to-do-them)
10. [Testing Your Changes](#testing-your-changes)
11. [Debugging Guide](#debugging-guide)
12. [Best Practices](#best-practices)
13. [Getting Help](#getting-help)

---

## What is RepoMind?

Imagine you're working on a huge codebase with thousands of files. You need to find "that function that handles user authentication" but you don't know what it's called or which file it's in. Traditional search (Ctrl+F) won't help because you need to search by *meaning*, not exact words.

**That's what RepoMind does:**

1. **Reads all your code** (Python, Java, TypeScript)
2. **Understands what each piece does** (using AI embeddings)
3. **Lets you search by meaning** ("find authentication code")
4. **Shows you connections** (what calls what, what depends on what)
5. **Works with AI assistants** like Claude to help you code

**In simple terms:** It's like Google for your codebase, but smarter.

---

## How to Read This Guide

### Symbols Used

- 💡 **Tip**: Helpful advice
- ⚠️ **Warning**: Important thing to watch out for
- 📖 **Learn More**: Links to deeper information
- ✅ **Checkpoint**: Verify you did it right
- 🎯 **Goal**: What you'll achieve

### Prerequisites Boxes

Before each section, you'll see a box like this:

```
Prerequisites:
✓ Python basics (variables, functions, classes)
✓ Git basics (clone, commit, push)
✓ Terminal/command line usage
```

This tells you what knowledge you need before starting that section.

---

## Setting Up Your Development Environment

### Prerequisites
```
✓ Basic computer skills
✓ Ability to use terminal/command line
✓ Nothing else! We'll guide you through everything.
```

### 🎯 Goal
Get RepoMind running on your machine so you can develop and test.

### Step 1: Install Python

**What is Python?** A programming language. Our project is written in it.

**Mac:**
```bash
# Install Homebrew (a package manager) if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Important:** Check "Add Python to PATH"

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

✅ **Checkpoint:** Run `python3 --version`. You should see "Python 3.11.x" or higher.

### Step 2: Install Git

**What is Git?** Version control - it tracks changes to code.

**Mac:**
```bash
brew install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/download/win)

**Linux:**
```bash
sudo apt install git
```

✅ **Checkpoint:** Run `git --version`. You should see a version number.

### Step 3: Clone the Repository

**What does "clone" mean?** It copies the project from GitHub to your computer.

```bash
# Navigate to where you want the project
cd ~/Documents/GitHub/repomind

# Clone the repository
git clone <repository-url>

# Go into the project folder
cd repomind
```

✅ **Checkpoint:** Run `ls` (or `dir` on Windows). You should see files like `README.md`, `pyproject.toml`.

### Step 4: Set Up Python Virtual Environment

**What is a virtual environment?** A separate space for this project's dependencies so they don't conflict with other Python projects.

```bash
# Create a virtual environment (one-time)
python3 -m venv .venv

# Activate it (do this every time you open a new terminal)
# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

💡 **Tip:** When activated, you'll see `(.venv)` at the start of your terminal prompt.

✅ **Checkpoint:** Run `which python` (Mac/Linux) or `where python` (Windows). It should point to `.venv/bin/python`.

### Step 5: Install Dependencies

**What are dependencies?** Other people's code (libraries) that our project uses.

```bash
# Make sure your virtual environment is activated!
pip install -e ".[dev]"
```

**What does this do?**
- `pip`: Python's package installer
- `install`: Installs packages
- `-e`: "Editable" mode - changes you make are immediately usable
- `".[dev]"`: Install this project + development tools

This will take a few minutes. You'll see lots of packages being downloaded.

✅ **Checkpoint:** Run `repomind --help`. You should see a help message.

### Step 6: Install an IDE

**What is an IDE?** A code editor with helpful features like autocomplete and error highlighting.

**Recommended:** [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) (free) or [VS Code](https://code.visualstudio.com/)

**Configure your IDE:**
1. Open the `repomind` folder
2. Set Python interpreter to `.venv/bin/python`
3. Install Python language support

✅ **Checkpoint:** Open `src/repomind/cli.py` in your IDE. You should see syntax highlighting.

---

## Understanding the Big Picture

### Prerequisites
```
✓ Completed "Setting Up Your Development Environment"
✓ Basic understanding of what functions and classes are
```

### 🎯 Goal
Understand what RepoMind does and why, without diving into code yet.

### The Problem We're Solving

**Scenario:** You're new to a large codebase (10,000+ lines). Someone says "We need to fix the authentication bug." You need to find:
1. Where authentication code is
2. What it does
3. What other code uses it

**Traditional solutions:**
- Text search (Ctrl+F) - only finds exact words
- Grep - same problem
- Reading everything - too slow

**Our solution:** RepoMind

### How It Works (Simple Version)

Think of it like a librarian who:

1. **Reads every book** (indexes your code)
2. **Understands the content** (creates "embeddings" - AI understanding)
3. **Organizes them** (stores in databases)
4. **Helps you find what you need** (searches by meaning)

### Real Example

**User asks:** "Find code that validates email addresses"

**System does:**
1. Converts your question to numbers (embedding)
2. Compares with all code (also converted to numbers)
3. Finds most similar code
4. Returns: `validateEmail()` function, even though it doesn't contain the word "validate"!

### The Three Main Operations

#### 1. Indexing (One-Time Setup)
```
Code Files → Parse → Understand → Store
```
Like creating an index in a book's back pages.

#### 2. Searching (Frequent)
```
Your Question → Convert to Numbers → Find Similar → Return Results
```
Like using that index to find page numbers.

#### 3. Navigation (Frequent)
```
Function Name → Lookup → Get Full Context
```
Like jumping to a specific page number.

---

## Key Concepts Explained Simply

### Prerequisites
```
✓ Understanding the Big Picture section
✓ Curiosity about how things work
```

### 🎯 Goal
Understand the technical terms you'll see in the code.

### 1. Embeddings (The Magic Sauce)

**Simple explanation:** Converting text to numbers that capture meaning.

**Example:**
```
"dog" → [0.8, 0.2, 0.1, ...]
"puppy" → [0.7, 0.3, 0.1, ...]  # Similar numbers!
"car" → [0.1, 0.1, 0.9, ...]    # Different numbers
```

**Why?** Computers can compare numbers easily. Similar meanings = similar numbers.

**In our code:** We use a model called "BGE" to create these embeddings.

📖 **Learn More:** [What are embeddings?](https://platform.openai.com/docs/guides/embeddings)

### 2. Vector Database (ChromaDB)

**Simple explanation:** A database that stores numbers and finds similar ones quickly.

**Regular database:** "Find all users where name = 'John'" (exact match)

**Vector database:** "Find all vectors similar to [0.8, 0.2, 0.1]" (similarity match)

**In our code:** ChromaDB stores code embeddings for fast similarity search.

### 3. AST (Abstract Syntax Tree)

**Simple explanation:** A tree structure representing code.

**Example:**
```python
def greet(name):
    print(f"Hello {name}")
```

Becomes:
```
FunctionDef
├─ name: "greet"
├─ args: ["name"]
└─ body:
   └─ Call
      ├─ func: "print"
      └─ args: [...]
```

**Why?** Easier to extract information than parsing raw text.

**In our code:** We use tree-sitter to create ASTs for Python, Java, TypeScript.

### 4. Code Chunks

**Simple explanation:** A meaningful piece of code (function, class, method).

**Example chunk:**
```python
{
  "name": "validateEmail",
  "type": "function",
  "file": "auth/validator.py",
  "line": 45,
  "code": "def validateEmail(email): ...",
  "docstring": "Check if email is valid"
}
```

**In our code:** Everything is broken into chunks for indexing.

### 5. Hybrid Search

**Simple explanation:** Combining AI understanding with keyword matching.

**Example:**
```
Query: "authentication middleware"

Semantic score: 0.7  (AI thinks it's 70% relevant)
Keyword bonus: +0.1  (found "auth" in function name)
Final score: 0.8     (80% relevant)
```

**Why?** Better accuracy than semantic or keyword alone.

### 6. MCP (Model Context Protocol)

**Simple explanation:** A standard way for AI assistants (like Claude) to use tools.

**Analogy:** Like USB - any device can plug into any computer if they both use USB.

**In our code:** We expose our search as "tools" that Claude can use.

---

## Project Structure Walkthrough

### Prerequisites
```
✓ Completed "Setting Up Your Development Environment"
✓ IDE open with project loaded
```

### 🎯 Goal
Know where to find code for different features.

### Directory Tree (Annotated)

```
repomind/
│
├── 📄 README.md                    # User documentation (what you read first)
├── 📄 DEVELOPER_GUIDE.md           # This file! (for developers)
├── 📄 pyproject.toml               # Python project config (dependencies, etc.)
├── 📄 mcp-config.json              # Configuration for MCP server
│
├── 📁 src/                         # All source code lives here
│   └── repomind/           # Main package
│       │
│       ├── 📄 __init__.py          # Makes this a Python package
│       ├── 📄 cli.py               # ⭐ Command-line interface (start here!)
│       ├── 📄 server.py            # MCP server (connects to Claude)
│       ├── 📄 config.py            # Configuration settings
│       ├── 📄 constants.py         # All constant values
│       ├── 📄 logging.py           # Logging setup
│       │
│       ├── 📁 models/              # Data structures
│       │   ├── __init__.py
│       │   └── chunk.py            # ⭐ CodeChunk class (fundamental!)
│       │
│       ├── 📁 parsers/             # Code parsing (AST)
│       │   ├── __init__.py
│       │   ├── base.py             # BaseParser (all parsers extend this)
│       │   ├── python_parser.py   # Parses Python code
│       │   ├── java_parser.py     # Parses Java code
│       │   └── typescript_parser.py # Parses TypeScript/JavaScript
│       │
│       ├── 📁 services/            # Core business logic
│       │   ├── __init__.py
│       │   ├── chunking.py         # ⭐ Orchestrates parsing
│       │   ├── embedding.py        # ⭐ Creates embeddings
│       │   ├── storage.py          # ⭐ Stores in ChromaDB
│       │   ├── symbol_table.py     # Fast symbol lookup (SQLite)
│       │   ├── call_graph.py       # Tracks function calls
│       │   └── manifest.py         # Tracks indexed files
│       │
│       └── 📁 tools/               # MCP tools (exposed to Claude)
│           ├── __init__.py
│           ├── index_repo.py       # ⭐ Indexing tool
│           ├── semantic_grep.py    # ⭐ Search tool
│           ├── get_context.py      # Context retrieval
│           └── code_nav.py         # Code navigation
│
├── 📁 tests/                       # Test code (mirrors src/ structure)
│   ├── __init__.py
│   └── test_chunking.py            # Tests for chunking service
│
└── 📁 .venv/                       # Virtual environment (don't touch!)
```

### Key Files You'll Work With

#### 1. `cli.py` - The Entry Point
**What it does:** Defines all command-line commands (`index`, `search`, etc.)

**When to edit:** Adding new CLI commands

**Example:**
```python
@main.command()
def search(query):
    """Search code semantically."""
    # Your code here
```

#### 2. `models/chunk.py` - The Data Model
**What it does:** Defines what a "code chunk" is

**When to edit:** Adding new fields to chunks, changing how embedding text is generated

**Key method:** `to_embedding_text()` - converts chunk to searchable text

#### 3. `services/embedding.py` - The AI Part
**What it does:** Converts text to embeddings using BGE model

**When to edit:** Changing embedding models, adding query expansion rules

**Key method:** `embed_query()` - converts search queries to embeddings

#### 4. `services/storage.py` - The Database
**What it does:** Stores and searches embeddings in ChromaDB

**When to edit:** Changing search algorithm, adding filters

**Key method:** `search()` - performs hybrid semantic search

#### 5. `tools/semantic_grep.py` - The Search Tool
**What it does:** Orchestrates search (embedding → storage → formatting)

**When to edit:** Changing search output format, adding new filters

### How Files Connect

```
User types command
       ↓
    cli.py (parses command)
       ↓
    tools/semantic_grep.py (orchestrates)
       ↓
    services/embedding.py (creates embedding)
       ↓
    services/storage.py (searches ChromaDB)
       ↓
    Returns results
```

---

## How Code Flows Through the System

### Prerequisites
```
✓ Understanding of project structure
✓ Basic Python knowledge (functions, classes)
```

### 🎯 Goal
Follow a request from start to finish to understand how everything connects.

### Flow 1: Indexing a Repository

**User command:**
```bash
repomind index ~/my-project
```

**What happens (step by step):**

#### Step 1: CLI Receives Command
```python
# In cli.py
@main.command()
def index(repo_path):
    result = index_repo(repo_path)  # Calls tool
```

#### Step 2: Tool Orchestrates
```python
# In tools/index_repo.py
def index_repo(repo_path, repo_name):
    # 1. Find all source files
    # 2. Parse them
    # 3. Create embeddings
    # 4. Store everything
```

#### Step 3: Chunking Service Finds Files
```python
# In services/chunking.py
def chunk_repository_full(repo_path, repo_name):
    files = find_source_files(repo_path)  # Find .py, .java, .ts files
    for file in files:
        parser = get_parser(file)  # Python/Java/TypeScript parser
        chunks.extend(parser.parse(file))  # Extract functions, classes
```

#### Step 4: Parser Extracts Code
```python
# In parsers/python_parser.py
def parse(self, file_path):
    tree = parser.parse(file_content)  # Create AST
    functions = find_all_functions(tree)  # Walk the tree
    for func in functions:
        chunk = CodeChunk(
            name=func.name,
            content=func.source_code,
            type="function"
        )
        chunks.append(chunk)
```

#### Step 5: Embedding Service Creates Vectors
```python
# In services/embedding.py
def embed_chunks(chunks):
    texts = [chunk.to_embedding_text() for chunk in chunks]
    # Load BGE model (cached after first load)
    embeddings = model.encode(texts, normalize=True)
    return embeddings  # List of [0.1, 0.2, ...] arrays
```

#### Step 6: Storage Service Saves
```python
# In services/storage.py
def store_chunks(chunks, embeddings):
    # Save to ChromaDB
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk metadata]
    )
    # Also save full data to JSON
    save_metadata_to_file(chunks)
```

**Result:** All code is now indexed and searchable!

### Flow 2: Searching for Code

**User command:**
```bash
repomind search "authentication middleware"
```

**What happens:**

#### Step 1: CLI Receives Query
```python
# In cli.py
@main.command()
def search(query):
    result = semantic_grep(query)
```

#### Step 2: Semantic Grep Tool Orchestrates
```python
# In tools/semantic_grep.py
def semantic_grep(query, n_results=10):
    # 1. Expand query with synonyms
    # 2. Create embedding
    # 3. Search
    # 4. Format results
```

#### Step 3: Query Expansion
```python
# In services/embedding.py
def _expand_query(query):
    # "auth" → "authentication authorization login"
    expanded = query + " (authentication authorization login)"
    return expanded
```

#### Step 4: Embedding Creation
```python
# In services/embedding.py
def embed_query(query):
    expanded = _expand_query(query)
    # For BGE models, add instruction
    text = f"Represent this code search query: {expanded}"
    embedding = model.encode([text])[0]
    return embedding  # [0.1, 0.2, ...]
```

#### Step 5: Hybrid Search
```python
# In services/storage.py
def search(query_embedding, query_text, threshold=0.35):
    # 1. Vector similarity search
    results = chromadb.query(
        query_embeddings=[query_embedding],
        n_results=30  # Get more for filtering
    )
    
    # 2. Compute keyword boost
    for chunk, score in results:
        if "auth" in chunk.name.lower():
            score += 0.10  # Boost for name match
        if "middleware" in chunk.docstring:
            score += 0.05  # Boost for docstring match
    
    # 3. Filter by threshold
    results = [r for r in results if r.score >= 0.35]
    
    # 4. Sort and return top N
    return sorted(results, reverse=True)[:10]
```

#### Step 6: Format and Display
```python
# In tools/semantic_grep.py
# Display results in a table
table = Table()
for chunk, score in results:
    confidence = "🟢" if score >= 0.70 else "🟡"
    table.add_row(
        f"{confidence} {score:.2f}",
        chunk.chunk_type,
        chunk.name,
        f"{chunk.file}:{chunk.line}"
    )
console.print(table)
```

**Result:** User sees ranked results with confidence indicators!

---

## Making Your First Contribution

### Prerequisites
```
✓ Git basics (branch, commit, push)
✓ Understanding of code flow
✓ An idea for what to improve
```

### 🎯 Goal
Make a small, safe change to learn the workflow.

### Good First Issues

1. **Add a new query expansion rule** (Easy)
2. **Improve code chunk text formatting** (Easy)
3. **Add a new CLI option** (Medium)
4. **Add a new parser for a language** (Hard)

### Let's Do Example #1: Add Query Expansion Rule

**Goal:** When someone searches for "config", also search for "configuration settings options"

#### Step 1: Create a Branch
```bash
git checkout -b feature/add-config-expansion
```

#### Step 2: Find the Right File
Open `src/repomind/services/embedding.py`

#### Step 3: Find the Right Method
Search for `_expand_query` (around line 250)

#### Step 4: Make Your Change
```python
def _expand_query(self, query: str) -> str:
    """Expand query with code-related synonyms."""
    expansions = {
        "auth": "authentication authorization login",
        "error": "exception handling catch try",
        "config": "configuration settings options",  # ← ADD THIS LINE
        # ... existing code ...
    }
```

#### Step 5: Test Your Change Locally
```bash
# Activate virtual environment
source .venv/bin/activate

# Try a search with "config"
repomind search "config file" -n 5

# Check if it found configuration-related code
```

#### Step 6: Write a Test (Optional but Recommended)
```python
# In tests/test_embedding.py
def test_query_expansion_config():
    service = EmbeddingService()
    expanded = service._expand_query("config")
    assert "configuration" in expanded
    assert "settings" in expanded
```

#### Step 7: Run Tests
```bash
pytest tests/test_embedding.py -v
```

#### Step 8: Commit Your Change
```bash
git add src/repomind/services/embedding.py
git commit -m "feat: Add query expansion for 'config' keyword

- Maps 'config' to 'configuration settings options'
- Improves search accuracy for configuration-related code"
```

#### Step 9: Push and Create Pull Request
```bash
git push origin feature/add-config-expansion
```

Then create a PR on GitHub with:
- **Title:** "feat: Add query expansion for 'config' keyword"
- **Description:** What you changed and why
- **Testing:** How you tested it

✅ **Congratulations!** You've made your first contribution!

---

## Common Tasks and How to Do Them

### Prerequisites
```
✓ Project set up and running
✓ Basic understanding of code flow
```

### Task 1: Add a New Embedding Model

**When:** You want to support a different AI model

**Files to change:**
1. `constants.py` - Add model name and dimensions
2. `embedding.py` - Add model handling logic

**Example:**
```python
# In constants.py
class EmbeddingModelName(str, Enum):
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    MY_MODEL = "my-org/my-model"  # ← ADD THIS

EMBEDDING_DIMENSIONS = {
    EmbeddingModelName.BGE_BASE.value: 768,
    EmbeddingModelName.MY_MODEL.value: 512,  # ← ADD THIS
}
```

### Task 2: Add a New Language Parser

**When:** You want to index code in a new language (e.g., Rust, Go)

**Steps:**
1. Install tree-sitter grammar: `pip install tree-sitter-rust`
2. Create new parser file: `parsers/rust_parser.py`
3. Extend `BaseParser`
4. Register in `parsers/__init__.py`

**Example:**
```python
# In parsers/rust_parser.py
from .base import BaseParser
import tree_sitter_rust

class RustParser(BaseParser):
    def __init__(self):
        super().__init__(language=tree_sitter_rust.language())
    
    def _extract_functions(self, tree):
        # Query for Rust function nodes
        query = """
        (function_item
          name: (identifier) @name
          body: (block) @body
        )
        """
        # ... implementation
```

### Task 3: Change Similarity Threshold

**When:** Search returns too many/few results

**Files to change:**
1. `config.py` - Change default
2. Or use CLI: `--threshold 0.5`

**Example:**
```python
# In config.py
class SearchConfig(BaseModel):
    similarity_threshold: float = Field(
        default=0.40,  # ← Change from 0.35 to 0.40
        description="Minimum similarity score"
    )
```

### Task 4: Add a New CLI Command

**When:** You want to expose a new feature

**Files to change:**
1. `cli.py` - Add command
2. `tools/` - Create tool function (if new)

**Example:**
```python
# In cli.py
@main.command()
@click.argument("file_path")
def analyze(file_path):
    """Analyze a single file without indexing."""
    from .tools.analyze_file import analyze_single_file
    
    result = analyze_single_file(file_path)
    console.print_json(result)
```

### Task 5: Improve Keyword Matching

**When:** Hybrid search isn't finding obvious matches

**Files to change:**
1. `storage.py` - Modify `_compute_keyword_boost()`

**Example:**
```python
# In storage.py
def _compute_keyword_boost(self, query_text, chunk):
    # ... existing code ...
    
    # Add boost for signature matches
    if chunk.signature:
        for keyword in query_keywords:
            if keyword in chunk.signature.lower():
                boost += boost_factor * 1.5  # ← Higher weight
    
    return min(boost, 0.30)
```

---

### Common Call Graph Tasks

#### Task 1: Find Impact of Changing a Function

```python
from repomind.services.call_graph import CallGraphService

call_graph = CallGraphService()

# Find all callers (direct and indirect)
callers = call_graph.find_callers(
    symbol="EmailService.send",
    repo_name="my-app",
    max_depth=3  # 3 levels up
)

print(f"Changing EmailService.send affects {len(callers)} functions:")
for caller in callers:
    print(f"  - {caller.caller} ({caller.caller_file}:{caller.caller_line})")
```

#### Task 2: Trace Execution Path

```python
# Find path from API endpoint to database call
# Note: This requires implementing find_path method
callees = call_graph.find_callees(
    symbol="UserController.register",
    repo_name="my-app",
    max_depth=10
)

# Build execution path
path = ["UserController.register"]
for callee in callees:
    path.append(callee.callee)

print("Execution path:")
print(" → ".join(path))
```

#### Task 3: Find All Dead Code

```python
from repomind.services.storage import StorageService

storage = StorageService()
chunks = storage.get_all_chunks(repo_name="my-app")

# Identify entry points
entry_points = call_graph.identify_entry_points(chunks, repo_name="my-app")
print(f"Found {len(entry_points)} entry points")

# Find unreachable code
dead_code = call_graph.find_unreachable_from_entry_points(
    entry_points=entry_points,
    repo_name="my-app"
)

print(f"Found {len(dead_code)} unreachable functions:")
for func in dead_code:
    print(f"  - {func}")
```

#### Task 4: Analyze Module Dependencies

```python
# Find all functions that module A calls in module B
import sqlite3

conn = call_graph._get_connection()
cursor = conn.execute("""
    SELECT DISTINCT caller, callee, caller_line
    FROM calls
    WHERE caller_file LIKE '%module_a%'
      AND callee IN (
          SELECT DISTINCT caller FROM calls WHERE caller_file LIKE '%module_b%'
          UNION
          SELECT DISTINCT callee FROM calls WHERE caller_file LIKE '%module_b%'
      )
    ORDER BY caller
""")

print("Module A dependencies on Module B:")
for row in cursor.fetchall():
    print(f"  {row['caller']} → {row['callee']} (line {row['caller_line']})")

conn.close()
```

#### Task 5: Find Hotspots (Most Called Functions)

```python
# Find functions that are called the most
conn = call_graph._get_connection()
cursor = conn.execute("""
    SELECT callee, COUNT(*) as call_count
    FROM calls
    WHERE repo_name = ?
    GROUP BY callee
    ORDER BY call_count DESC
    LIMIT 10
""", ("my-app",))

print("Top 10 most called functions (hotspots):")
for i, row in enumerate(cursor.fetchall(), 1):
    print(f"  {i}. {row['callee']} ({row['call_count']} callers)")

conn.close()
```

---

### MCP Integration Examples

Use call graph features in VS Code with Copilot:

**Example 1: Find Callers**

```
You: @repomind show me all callers of UserService.save

Copilot: Found 8 callers of UserService.save:

Direct Callers:
1. UserController.createUser (src/controllers/user.py:45)
2. AdminController.updateUser (src/controllers/admin.py:89)

Indirect Callers (depth=2):
3. UserAPI.register → UserController.createUser
4. AdminAPI.modify → AdminController.updateUser

These are the functions that will be affected if you change UserService.save.
```

**Example 2: Find Dead Code**

```
You: @repomind find dead code in my-app

Copilot: Found 5 unreachable functions in my-app:

1. UserService.calculateAge_old (0 callers) - SAFE TO DELETE
   Last modified: 6 months ago
   
2. EmailService.sendFax (0 callers) - SAFE TO DELETE
   Last modified: 1 year ago
   
3. ReportService.generatePDF_v1 (0 callers) - SAFE TO DELETE
   Note: Replaced by generatePDF_v2

Recommendation: Remove these functions to reduce technical debt.
Would you like me to create a PR to remove them?
```

**Example 3: Detect Cycles**

```
You: @repomind detect circular dependencies

Copilot: Found 2 circular dependencies:

Cycle 1: user_service.py ↔ order_service.py
  - UserService.getOrders calls OrderService.getUserOrders
  - OrderService.getUserInfo calls UserService.getProfile
  
  Recommendation: Extract shared user data logic to user_repository.py

Cycle 2: auth/validator.py ↔ auth/token.py
  - Validator.validateToken calls Token.decode
  - Token.validate calls Validator.checkExpiry
  
  Recommendation: Move checkExpiry to token.py to break the cycle

Would you like me to show you how to refactor these?
```

---

## Testing Advanced Features

### Testing Call Graph Service

```python
# tests/test_call_graph_advanced.py
import pytest
from repomind.services.call_graph import CallGraphService, CallRelation
from repomind.services.symbol_table import SymbolTableService, Symbol

@pytest.fixture
def call_graph():
    """Create a fresh call graph for testing."""
    return CallGraphService()

@pytest.fixture
def symbol_table():
    """Create a fresh symbol table for testing."""
    return SymbolTableService()

def test_transitive_callers(call_graph):
    """Test finding indirect callers (A→B→C, A is transitive caller of C)."""
    # Create call chain: A → B → C
    call_graph.add_calls_bulk([
        CallRelation("A", "B", "file.py", 1, "test", "direct"),
        CallRelation("B", "C", "file.py", 2, "test", "direct"),
        CallRelation("D", "C", "file.py", 3, "test", "direct"),
    ])
    
    # Find all callers of C (should include both B and A)
    callers = call_graph.find_callers("C", repo_name="test", max_depth=2)
    caller_names = [c.caller for c in callers]
    
    assert "B" in caller_names  # Direct caller
    assert "A" in caller_names  # Transitive caller
    assert "D" in caller_names  # Another direct caller

def test_polymorphic_resolution(call_graph, symbol_table):
    """Test resolving calls through interfaces."""
    # Add inheritance: UserServiceImpl implements UserService
    symbol_table.add_symbol(Symbol(
        name="UserService",
        qualified_name="UserService",
        symbol_type="interface",
        file_path="src/service.py",
        repo_name="test",
        start_line=1,
        end_line=10
    ))
    
    symbol_table.add_symbol(Symbol(
        name="UserServiceImpl",
        qualified_name="UserServiceImpl",
        symbol_type="class",
        file_path="src/impl.py",
        repo_name="test",
        start_line=1,
        end_line=20,
        parent_name="UserService"
    ))
    
    # Add call to interface method
    call_graph.add_call(CallRelation(
        "Controller.create",
        "UserService.save",
        "src/controller.py",
        10,
        "test",
        "direct"
    ))
    
    # Find callees (should include implementation)
    callees = call_graph.find_callees_polymorphic(
        "Controller.create",
        repo_name="test",
        symbol_table=symbol_table
    )
    
    callee_names = [c.callee for c in callees]
    assert "UserService.save" in callee_names
    assert "UserServiceImpl.save" in callee_names

def test_entry_point_identification(call_graph):
    """Test identifying entry points from chunks."""
    from repomind.models.chunk import CodeChunk, ChunkType
    
    chunks = [
        # API route (entry point)
        CodeChunk(
            content="def get_users():",
            type=ChunkType.FUNCTION,
            file_path="src/api.py",
            name="get_users",
            repo_name="test",
            metadata={"route": "/users", "method": "GET"}
        ),
        # Main function (entry point)
        CodeChunk(
            content="def main():",
            type=ChunkType.FUNCTION,
            file_path="src/main.py",
            name="main",
            repo_name="test"
        ),
        # Regular function (not entry point)
        CodeChunk(
            content="def helper():",
            type=ChunkType.FUNCTION,
            file_path="src/utils.py",
            name="helper",
            repo_name="test"
        ),
        # Test function (entry point)
        CodeChunk(
            content="def test_user_creation():",
            type=ChunkType.FUNCTION,
            file_path="tests/test_user.py",
            name="test_user_creation",
            repo_name="test"
        ),
    ]
    
    entry_points = call_graph.identify_entry_points(chunks, repo_name="test")
    
    assert "get_users" in entry_points
    assert "main" in entry_points
    assert "test_user_creation" in entry_points
    assert "helper" not in entry_points

def test_dead_code_detection(call_graph):
    """Test finding unreachable code."""
    from repomind.models.chunk import CodeChunk, ChunkType
    
    # Build call graph: main → used_func
    call_graph.add_calls_bulk([
        CallRelation("main", "used_func", "main.py", 5, "test", "direct"),
        CallRelation("used_func", "helper", "utils.py", 10, "test", "direct"),
    ])
    
    # Create chunks
    chunks = [
        CodeChunk(content="def main():", type=ChunkType.FUNCTION,
                 file_path="main.py", name="main", repo_name="test"),
        CodeChunk(content="def used_func():", type=ChunkType.FUNCTION,
                 file_path="main.py", name="used_func", repo_name="test"),
        CodeChunk(content="def helper():", type=ChunkType.FUNCTION,
                 file_path="utils.py", name="helper", repo_name="test"),
        CodeChunk(content="def dead_code():", type=ChunkType.FUNCTION,
                 file_path="utils.py", name="dead_code", repo_name="test"),
    ]
    
    entry_points = ["main"]
    unreachable = call_graph.find_unreachable_from_entry_points(
        entry_points=entry_points,
        repo_name="test"
    )
    
    assert "dead_code" in unreachable
    assert "used_func" not in unreachable
    assert "helper" not in unreachable

def test_cycle_detection(call_graph):
    """Test detecting circular dependencies."""
    # Create a cycle: A → B → C → A
    call_graph.add_calls_bulk([
        CallRelation("A", "B", "file1.py", 1, "test", "direct"),
        CallRelation("B", "C", "file2.py", 1, "test", "direct"),
        CallRelation("C", "A", "file3.py", 1, "test", "direct"),
    ])
    
    # Also create a non-cyclic path: D → E
    call_graph.add_calls_bulk([
        CallRelation("D", "E", "file4.py", 1, "test", "direct"),
    ])
    
    cycles = call_graph.detect_cycles(repo_name="test", group_by_file=False)
    
    assert len(cycles) == 1
    cycle = cycles[0]
    assert set(cycle) == {"A", "B", "C"}

def test_dependency_injection(call_graph):
    """Test DI resolution."""
    # Add class dependencies
    call_graph.add_class_dependencies(
        class_name="UserController",
        dependencies=[
            {"name": "userService", "type": "UserService", "injection_type": "field"},
            {"name": "emailService", "type": "EmailService", "injection_type": "constructor"}
        ],
        repo_name="test"
    )
    
    # Test resolution
    resolved = call_graph.resolve_callee("UserController", "userService.save", "test")
    assert resolved == "UserService.save"
    
    resolved = call_graph.resolve_callee("UserController", "emailService.send", "test")
    assert resolved == "EmailService.send"
    
    # Test non-DI field (should return as-is)
    resolved = call_graph.resolve_callee("UserController", "this.method", "test")
    assert resolved == "this.method"
```

---

## Testing Your Changes

### Prerequisites
```
✓ Change made to code
✓ pytest installed (included in dev dependencies)
```

### 🎯 Goal
Verify your changes work and don't break anything.

### Types of Testing

#### 1. Manual Testing (Quick)

**For search changes:**
```bash
# Index a small test repository
repomind index ~/test-repo

# Try your change
repomind search "your test query"

# Verify output looks correct
```

**For parsing changes:**
```bash
# Create a test file with specific code
echo 'def test(): pass' > test.py

# Index it
repomind index . --name test

# Search for it
repomind search "test function"
```

#### 2. Automated Tests (Proper)

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_chunking.py -v
```

**Run specific test function:**
```bash
pytest tests/test_chunking.py::test_chunk_python_file -v
```

### Writing a Simple Test

**Example:** Test query expansion

```python
# In tests/test_embedding.py
from repomind.services.embedding import EmbeddingService

def test_query_expansion():
    """Test that queries are expanded with synonyms."""
    service = EmbeddingService()
    
    # Test input
    query = "auth middleware"
    
    # Call method
    expanded = service._expand_query(query)
    
    # Assert expectations
    assert "authentication" in expanded
    assert "authorization" in expanded
    assert "middleware" in expanded  # Original word kept
```

### Understanding Test Output

**Success:**
```
tests/test_embedding.py::test_query_expansion PASSED [100%]
```

**Failure:**
```
tests/test_embedding.py::test_query_expansion FAILED [100%]
AssertionError: assert 'authentication' in 'auth middleware'
```

**What to do:** Fix your code or test, then run again.

### Testing Checklist

Before submitting a PR:
- [ ] Manual test: Does it work for you?
- [ ] Unit tests: Do all tests pass?
- [ ] New tests: Did you add tests for new code?
- [ ] Edge cases: What if input is empty? Invalid?

---

## Debugging Guide

### Prerequisites
```
✓ Code that's not working as expected
✓ Patience!
```

### 🎯 Goal
Find and fix bugs efficiently.

### Debugging Technique 1: Print Statements

**Simplest method:** Add print statements to see what's happening.

```python
# In embedding.py
def embed_query(self, query):
    print(f"🐛 Original query: {query}")
    
    expanded = self._expand_query(query)
    print(f"🐛 Expanded query: {expanded}")
    
    embedding = self.model.encode([expanded])[0]
    print(f"🐛 Embedding shape: {len(embedding)}")
    
    return embedding
```

Then run your code and watch the output.

### Debugging Technique 2: Python Debugger (pdb)

**More powerful:** Step through code line by line.

```python
# In any file
def some_function():
    x = 10
    import pdb; pdb.set_trace()  # ← Execution stops here
    y = x * 2
    return y
```

**Commands in pdb:**
- `n` (next): Execute next line
- `s` (step): Step into function
- `c` (continue): Continue until next breakpoint
- `p variable`: Print variable value
- `l` (list): Show current code
- `q` (quit): Exit debugger

### Debugging Technique 3: IDE Debugger

**Best for complex issues:** Visual debugging.

**In PyCharm:**
1. Click left margin to set breakpoint (red dot)
2. Right-click file → Debug
3. Use toolbar buttons to step through

**In VS Code:**
1. Click left margin for breakpoint
2. Press F5 or Run → Start Debugging
3. Use toolbar to step

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError"
```
ModuleNotFoundError: No module named 'chromadb'
```

**Solution:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -e ".[dev]"
```

#### Issue: "Search returns 0 results"
```bash
# Check if anything is indexed
repomind stats

# If empty, index something
repomind index ~/test-repo
```

#### Issue: "Embedding model fails to load"
```
OSError: [model-name] is not a local folder
```

**Solution:**
```python
# In config.py or CLI
# Make sure model name is correct
model = "BAAI/bge-base-en-v1.5"  # ✅ Correct
model = "EmbeddingModel.LOCAL_BGE"  # ❌ Wrong
```

#### Issue: "Tests fail after my change"

**Steps:**
1. Read error message carefully
2. Find which test failed
3. Run just that test: `pytest tests/test_file.py::test_name -v`
4. Add print statements to see what's happening
5. Fix code or update test

---

## Best Practices

### Prerequisites
```
✓ Experience making at least one contribution
```

### 🎯 Goal
Write code that's maintainable, readable, and follows project standards.

### Code Style

#### 1. Use Type Hints
```python
# ❌ Bad
def process_chunks(chunks):
    return [c.name for c in chunks]

# ✅ Good
def process_chunks(chunks: list[CodeChunk]) -> list[str]:
    """Extract names from code chunks."""
    return [c.name for c in chunks]
```

#### 2. Write Docstrings
```python
# ✅ Good
def search(query: str, n_results: int = 10) -> dict:
    """
    Search for code semantically.
    
    Args:
        query: Natural language search query
        n_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results and metadata
        
    Example:
        >>> search("authentication code", n_results=5)
        {"results": [...], "total": 5}
    """
    # Implementation
```

#### 3. Use Descriptive Names
```python
# ❌ Bad
def proc(x):
    return x * 2

# ✅ Good
def calculate_boosted_score(base_score: float) -> float:
    """Double the base similarity score."""
    return base_score * 2
```

#### 4. Keep Functions Small
```python
# ❌ Bad - one function doing everything
def index_and_search_and_display(repo, query):
    # 100 lines of code...
    pass

# ✅ Good - small, focused functions
def index_repository(repo_path: Path) -> IndexResult:
    """Index a repository."""
    # 20 lines

def search_code(query: str) -> SearchResult:
    """Search indexed code."""
    # 15 lines

def display_results(results: SearchResult) -> None:
    """Display search results."""
    # 10 lines
```

### Git Practices

#### 1. Write Good Commit Messages
```bash
# ❌ Bad
git commit -m "fixed stuff"

# ✅ Good
git commit -m "fix: Correct keyword boost calculation in hybrid search

- Fixed double-counting of keywords in docstrings
- Added test case for edge case
- Improves search precision by ~5%"
```

#### 2. One Feature Per Branch
```bash
# Create focused branches
git checkout -b fix/search-threshold
git checkout -b feat/add-rust-parser
git checkout -b docs/improve-developer-guide
```

#### 3. Pull Before Push
```bash
# Update your branch with latest changes
git checkout main
git pull origin main
git checkout your-branch
git rebase main
```

### Performance Considerations

#### 1. Don't Load Large Data in Loops
```python
# ❌ Bad
for chunk_id in chunk_ids:
    chunk = load_full_chunk(chunk_id)  # Database hit each time!
    process(chunk)

# ✅ Good
chunks = load_all_chunks(chunk_ids)  # One database hit
for chunk in chunks:
    process(chunk)
```

#### 2. Use Caching
```python
# ✅ Good - model is cached
_model_cache = {}

def get_model(name):
    if name not in _model_cache:
        _model_cache[name] = load_model(name)
    return _model_cache[name]
```

#### 3. Batch Operations
```python
# ❌ Bad
for chunk in chunks:
    embedding = embed(chunk)  # Slow, one at a time
    store(embedding)

# ✅ Good
embeddings = embed_batch(chunks)  # Fast, batched
store_batch(embeddings)
```

---

## Getting Help

### Prerequisites
```
✓ You've tried to solve the problem yourself
✓ You can describe what's not working
```

### 🎯 Goal
Get unblocked quickly without wasting anyone's time.

### Before Asking for Help

1. **Read the error message** - Often tells you exactly what's wrong
2. **Check this guide** - Your answer might be here
3. **Search the codebase** - Similar code might exist
4. **Google the error** - Someone else probably had it

### How to Ask Good Questions

#### ❌ Bad Question
"It doesn't work, help!"

#### ✅ Good Question
**Subject:** "Search returns 0 results after adding new expansion rule"

**Body:**
```
What I'm trying to do:
- Add query expansion for "database" keyword
- Modified _expand_query() in embedding.py

What I expected:
- Search for "database" should find DB-related code

What actually happened:
- Search returns 0 results
- Stats shows 100 chunks indexed

What I've tried:
- Verified code is indexed (repomind stats)
- Printed expanded query - it shows correctly
- Tested with other queries - they work

Code change:
[paste your change]

Error/Output:
[paste relevant output]
```

### Where to Ask

1. **Team Chat** - For quick questions
2. **GitHub Issues** - For bugs
3. **GitHub Discussions** - For feature ideas
4. **Code Comments** - For specific code questions

### How to Help Others

When someone asks for help:
1. **Be kind** - We all start somewhere
2. **Ask clarifying questions** - Understand the problem
3. **Point to resources** - Teach them to fish
4. **Share knowledge** - Explain your thinking

---

## Route Analysis and API Endpoint Tracking

**What is Route Analysis?**

RepoMind can track and link API endpoints between frontend and backend code:
- Find which backend endpoint handles a frontend API call
- Detect route conflicts (multiple endpoints with same path)
- Track API contract changes

**Why Is This Useful?**

1. **Frontend-Backend Linking**: "Which backend handler serves this API call?"
2. **Contract Validation**: "Are frontend and backend in sync?"
3. **Impact Analysis**: "If I change this endpoint, which frontend code breaks?"
4. **Conflict Detection**: "Do I have duplicate routes?"

---

### How Route Analysis Works

**Step 1: Normalize Routes**

Different frameworks use different syntax. RepoMind normalizes them:

```python
# FastAPI (Python)
@app.get("/users/{user_id}")
# → Normalized: GET /users/{user_id}

# Spring (Java)
@GetMapping("/users/{userId}")
# → Normalized: GET /users/{userId}

# Express (JavaScript)
app.get('/users/:userId')
# → Normalized: GET /users/{userId}

# NestJS (TypeScript)
@Get('/users/:userId')
# → Normalized: GET /users/{userId}
```

**Step 2: Extract Parameters**

```python
from repomind.models.route import RouteParameter, ParameterSource

# For route: /users/{userId}/orders/{orderId}?limit=10
parameters = [
    RouteParameter(
        name="userId",
        position=1,
        source=ParameterSource.PATH,
        required=True
    ),
    RouteParameter(
        name="orderId",
        position=3,
        source=ParameterSource.PATH,
        required=True
    ),
    RouteParameter(
        name="limit",
        position=-1,  # -1 for query params
        source=ParameterSource.QUERY,
        required=False
    ),
]
```

**Step 3: Link Frontend Calls to Backend Endpoints**

```python
from repomind.services.route_linker import RouteLinker

linker = RouteLinker()

# Match frontend call to backend endpoint
matches = linker.find_endpoint_matches(
    api_call=frontend_call,
    repo_name="my-app"
)

# Returns RouteMatch objects with confidence scores
for match in matches:
    print(f"Match: {match.endpoint.qualified_name}")
    print(f"Confidence: {match.confidence:.2%}")
    if match.is_ambiguous:
        print(f"Warning: {match.alternatives} other endpoints match!")
```

---

### Route Data Models

**1. RouteEndpoint (Backend)**

```python
from repomind.models.route import RouteEndpoint, NormalizedRoute

endpoint = RouteEndpoint(
    id="endpoint_123",
    method="GET",
    normalized_route=NormalizedRoute(
        method="GET",
        path_pattern="/users/{userId}",
        path_segments=["users", "{userId}"],
        parameters=[...],
        framework="fastapi"
    ),
    repo_name="backend-api",
    file_path="src/api/users.py",
    qualified_name="UserAPI.get_user",
    language="python",
    framework="fastapi",
    chunk_id="chunk_789"
)
```

**2. ApiCall (Frontend)**

```python
from repomind.models.route import ApiCall

api_call = ApiCall(
    id="call_456",
    method="GET",
    url="https://api.example.com/users/123",
    normalized_path="/users/{userId}",
    repo_name="frontend-app",
    file_path="src/services/UserService.ts",
    line_number=42,
    caller_qualified_name="UserService.fetchUser",
    metadata={"caller_function": "fetchUser"}
)
```

**3. RouteMatch (Link)**

```python
from repomind.models.route import RouteMatch

match = RouteMatch(
    call=api_call,
    endpoint=endpoint,
    confidence=0.95,  # 95% confidence
    is_ambiguous=False,  # No other endpoints match
    alternatives=0,
    method_match=True,
    path_match=True,
    param_match=True
)
```

---

### Practical Examples: Route Analysis

**Example 1: Find Backend Handler for Frontend Call**

```python
from repomind.services.route_registry import RouteRegistry
from repomind.services.route_linker import RouteLinker

registry = RouteRegistry()
linker = RouteLinker()

# Get all API calls from frontend
calls = registry.get_all_api_calls(repo_name="frontend-app")

for call in calls:
    matches = linker.find_endpoint_matches(call, repo_name="backend-api")
    
    if not matches:
        print(f"⚠️ No backend handler for: {call.method} {call.normalized_path}")
        print(f"   Called from: {call.file_path}:{call.line_number}")
    elif len(matches) > 1:
        print(f"⚠️ Multiple handlers for: {call.method} {call.normalized_path}")
        for match in matches:
            print(f"   - {match.endpoint.qualified_name} ({match.confidence:.0%})")
    else:
        match = matches[0]
        print(f"✅ {call.method} {call.normalized_path}")
        print(f"   → {match.endpoint.qualified_name}")
```

**Example 2: Detect Route Conflicts**

```python
# Find endpoints with same path but different handlers
conflicts = registry.detect_conflicts(repo_name="backend-api")

for conflict in conflicts:
    print(f"⚠️ Route conflict: {conflict.method} {conflict.path_pattern}")
    print(f"   Found in {len(conflict.endpoints)} places:")
    for endpoint in conflict.endpoints:
        print(f"   - {endpoint.qualified_name} ({endpoint.file_path})")
```

**Example 3: Track API Contract Changes**

```python
# Compare two versions of the API
old_endpoints = registry.get_all_endpoints(repo_name="backend-api-v1")
new_endpoints = registry.get_all_endpoints(repo_name="backend-api-v2")

# Find removed endpoints
old_paths = {(e.method, e.normalized_route.path_pattern) for e in old_endpoints}
new_paths = {(e.method, e.normalized_route.path_pattern) for e in new_endpoints}

removed = old_paths - new_paths
added = new_paths - old_paths

print("API Changes:")
print(f"  Removed: {len(removed)} endpoints")
for method, path in removed:
    print(f"    - {method} {path}")

print(f"  Added: {len(added)} endpoints")
for method, path in added:
    print(f"    + {method} {path}")
```

**Example 4: Find All Callers of an Endpoint**

```python
# Find all frontend code that calls a specific backend endpoint
endpoint = registry.get_endpoint_by_path(
    method="POST",
    path_pattern="/users/{userId}",
    repo_name="backend-api"
)

if endpoint:
    links = registry.get_links_for_endpoint(endpoint.id)
    
    print(f"Frontend callers of {endpoint.qualified_name}:")
    for link in links:
        call = registry.get_api_call(link.call_id)
        print(f"  - {call.file_path}:{call.line_number}")
        print(f"    Function: {call.caller_qualified_name}")
```

---

### Supported Frameworks

**Backend Frameworks:**

1. **FastAPI (Python)**
   ```python
   @app.get("/users/{user_id}")
   @app.post("/users")
   @router.put("/users/{user_id}")
   ```

2. **Spring Boot (Java)**
   ```java
   @GetMapping("/users/{userId}")
   @PostMapping("/users")
   @RequestMapping(value = "/users/{userId}", method = RequestMethod.PUT)
   ```

3. **Express (JavaScript)**
   ```javascript
   app.get('/users/:userId')
   app.post('/users')
   router.put('/users/:userId')
   ```

4. **NestJS (TypeScript)**
   ```typescript
   @Get('/users/:userId')
   @Post('/users')
   @Put('/users/:userId')
   ```

**Frontend Frameworks:**

1. **Axios (JavaScript/TypeScript)**
   ```javascript
   axios.get('/users/123')
   axios.post('/users', data)
   ```

2. **Fetch API**
   ```javascript
   fetch('/users/123')
   fetch('/users', { method: 'POST', body: data })
   ```

3. **React Query**
   ```typescript
   useQuery(['user', userId], () => fetch(`/users/${userId}`))
   ```

---

### Testing Route Analysis

```python
# tests/test_route_analysis.py
import pytest
from repomind.models.route import (
    RouteEndpoint, ApiCall, NormalizedRoute, 
    RouteParameter, ParameterSource
)
from repomind.services.route_registry import RouteRegistry
from repomind.services.route_linker import RouteLinker

@pytest.fixture
def registry():
    """Create a fresh route registry for testing."""
    return RouteRegistry()

@pytest.fixture
def linker():
    """Create a route linker for testing."""
    return RouteLinker()

def test_route_normalization():
    """Test that different framework syntaxes normalize to same pattern."""
    from repomind.services.route_normalizer import RouteNormalizer
    
    normalizer = RouteNormalizer()
    
    # FastAPI syntax
    fastapi_route = normalizer.normalize("/users/{user_id}", "GET", "fastapi")
    
    # Spring syntax
    spring_route = normalizer.normalize("/users/{userId}", "GET", "spring")
    
    # Express syntax
    express_route = normalizer.normalize("/users/:userId", "GET", "express")
    
    # All should normalize to similar patterns
    assert fastapi_route.path_pattern == "/users/{user_id}"
    assert spring_route.path_pattern == "/users/{userId}"
    assert express_route.path_pattern == "/users/{userId}"

def test_endpoint_registration(registry):
    """Test registering a backend endpoint."""
    endpoint = RouteEndpoint(
        id="test_endpoint",
        method="GET",
        normalized_route=NormalizedRoute(
            method="GET",
            path_pattern="/users/{userId}",
            path_segments=["users", "{userId}"],
            parameters=[
                RouteParameter("userId", 1, ParameterSource.PATH)
            ],
            framework="fastapi"
        ),
        repo_name="test-backend",
        file_path="api/users.py",
        qualified_name="UserAPI.get_user",
        language="python",
        framework="fastapi"
    )
    
    endpoint_id = registry.register_endpoint(endpoint)
    assert endpoint_id == "test_endpoint"
    
    # Retrieve and verify
    retrieved = registry.get_endpoint(endpoint_id)
    assert retrieved.method == "GET"
    assert retrieved.qualified_name == "UserAPI.get_user"

def test_call_endpoint_matching(registry, linker):
    """Test matching a frontend call to backend endpoint."""
    # Register endpoint
    endpoint = RouteEndpoint(
        id="endpoint_1",
        method="GET",
        normalized_route=NormalizedRoute(
            method="GET",
            path_pattern="/users/{userId}",
            path_segments=["users", "{userId}"],
            parameters=[RouteParameter("userId", 1, ParameterSource.PATH)],
            framework="spring"
        ),
        repo_name="backend",
        file_path="UserController.java",
        qualified_name="UserController.getUser",
        language="java",
        framework="spring"
    )
    registry.register_endpoint(endpoint)
    
    # Create frontend call
    call = ApiCall(
        id="call_1",
        method="GET",
        url="/users/123",
        normalized_path="/users/{userId}",
        repo_name="frontend",
        file_path="UserService.ts",
        line_number=42,
        caller_qualified_name="UserService.fetchUser"
    )
    
    # Find matches
    matches = linker.find_endpoint_matches(call, repo_name="backend")
    
    assert len(matches) > 0
    assert matches[0].endpoint.qualified_name == "UserController.getUser"
    assert matches[0].confidence > 0.8

def test_conflict_detection(registry):
    """Test detecting route conflicts."""
    # Register two endpoints with same path
    for i in range(2):
        endpoint = RouteEndpoint(
            id=f"endpoint_{i}",
            method="GET",
            normalized_route=NormalizedRoute(
                method="GET",
                path_pattern="/users/{userId}",
                path_segments=["users", "{userId}"],
                framework="spring"
            ),
            repo_name="backend",
            file_path=f"controller{i}.java",
            qualified_name=f"Controller{i}.getUser",
            language="java",
            framework="spring"
        )
        registry.register_endpoint(endpoint)
    
    # Detect conflicts
    conflicts = registry.detect_conflicts(repo_name="backend")
    
    assert len(conflicts) > 0
    assert conflicts[0].method == "GET"
    assert len(conflicts[0].endpoints) == 2
```

---

## Advanced Features Explained

### Prerequisites
```
✓ Understanding of basic RepoMind concepts
✓ Familiarity with project structure
✓ Knowledge of functions and classes
```

### 🎯 Goal
Master advanced features like call graphs, dead code detection, and dependency analysis.

---

### Call Graph Service (Finding Code Relationships)

**What is a Call Graph?**

Think of it like a phone directory that shows:
- "Who calls this function?" (callers)
- "What does this function call?" (callees)

**Real-World Example:**

```python
# File: user_service.py
def save_user(user):
    validate_email(user.email)  # save_user CALLS validate_email
    send_welcome_email(user)     # save_user CALLS send_welcome_email
```

In this example:
- `save_user` is the **caller**
- `validate_email` and `send_welcome_email` are **callees**

**Why Is This Useful?**

1. **Impact Analysis**: "If I change this function, what else breaks?"
2. **Understanding Code Flow**: "How does data move through the system?"
3. **Finding Dead Code**: "Is anyone using this function?"
4. **Detecting Cycles**: "Are there circular dependencies?"

---

### How Call Graph Works

**Step 1: Capturing Calls**

When RepoMind indexes your code, it finds every function call:

```python
# What RepoMind sees:
CallRelation(
    caller="UserService.save_user",
    callee="EmailService.validate_email",
    caller_file="src/user_service.py",
    caller_line=42,
    repo_name="my-app"
)
```

**Step 2: Storing in Database**

All relationships are stored in SQLite:

```sql
-- Simplified schema
CREATE TABLE calls (
    caller TEXT,      -- "UserService.save_user"
    callee TEXT,      -- "EmailService.validate_email"
    caller_file TEXT, -- "src/user_service.py"
    caller_line INT,  -- 42
    call_type TEXT    -- "direct", "virtual", etc.
);
```

**Step 3: Querying Relationships**

```python
# Find all functions that call validate_email
callers = call_graph.find_callers("EmailService.validate_email")
# Returns: ["UserService.save_user", "AdminService.update_user", ...]

# Find all functions called by save_user
callees = call_graph.find_callees("UserService.save_user")
# Returns: ["EmailService.validate_email", "EmailService.send_welcome_email", ...]
```

---

### Dependency Injection Resolution

**What is Dependency Injection?**

Instead of creating objects yourself, the framework gives them to you:

```java
// Without DI (bad - tightly coupled)
class UserController {
    private UserService service = new UserService();
}

// With DI (good - flexible)
class UserController {
    @Autowired
    private UserService service;  // Framework injects this
}
```

**The Problem for Call Graphs**

When you see this code:
```java
userService.save(user);
```

RepoMind needs to know:
- What is `userService`? 
- What class is it an instance of?
- So it can resolve: `UserService.save` (not just `userService.save`)

**How RepoMind Solves This**

**Step 1: Store Dependencies**

```python
# RepoMind saves: "UserController has a field called userService of type UserService"
call_graph.add_class_dependencies(
    class_name="UserController",
    dependencies=[
        {
            "name": "userService",
            "type": "UserService",
            "injection_type": "field"
        }
    ],
    repo_name="my-app"
)
```

**Step 2: Resolve Calls**

```python
# When indexing: userService.save(user)
caller_class = "UserController"
callee = "userService.save"

# RepoMind resolves it:
resolved = call_graph.resolve_callee(
    caller_class="UserController",
    callee="userService.save"
)
# Returns: "UserService.save" ✅
```

---

### Polymorphic Call Resolution

**What is Polymorphism?**

Multiple classes can implement the same interface:

```java
// Interface
interface Repository {
    void save(Object data);
}

// Implementations
class DatabaseRepository implements Repository {
    void save(Object data) { /* save to DB */ }
}

class CacheRepository implements Repository {
    void save(Object data) { /* save to cache */ }
}

// Usage
class UserService {
    @Autowired
    private Repository repo;  // Could be either implementation!
    
    void saveUser(User user) {
        repo.save(user);  // Which save() gets called?
    }
}
```

**The Challenge**

When you call `repo.save(user)`:
- You're calling `Repository.save` (interface)
- But at runtime, it could be:
  - `DatabaseRepository.save`
  - `CacheRepository.save`
  - Any other implementation!

**How RepoMind Handles This**

**Step 1: Find Interface Calls**

```python
# Find direct calls
callees = call_graph.find_callees("UserService.saveUser")
# Returns: ["Repository.save"]
```

**Step 2: Find All Implementations**

```python
# Find polymorphic calls (includes implementations)
callees = call_graph.find_callees_polymorphic(
    "UserService.saveUser",
    symbol_table=symbol_table
)
# Returns: [
#   "Repository.save",           # Original interface call
#   "DatabaseRepository.save",   # Implementation 1
#   "CacheRepository.save"       # Implementation 2
# ]
```

This gives you **complete** call information!

---

### Dead Code Detection

**What is Dead Code?**

Code that exists but is never executed:

```python
# This function is never called by anyone
def calculate_tax_1995_rules(amount):
    return amount * 0.15  # Old tax rate, no longer used
```

**Why Remove It?**

- **Confuses developers**: "Should I update this too?"
- **Security risk**: Old code may have vulnerabilities
- **Maintenance burden**: You're testing code that's never run
- **Slower codebase**: More code to read through

**How RepoMind Detects It**

**Method 1: Find Functions With No Callers**

```python
# Find functions that are never called
unused = call_graph.find_symbols_with_no_callers(repo_name="my-app")
# Returns: ["calculate_tax_1995_rules", "old_validation_logic", ...]
```

⚠️ **Warning**: This might include:
- Entry points (main functions, API routes)
- Test functions
- Functions called by external code

**Method 2: Entry Point Analysis (More Accurate)**

```python
# Step 1: Identify entry points
entry_points = call_graph.identify_entry_points(chunks, repo_name="my-app")
# Returns: ["main", "UserController.getUser", "OrderController.create", ...]

# Step 2: Find unreachable code
dead_code = call_graph.find_unreachable_from_entry_points(
    entry_points=entry_points,
    repo_name="my-app"
)
# Returns: Only functions that can't be reached from ANY entry point
```

**What Are Entry Points?**

Functions that are called by external code:

1. **Main Functions**: `def main()`, `if __name__ == "__main__"`
2. **API Routes**: Functions with `@app.route`, `@RestController`
3. **React Pages**: Components in `pages/` or `app/` directory
4. **Test Functions**: `test_*`, `Test*` classes

**Example Walkthrough**

```python
# Entry point (API route)
@app.route("/users")
def get_users():
    users = fetch_all_users()  # Called by get_users
    return users

def fetch_all_users():
    return query_database()  # Called by fetch_all_users

def query_database():
    pass  # Called by fetch_all_users

def old_user_sync():  # ❌ Never called by anyone
    pass
```

**Reachability Analysis:**
- ✅ `get_users` - Entry point
- ✅ `fetch_all_users` - Reachable from `get_users`
- ✅ `query_database` - Reachable from `fetch_all_users`
- ❌ `old_user_sync` - Not reachable from any entry point → **Dead Code**

---

### Circular Dependency Detection

**What is a Circular Dependency?**

When code depends on itself (directly or indirectly):

```python
# File: user_service.py
from order_service import get_user_orders

def get_user_profile(user_id):
    orders = get_user_orders(user_id)  # Calls order_service
    return {"user": user_id, "orders": orders}

# File: order_service.py
from user_service import get_user_profile

def get_user_orders(user_id):
    profile = get_user_profile(user_id)  # Calls user_service ⚠️ CYCLE!
    return profile["orders"]
```

**The Problem:**

```
user_service → order_service → user_service → order_service → ...
```

This creates:
- **Import errors** in Python
- **Circular initialization** issues
- **Hard-to-understand code** flow
- **Testing difficulties** (can't mock one without the other)

**Types of Cycles**

**1. Function-Level Cycle**

```python
def func_a():
    func_b()  # A calls B

def func_b():
    func_a()  # B calls A ⚠️ CYCLE
```

**2. File-Level Cycle**

```python
# user.py imports order.py
# order.py imports user.py
# ⚠️ CYCLE
```

**How RepoMind Detects Cycles**

Uses **Tarjan's Algorithm** (don't worry about the math, just the concept):

```python
# Find function-level cycles
cycles = call_graph.detect_cycles(
    repo_name="my-app",
    group_by_file=False
)
# Returns: [
#   ["UserService.getProfile", "OrderService.getUserOrders"],
#   ["func_a", "func_b"]
# ]

# Find file-level cycles
cycles = call_graph.detect_cycles(
    repo_name="my-app",
    group_by_file=True
)
# Returns: [
#   ["src/user_service.py", "src/order_service.py"]
# ]
```

**How to Fix Cycles**

**Solution 1: Extract Common Code**

```python
# Before (cycle):
# user_service.py → order_service.py → user_service.py

# After (no cycle):
# user_service.py → common.py
# order_service.py → common.py

# common.py
def get_user_data(user_id):
    return database.query(user_id)

# user_service.py
from common import get_user_data

def get_user_profile(user_id):
    return get_user_data(user_id)

# order_service.py
from common import get_user_data

def get_user_orders(user_id):
    user = get_user_data(user_id)
    return user.orders
```

**Solution 2: Dependency Injection**

```java
// Before (cycle)
class UserService {
    private OrderService orderService = new OrderService();
}
class OrderService {
    private UserService userService = new UserService(); // ⚠️ CYCLE
}

// After (no cycle - DI breaks it)
class UserService {
    @Autowired private OrderService orderService;
}
class OrderService {
    // Remove direct dependency on UserService
}
```

---

### Symbol Table Service

**What is a Symbol Table?**

A fast lookup database for finding code symbols (functions, classes, methods) by name.

**Think of it as:** A phone book for your code - you know the name, you need the details.

**Key Features:**

1. **O(log n) Lookups**: Very fast even with millions of symbols
2. **Prefix Search**: Type "valid" and find "validateEmail", "validateUser", etc.
3. **Filtering**: Find all classes in a file, all methods of a class, etc.
4. **Parent-Child Relationships**: Track inheritance and implementation

**Example Usage:**

```python
from repomind.services.symbol_table import SymbolTableService

symbol_table = SymbolTableService()

# Find exact symbol
symbol = symbol_table.lookup_exact("UserService.save", repo_name="my-app")
# Returns: Symbol(name="save", qualified_name="UserService.save", 
#                 file_path="src/user_service.py", start_line=42, ...)

# Prefix search (autocomplete)
matches = symbol_table.search_prefix("valid", repo_name="my-app")
# Returns: [
#   Symbol(name="validateEmail", ...),
#   Symbol(name="validateUser", ...),
#   Symbol(name="validation_utils", ...)
# ]

# Find all methods in a class
methods = symbol_table.find_children("UserService", repo_name="my-app")
# Returns: All methods defined in UserService class

# Find parent classes/interfaces
parents = symbol_table.find_parents("UserServiceImpl", repo_name="my-app")
# Returns: [
#   {"parent_name": "UserService", "relation_type": "implements"},
#   {"parent_name": "BaseService", "relation_type": "extends"}
# ]
```

---

### Practical Examples: Using Call Graph

**Scenario**: You're about to change `EmailService.send`, and you want to know what breaks.

**Step 1: Find All Callers**

```python
from repomind.services.call_graph import CallGraphService

call_graph = CallGraphService()

# Find all callers (direct and indirect)
callers = call_graph.find_callers(
    symbol="EmailService.send",
    repo_name="my-app",
    max_depth=3  # 3 levels up
)

print(f"Changing EmailService.send affects {len(callers)} functions:")
for caller in callers:
    print(f"  - {caller.caller} ({caller.caller_file}:{caller.caller_line})")
```

**Output:**
```
Changing EmailService.send affects 12 functions:
  - UserService.createUser (src/user_service.py:45)
  - OrderService.sendReceipt (src/order_service.py:89)
  - AdminService.notifyUser (src/admin_service.py:123)
  ...
```

**Step 2: Check for Dead Code**

```python
from repomind.services.storage import StorageService

storage = StorageService()
chunks = storage.get_all_chunks(repo_name="my-app")

# Identify entry points
entry_points = call_graph.identify_entry_points(chunks, repo_name="my-app")
print(f"Found {len(entry_points)} entry points")

# Find unreachable code
dead_code = call_graph.find_unreachable_from_entry_points(
    entry_points=entry_points,
    repo_name="my-app"
)

print(f"Found {len(dead_code)} unreachable functions:")
for func in dead_code:
    print(f"  - {func}")
```

**Output:**
```
Found 45 entry points
Found 5 unreachable functions:
  - EmailService.sendFax (0 callers) - SAFE TO DELETE
  - UserService.calculateAge_old (0 callers) - SAFE TO DELETE
  - OrderService.applyDiscount_2020 (0 callers) - SAFE TO DELETE
```

**Step 3: Check for Cycles**

```python
# Detect cycles
cycles = call_graph.detect_cycles(repo_name="my-app", group_by_file=True)

if cycles:
    print(f"Found {len(cycles)} circular dependencies:")
    for i, cycle in enumerate(cycles, 1):
        print(f"\nCycle {i}:")
        for j, file in enumerate(cycle):
            next_file = cycle[(j + 1) % len(cycle)]
            print(f"  {file} → {next_file}")
else:
    print("No circular dependencies found! ✅")
```

**Output:**
```
Found 2 circular dependencies:

Cycle 1:
  src/user_service.py → src/order_service.py
  src/order_service.py → src/user_service.py

Cycle 2:
  src/auth/validator.py → src/auth/token.py
  src/auth/token.py → src/auth/validator.py

Recommendation: Extract shared logic to separate modules
```

---

### Debugging Call Graph Issues

**Problem**: "Call graph shows no callers, but I know the function is used"

**Solutions**:

1. **Check if code was indexed recently**
   ```bash
   # Re-index to ensure latest code is captured
   repomind index ~/my-project --repo my-app
   ```

2. **Verify the function name**
   ```python
   # Use exact qualified name
   callers = call_graph.find_callers("UserService.save")  # ✅
   # Not:
   callers = call_graph.find_callers("save")  # ❌ Too vague
   ```

3. **Check for polymorphic calls**
   ```python
   # Use polymorphic search for interfaces
   from repomind.services.symbol_table import SymbolTableService
   
   symbol_table = SymbolTableService()
   callers = call_graph.find_callers_polymorphic(
       "Repository.save",
       symbol_table=symbol_table
   )
   ```

**Problem**: "Circular dependency detected, but I don't understand why"

**Debug Steps**:

1. **Visualize the cycle**
   ```python
   cycles = call_graph.detect_cycles(group_by_file=True)
   for cycle in cycles:
       print("Cycle found:")
       for i, file in enumerate(cycle):
           next_file = cycle[(i + 1) % len(cycle)]
           print(f"  {file} → {next_file}")
   ```

2. **Find the exact calls causing the cycle**
   ```python
   # For each file in the cycle, find what it calls
   for file in cycle:
       # Get all symbols in this file
       conn = call_graph._get_connection()
       cursor = conn.execute(
           "SELECT DISTINCT caller, callee, caller_line FROM calls WHERE caller_file = ?",
           (file,)
       )
       for row in cursor.fetchall():
           print(f"  {row['caller']} calls {row['callee']} (line {row['caller_line']})")
       conn.close()
   ```

---

### Performance Tips

**Tip 1: Use Bulk Operations**

```python
# ❌ Slow (1000 database calls)
for relation in relations:
    call_graph.add_call(relation)

# ✅ Fast (1 database call)
call_graph.add_calls_bulk(relations)
```

**Tip 2: Limit Depth for Large Codebases**

```python
# ❌ Slow (traverses entire graph)
callers = call_graph.find_callers("func", max_depth=999)

# ✅ Fast (stops at 3 levels)
callers = call_graph.find_callers("func", max_depth=3)
```

**Tip 3: Use Repo Filter**

```python
# ❌ Slow (searches all repos)
callers = call_graph.find_callers("UserService.save")

# ✅ Fast (searches one repo)
callers = call_graph.find_callers("UserService.save", repo_name="my-app")
```

**Tip 4: Cache Symbol Table Lookups**

```python
# ✅ Good - reuse service instance
symbol_table = SymbolTableService()  # Create once

for symbol in many_symbols:
    result = symbol_table.lookup_exact(symbol)  # Uses same connection pool
```

---

### Testing Your Call Graph Changes

```python
# tests/test_call_graph.py
import pytest
from repomind.services.call_graph import CallGraphService, CallRelation

def test_find_callers():
    """Test finding callers of a function."""
    call_graph = CallGraphService()
    
    # Add test data
    relation = CallRelation(
        caller="UserService.save",
        callee="Database.execute",
        caller_file="src/user.py",
        caller_line=42,
        repo_name="test-repo",
        call_type="direct"
    )
    call_graph.add_call(relation)
    
    # Test find_callers
    callers = call_graph.find_callers("Database.execute", repo_name="test-repo")
    
    assert len(callers) == 1
    assert callers[0].caller == "UserService.save"
    assert callers[0].caller_line == 42

def test_detect_cycles():
    """Test circular dependency detection."""
    call_graph = CallGraphService()
    
    # Create a cycle: A → B → A
    call_graph.add_calls_bulk([
        CallRelation("A", "B", "file.py", 1, "test", "direct"),
        CallRelation("B", "A", "file.py", 2, "test", "direct")
    ])
    
    cycles = call_graph.detect_cycles(repo_name="test", group_by_file=False)
    
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B"}

def test_dependency_injection_resolution():
    """Test DI resolution."""
    call_graph = CallGraphService()
    
    # Add class dependency
    call_graph.add_class_dependencies(
        class_name="UserController",
        dependencies=[{"name": "userService", "type": "UserService", "injection_type": "field"}],
        repo_name="test"
    )
    
    # Resolve callee
    resolved = call_graph.resolve_callee("UserController", "userService.save", "test")
    
    assert resolved == "UserService.save"
```

---

## Code Metrics and Quality Analysis

**What are Code Metrics?**

Quantitative measurements of code quality and complexity:
- **Cyclomatic Complexity**: How many decision paths exist
- **Cognitive Complexity**: How hard the code is to understand
- **Lines of Code**: Size metrics (total, source, comments)
- **Maintainability Index**: Overall quality score
- **Halstead Metrics**: Volume, difficulty, effort

**Why Are Metrics Important?**

1. **Identify Hotspots**: Find complex code that needs refactoring
2. **Code Review Focus**: Spend more time on complex functions
3. **Technical Debt**: Quantify and track code quality over time
4. **Refactoring Targets**: Prioritize which code to improve first

---

### Understanding Complexity Metrics

**Cyclomatic Complexity**

Counts the number of independent paths through code:

```python
# Cyclomatic Complexity = 1 (no decisions)
def simple_function(x):
    return x + 1

# Cyclomatic Complexity = 2 (1 if statement)
def moderate_function(x):
    if x > 0:
        return x
    return 0

# Cyclomatic Complexity = 5 (4 conditions + 1)
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            return x + y
        return x
    return 0
```

**Complexity Ratings:**
- **1-5**: Low (Simple, easy to test)
- **6-10**: Moderate (Acceptable complexity)
- **11-20**: High (Consider refactoring)
- **21+**: Very High (Difficult to maintain, hard to test)

**Cognitive Complexity**

Measures how hard code is to understand (includes nesting penalties):

```python
# Cyclomatic: 3, Cognitive: 3
def example1(items):
    for item in items:         # +1
        if item.active:        # +1
            if item.valid:     # +2 (nested)
                process(item)

# Cyclomatic: 3, Cognitive: 1
def example2(items):
    for item in items:         # +1
        if not item.active:    # +1 (but early return)
            continue
        if not item.valid:     # +1 (but early return)
            continue
        process(item)
```

Cognitive complexity penalizes:
- Deep nesting
- Complex boolean expressions
- Breaks in linear flow

---

### Using the Metrics Service

**Example 1: Analyze a Function**

```python
from repomind.services.metrics import MetricsService

metrics = MetricsService()

# Analyze a function
code = """
def calculate_discount(price, customer_type, quantity):
    if customer_type == "vip":
        if quantity > 100:
            return price * 0.7
        elif quantity > 50:
            return price * 0.8
        else:
            return price * 0.9
    elif customer_type == "regular":
        if quantity > 100:
            return price * 0.9
        else:
            return price * 0.95
    return price
"""

result = metrics.analyze_function(code, language="python")

print(f"Function: {result.name}")
print(f"Cyclomatic Complexity: {result.cyclomatic.complexity} ({result.cyclomatic.rating})")
print(f"Cognitive Complexity: {result.cognitive.complexity}")
print(f"Lines of Code: {result.loc.source_lines}")
print(f"Comment Ratio: {result.loc.comment_ratio:.1%}")

if result.is_complex:
    print("⚠️ This function is complex and should be refactored!")
```

**Output:**
```
Function: calculate_discount
Cyclomatic Complexity: 7 (moderate)
Cognitive Complexity: 9
Lines of Code: 13
Comment Ratio: 0.0%
⚠️ This function is complex and should be refactored!
```

**Example 2: Analyze Entire Repository**

```python
# Analyze all code in repository
report = metrics.analyze_repository("my-repo")

print(f"Repository Metrics Summary:")
print(f"Total Functions: {report.total_functions}")
print(f"Average Complexity: {report.avg_cyclomatic:.1f}")
print(f"Complex Functions: {len(report.complex_functions)}")

# Show top 10 most complex functions
print("\nTop 10 Complexity Hotspots:")
for i, func in enumerate(report.complex_functions[:10], 1):
    print(f"{i}. {func.qualified_name}")
    print(f"   Cyclomatic: {func.cyclomatic.complexity}")
    print(f"   Location: {func.file_path}:{func.start_line}")
    print()
```

**Example 3: Track Metrics Over Time**

```python
# Compare metrics between versions
old_report = metrics.analyze_repository("my-repo", commit="v1.0.0")
new_report = metrics.analyze_repository("my-repo", commit="HEAD")

# Calculate delta
complexity_change = new_report.avg_cyclomatic - old_report.avg_cyclomatic

if complexity_change > 0:
    print(f"⚠️ Complexity increased by {complexity_change:.1f}")
    print("Consider refactoring before merging")
else:
    print(f"✅ Complexity decreased by {abs(complexity_change):.1f}")
    print("Good job simplifying the code!")
```

**Example 4: Set Quality Gates**

```python
# Use in CI/CD to enforce quality standards
def check_quality_gate(file_path: str) -> bool:
    """Return True if code meets quality standards."""
    result = metrics.analyze_file(file_path)
    
    # Define thresholds
    MAX_CYCLOMATIC = 10
    MAX_COGNITIVE = 15
    MIN_COMMENT_RATIO = 0.1  # 10% comments
    
    issues = []
    
    for func in result.functions:
        if func.cyclomatic.complexity > MAX_CYCLOMATIC:
            issues.append(f"{func.name}: Cyclomatic complexity too high")
        
        if func.cognitive.complexity > MAX_COGNITIVE:
            issues.append(f"{func.name}: Cognitive complexity too high")
        
        if func.loc.comment_ratio < MIN_COMMENT_RATIO:
            issues.append(f"{func.name}: Insufficient comments")
    
    if issues:
        print("❌ Quality gate failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Quality gate passed")
    return True
```

---

### Halstead Metrics

**What Are Halstead Metrics?**

Scientific measurements of code complexity based on operators and operands:

- **Vocabulary**: Number of unique operators + operands
- **Length**: Total operators + operands
- **Volume**: How much information a programmer must understand
- **Difficulty**: How hard the code is to write/maintain
- **Effort**: Time needed to write/understand the code

**Example:**

```python
# Code: result = a + b * c
# Operators: =, +, * (3 total, 3 unique)
# Operands: result, a, b, c (4 total, 4 unique)
# Vocabulary: 7
# Volume: 4 * log2(7) = 11.2
```

**Using Halstead Metrics:**

```python
result = metrics.analyze_function(code, language="python")

if result.halstead:
    print(f"Vocabulary: {result.halstead.vocabulary}")
    print(f"Volume: {result.halstead.volume:.1f}")
    print(f"Difficulty: {result.halstead.difficulty:.1f}")
    print(f"Effort: {result.halstead.effort:.1f}")
    
    # High difficulty means code is hard to maintain
    if result.halstead.difficulty > 30:
        print("⚠️ This code is difficult to maintain")
```

---

## Code Ownership and Contributor Tracking

**What is Code Ownership?**

Identifying who is responsible for maintaining different parts of the codebase:
- Parse `CODEOWNERS` files
- Analyze git history
- Track contributor expertise
- Suggest reviewers for changes

**Why Track Ownership?**

1. **Better Code Reviews**: Route changes to the right reviewers
2. **Knowledge Transfer**: Identify experts for questions
3. **Bus Factor**: Prevent single points of failure
4. **Onboarding**: Know who to ask about specific code

---

### CODEOWNERS File Format

**Basic Syntax:**

```
# CODEOWNERS file

# Default owners for everything
* @backend-team

# Frontend code
/frontend/ @frontend-team @ui-team

# API routes
/src/api/ @backend-team @api-team

# Security-critical code
/src/auth/ @security-team
/src/crypto/ @security-team

# Specific files
package.json @tech-lead
README.md @docs-team

# Pattern matching
*.py @python-team
*.java @java-team
```

---

### Using the Ownership Service

**Example 1: Find Owners of a File**

```python
from repomind.services.ownership import OwnershipService

ownership = OwnershipService("/path/to/repo")

# Get owners for a file
owners = ownership.get_owners("src/services/auth.py")

print(f"Owners of auth.py:")
for owner in owners.codeowners:
    print(f"  - {owner}")

print(f"\nTop Contributors:")
for author, commits in owners.top_contributors[:5]:
    print(f"  - {author}: {commits} commits")

print(f"\nLast modified by: {owners.last_modified_by}")
```

**Output:**
```
Owners of auth.py:
  - @backend-team
  - @security-team

Top Contributors:
  - alice@company.com: 47 commits
  - bob@company.com: 23 commits
  - charlie@company.com: 15 commits

Last modified by: alice@company.com
```

**Example 2: Suggest Reviewers for PR**

```python
# Suggest reviewers for changed files
changed_files = [
    "src/services/auth.py",
    "src/services/user.py",
    "src/api/auth_routes.py"
]

suggestions = ownership.suggest_reviewers(changed_files)

print("Suggested Reviewers:")
for suggestion in suggestions:
    print(f"\n{suggestion.name}")
    print(f"  Reason: {suggestion.reason}")
    print(f"  Confidence: {suggestion.confidence:.0%}")
    print(f"  Expertise Match: {suggestion.expertise_match}")
    print(f"  Recently Active: {suggestion.recent_activity}")
```

**Output:**
```
Suggested Reviewers:

alice@company.com
  Reason: Top contributor to auth.py (47 commits)
  Confidence: 95%
  Expertise Match: True
  Recently Active: True

@security-team
  Reason: CODEOWNERS for security-critical files
  Confidence: 100%
  Expertise Match: True
  Recently Active: N/A

bob@company.com
  Reason: Recent contributor to user.py
  Confidence: 70%
  Expertise Match: True
  Recently Active: True
```

**Example 3: Analyze Contributor Expertise**

```python
# Get detailed contributor stats
contributor = ownership.get_contributor_stats("alice@company.com")

print(f"Contributor: {contributor.name}")
print(f"Total Commits: {contributor.commit_count}")
print(f"Files Touched: {contributor.files_touched}")
print(f"Lines Added: {contributor.lines_added}")
print(f"Lines Removed: {contributor.lines_removed}")
print(f"Last Commit: {contributor.last_commit_date}")

print(f"\nExpertise Areas:")
for area in contributor.expertise_areas:
    print(f"  - {area}")
```

**Output:**
```
Contributor: Alice Smith
Total Commits: 347
Files Touched: 156
Lines Added: 12,450
Lines Removed: 8,230
Last Commit: 2026-01-25

Expertise Areas:
  - Authentication (47 commits)
  - User Management (34 commits)
  - API Routes (28 commits)
  - Database (21 commits)
```

**Example 4: Find Experts for a Topic**

```python
# Find who knows most about authentication
experts = ownership.find_experts("auth", limit=5)

print("Authentication Experts:")
for i, expert in enumerate(experts, 1):
    print(f"{i}. {expert.name}")
    print(f"   Commits in auth: {expert.auth_commits}")
    print(f"   Last contribution: {expert.last_commit_date}")
```

---

### Integration with Code Review

**Automatic Reviewer Assignment:**

```python
def auto_assign_reviewers(pr_files: list[str]) -> list[str]:
    """Automatically assign reviewers based on ownership and expertise."""
    ownership = OwnershipService()
    
    suggestions = ownership.suggest_reviewers(pr_files)
    
    # Filter to high-confidence suggestions
    reviewers = [
        s.name for s in suggestions 
        if s.confidence > 0.7 and s.recent_activity
    ]
    
    # Ensure at least 2 reviewers
    while len(reviewers) < 2 and suggestions:
        reviewers.append(suggestions.pop(0).name)
    
    return reviewers[:3]  # Max 3 reviewers
```

---

### Testing Metrics and Ownership

```python
# tests/test_metrics.py
import pytest
from repomind.services.metrics import MetricsService, ComplexityRating

def test_cyclomatic_complexity():
    """Test cyclomatic complexity calculation."""
    metrics = MetricsService()
    
    # Simple function (complexity = 1)
    code = "def simple(x): return x + 1"
    result = metrics.analyze_function(code, "python")
    assert result.cyclomatic.complexity == 1
    assert result.cyclomatic.rating == ComplexityRating.LOW
    
    # Complex function (complexity = 5)
    code = """
def complex(x, y):
    if x > 0:
        if y > 0:
            return x + y
        return x
    return 0
    """
    result = metrics.analyze_function(code, "python")
    assert result.cyclomatic.complexity >= 4

def test_cognitive_complexity():
    """Test cognitive complexity calculation."""
    metrics = MetricsService()
    
    # Nested conditions have higher cognitive complexity
    code = """
def nested(items):
    for item in items:
        if item.active:
            if item.valid:
                process(item)
    """
    result = metrics.analyze_function(code, "python")
    assert result.cognitive.complexity > result.cyclomatic.complexity

def test_ownership_detection():
    """Test CODEOWNERS parsing."""
    from repomind.services.ownership import OwnershipService
    
    ownership = OwnershipService(".")
    
    # Get owners for a Python file
    owners = ownership.get_owners("src/repomind/server.py")
    
    assert owners is not None
    assert len(owners.codeowners) > 0 or len(owners.top_contributors) > 0
```

---

## Security Scanning

**What is Security Scanning?**

RepoMind can detect hardcoded secrets and security vulnerabilities in your code:
- Find API keys, tokens, passwords in source code
- Detect common security anti-patterns
- Identify potential data leaks

**Why Is This Important?**

1. **Prevent Secret Leaks**: Catch hardcoded credentials before they reach production
2. **Security Best Practices**: Identify insecure patterns early
3. **Compliance**: Meet security audit requirements
4. **Peace of Mind**: Automated checking vs manual review

---

### Types of Secrets Detected

**26 Secret Types Detected:**

1. **Cloud Provider Keys**
   - AWS Access Keys (AKIA...)
   - AWS Secret Keys
   - Azure Keys
   - GCP Service Account Keys

2. **Version Control Tokens**
   - GitHub Personal Access Tokens
   - GitLab Tokens
   - Bitbucket Tokens

3. **Communication Services**
   - Slack Tokens
   - Slack Webhooks
   - Twilio API Keys

4. **Payment Processing**
   - Stripe API Keys
   - PayPal Tokens

5. **Generic Patterns**
   - JWT Secrets
   - Database Connection Strings with Passwords
   - Private Keys (RSA, SSH, PGP)
   - API Keys and Tokens
   - Bearer Tokens
   - Base64-encoded Secrets

---

### Using the Security Scanner

**Example 1: Scan a Single File**

```python
from repomind.services.security_scanner import SecurityScanner

scanner = SecurityScanner()

# Scan a file
findings = scanner.scan_file("config.py")

for finding in findings:
    print(f"⚠️ {finding.severity.upper()}: {finding.secret_type}")
    print(f"   File: {finding.file_path}:{finding.line_number}")
    print(f"   Description: {finding.description}")
    print(f"   Recommendation: {finding.recommendation}")
    print()
```

**Output:**
```
⚠️ CRITICAL: aws_access_key
   File: config.py:15
   Description: AWS Access Key found in source code
   Recommendation: Move to environment variables or AWS Secrets Manager

⚠️ HIGH: database_url
   File: config.py:23
   Description: Database URL with embedded password
   Recommendation: Use environment variables for database credentials
```

**Example 2: Scan Entire Repository**

```python
# Scan all files in a repository
report = scanner.scan_repository("/path/to/repo")

print(f"Scanned {report.total_files_scanned} files")
print(f"Found {report.total_findings} security issues")
print(f"\nBreakdown by severity:")
for severity, count in report.findings_by_severity.items():
    print(f"  {severity}: {count}")

# Show critical findings only
critical = [f for f in report.findings if f.severity == "critical"]
print(f"\n{len(critical)} critical issues that need immediate attention:")
for finding in critical:
    print(f"  - {finding.file_path}:{finding.line_number} - {finding.description}")
```

**Example 3: Filter by Secret Type**

```python
from repomind.services.security_scanner import SecretType

# Find all AWS credentials
aws_findings = [
    f for f in findings 
    if f.secret_type in [SecretType.AWS_ACCESS_KEY, SecretType.AWS_SECRET_KEY]
]

if aws_findings:
    print("⚠️ Found AWS credentials in code!")
    for finding in aws_findings:
        print(f"  {finding.file_path}:{finding.line_number}")
```

---

### Security Best Practices

**DO:**
- ✅ Use environment variables for secrets
- ✅ Use secret management services (AWS Secrets Manager, Azure Key Vault)
- ✅ Add secrets to `.gitignore` before committing
- ✅ Run security scans in CI/CD pipelines
- ✅ Rotate secrets immediately if leaked

**DON'T:**
- ❌ Hardcode API keys in source code
- ❌ Commit `.env` files with real secrets
- ❌ Share secrets in comments or documentation
- ❌ Store secrets in version control
- ❌ Use the same secret across environments

**Example: Proper Secret Management**

```python
# ❌ BAD - Hardcoded secret
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"

# ✅ GOOD - Environment variable
import os
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")

# ✅ BETTER - Secret manager
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='prod/api/aws_key')
AWS_ACCESS_KEY = response['SecretString']
```

---

## Pattern Analysis

**What is Pattern Analysis?**

RepoMind analyzes your codebase to understand:
- Which libraries/frameworks are used most
- What coding patterns are prevalent
- Which conventions the team follows
- Which files serve as good examples

**Why Is This Useful?**

1. **AI Code Generation**: Help Copilot generate code matching your style
2. **Onboarding**: Show new developers "the way we do things"
3. **Consistency**: Identify and enforce team conventions
4. **Library Selection**: Know which libraries are already in use

---

### Pattern Categories

RepoMind tracks patterns in 10 categories:

1. **Dependency Injection**: Constructor vs field vs setter injection
2. **Error Handling**: Try-catch vs error boundaries vs result types
3. **Async Patterns**: Promises vs async/await vs callbacks
4. **HTTP Clients**: axios vs fetch vs got
5. **Testing**: Jest vs Mocha, pytest vs unittest
6. **Logging**: console.log vs winston vs pino
7. **Configuration**: .env vs config files vs feature flags
8. **Database**: ORM vs query builders vs raw SQL
9. **Authentication**: JWT vs sessions vs OAuth
10. **State Management**: Redux vs Context vs Zustand

---

### Using the Pattern Analyzer

**Example 1: Analyze Library Usage**

```python
from repomind.services.pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()

# Get library usage for HTTP clients
libs = analyzer.analyze_library_usage("my-repo", category="http_client")

print("HTTP Client Usage:")
for lib_name, usage in sorted(libs.items(), key=lambda x: x[1].import_count, reverse=True):
    print(f"  {lib_name}:")
    print(f"    Imported in {usage.file_count} files")
    print(f"    Used {usage.import_count} times")
    print(f"    Trend: {usage.momentum}")
```

**Output:**
```
HTTP Client Usage:
  axios:
    Imported in 45 files
    Used 127 times
    Trend: stable
  fetch:
    Imported in 12 files
    Used 23 times
    Trend: declining
  got:
    Imported in 3 files
    Used 5 times
    Trend: rising
```

**Example 2: Get Pattern Recommendations**

```python
# Get recommended pattern for dependency injection
report = analyzer.get_pattern_recommendations(
    category="dependency_injection",
    repo_name="my-repo"
)

print(f"Recommended Pattern: {report.pattern_name}")
print(f"Usage: {report.percentage:.1f}%")
print(f"Momentum: {report.momentum}")
print(f"Occurrence: {report.occurrence_count} times")

if report.alternatives:
    print(f"\nAlternatives:")
    for alt in report.alternatives:
        print(f"  - {alt}")

print(f"\nExample from your codebase:")
for example in report.examples[:1]:  # Show first example
    print(f"  File: {example.file_path}:{example.line_number}")
    print(f"  Code: {example.code_snippet}")
```

**Output:**
```
Recommended Pattern: constructor_injection
Usage: 78.3%
Momentum: stable
Occurrence: 156 times

Alternatives:
  - field_injection (18.7%)
  - setter_injection (3.0%)

Example from your codebase:
  File: src/services/UserService.java:15
  Code: public UserService(UserRepository repo, EmailService email) {
    this.repo = repo;
    this.email = email;
  }
```

**Example 3: Find Golden Files**

```python
# Find exemplary implementations that follow best practices
golden_files = analyzer.find_golden_files("my-repo")

print("Golden Files (Good Examples):")
for gf in golden_files:
    print(f"\n{gf.file_path}:")
    print(f"  Score: {gf.quality_score:.2f}/10")
    print(f"  Why: {', '.join(gf.reasons)}")
    print(f"  Good for learning: {', '.join(gf.exemplifies)}")
```

**Output:**
```
Golden Files (Good Examples):

src/services/UserService.ts:
  Score: 9.2/10
  Why: consistent_patterns, well_tested, good_documentation
  Good for learning: dependency_injection, error_handling, async_patterns

src/controllers/AuthController.java:
  Score: 8.7/10
  Why: follows_conventions, complete_error_handling
  Good for learning: authentication, error_handling, logging
```

**Example 4: Testing Convention Detection**

```python
# Detect which testing framework is most used
testing_report = analyzer.get_pattern_recommendations(
    category="testing",
    repo_name="my-repo"
)

print(f"Recommended Testing Framework: {testing_report.pattern_name}")
print(f"Usage: {testing_report.percentage:.0f}%")

# Use this info to generate test files with the right framework
if testing_report.pattern_name == "jest":
    print("\nGenerate tests using Jest:")
    print("  describe('UserService', () => {")
    print("    it('should create user', async () => {")
    print("      // test code")
    print("    });")
    print("  });")
```

---

### Pattern Momentum

**What is Momentum?**

Tracks whether a pattern is becoming more or less popular over time.

**How It Works:**

```python
# RepoMind looks at file modification dates
recent_usage = count_pattern_in_files_modified_last_30_days()
older_usage = count_pattern_in_files_modified_30_60_days_ago()

if recent_usage > older_usage * 1.2:  # 20% increase
    momentum = Momentum.RISING
elif recent_usage < older_usage * 0.8:  # 20% decrease
    momentum = Momentum.DECLINING
else:
    momentum = Momentum.STABLE
```

**Why This Matters:**

- **Rising**: Team is moving toward this pattern → safe to use in new code
- **Stable**: Established pattern → good choice
- **Declining**: Team is moving away → consider alternatives

**Example:**

```python
# Check if async/await is gaining traction
async_report = analyzer.get_pattern_recommendations(
    category="async_patterns",
    repo_name="my-repo"
)

if async_report.momentum == "rising":
    print("✅ Team is adopting async/await - use it in new code")
elif async_report.momentum == "declining":
    print("⚠️ Team is moving away from async/await - check alternatives")
```

---

### Integration with AI Code Generation

**How Copilot Uses This:**

```
You: @repomind generate a service for handling payments

Copilot (checks patterns first):
1. Library: Using Stripe (92% of payment code)
2. DI Pattern: Constructor injection (78% usage)
3. Error Handling: Try-catch with custom errors (85% usage)
4. Testing: Jest (100% of test files)
5. Logging: Winston logger (67% usage)

Generated code:
import Stripe from 'stripe';
import { Logger } from './Logger';

export class PaymentService {
  constructor(
    private stripe: Stripe,
    private logger: Logger
  ) {}

  async processPayment(amount: number): Promise<PaymentResult> {
    try {
      const charge = await this.stripe.charges.create({ amount });
      this.logger.info('Payment processed', { chargeId: charge.id });
      return { success: true, chargeId: charge.id };
    } catch (error) {
      this.logger.error('Payment failed', { error });
      throw new PaymentError('Failed to process payment', error);
    }
  }
}
```

---

### Testing Pattern Analysis

```python
# tests/test_pattern_analyzer.py
import pytest
from repomind.services.pattern_analyzer import PatternAnalyzer, PatternCategory

def test_library_usage_detection():
    """Test detecting library usage."""
    analyzer = PatternAnalyzer()
    
    # Analyze library usage
    libs = analyzer.analyze_library_usage("test-repo", category="http_client")
    
    # Should find axios as most used
    assert "axios" in libs
    assert libs["axios"].import_count > 0

def test_pattern_recommendations():
    """Test getting pattern recommendations."""
    analyzer = PatternAnalyzer()
    
    report = analyzer.get_pattern_recommendations(
        category=PatternCategory.DEPENDENCY_INJECTION,
        repo_name="test-repo"
    )
    
    assert report.pattern_name is not None
    assert 0 <= report.percentage <= 100
    assert report.occurrence_count >= 0

def test_golden_file_detection():
    """Test finding golden files."""
    analyzer = PatternAnalyzer()
    
    golden_files = analyzer.find_golden_files("test-repo")
    
    # Should have some golden files
    assert len(golden_files) > 0
    
    # Each should have a quality score
    for gf in golden_files:
        assert 0 <= gf.quality_score <= 10
        assert len(gf.reasons) > 0

def test_momentum_calculation():
    """Test pattern momentum detection."""
    from repomind.services.pattern_analyzer import Momentum
    
    analyzer = PatternAnalyzer()
    
    # Get report with momentum
    report = analyzer.get_pattern_recommendations(
        category=PatternCategory.ASYNC_PATTERNS,
        repo_name="test-repo"
    )
    
    assert report.momentum in [Momentum.RISING, Momentum.STABLE, Momentum.DECLINING]
```

---

## Diagram Generation

**What is Diagram Generation?**

RepoMind can automatically generate visual diagrams based on semantic search:
- Architecture diagrams (components and layers)
- Data flow diagrams (how data moves)
- Call flow diagrams (execution paths)
- Class diagrams (inheritance and composition)
- Sequence diagrams (call order)
- Dependency diagrams (module relationships)

**Why Generate Diagrams?**

1. **Understanding Code**: Visualize complex relationships
2. **Documentation**: Generate up-to-date diagrams automatically
3. **Onboarding**: Help new developers understand the system
4. **Communication**: Share architecture with non-technical stakeholders

---

### Diagram Types

**1. Architecture Diagram**

Shows components grouped by architectural layer:

```python
from repomind.tools.generate_diagrams import generate_architecture_diagram

result = generate_architecture_diagram(
    query="user authentication",
    repo_name="my-app",
    output_format="mermaid"
)

print(result["diagram"])
```

**Output (Mermaid):**
```mermaid
flowchart TD
    %% Architecture Diagram: user authentication
    subgraph API Layer
        auth_controller[AuthController]
    end
    subgraph Service Layer
        auth_service[AuthService]
        token_service[TokenService]
    end
    subgraph Repository Layer
        user_repository[UserRepository]
    end
    auth_controller --> auth_service
    auth_service --> token_service
    auth_service --> user_repository
```

**2. Data Flow Diagram**

Shows how data moves through the system:

```python
from repomind.tools.generate_diagrams import generate_dataflow_diagram

result = generate_dataflow_diagram(
    query="order processing",
    repo_name="my-app",
    output_format="mermaid"
)
```

**Output (Mermaid):**
```mermaid
flowchart LR
    %% Data Flow Diagram: order processing
    create_order[(Order Input)]
    validate_order[Validate Order]
    process_payment[Process Payment]
    send_confirmation[/Send Email/]
    
    create_order -->|validate| validate_order
    validate_order -->|process| process_payment
    process_payment -->|notify| send_confirmation
```

**3. Call Flow Diagram**

Shows function call relationships:

```python
from repomind.tools.generate_diagrams import generate_callflow_diagram

result = generate_callflow_diagram(
    query="login endpoint",
    repo_name="my-app",
    depth=3
)
```

**4. Class Diagram**

Shows class inheritance and interfaces:

```python
from repomind.tools.generate_diagrams import generate_class_diagram

result = generate_class_diagram(
    query="user models",
    repo_name="my-app"
)
```

**Output (Mermaid):**
```mermaid
classDiagram
    %% Class Diagram: user models
    class User {
        +getId()
        +getEmail()
        +setPassword()
    }
    class Admin {
        +getPermissions()
    }
    class UserRepository {
        <<interface>>
        +findById()
        +save()
    }
    User <|-- Admin
    UserRepository <|.. UserRepositoryImpl
```

**5. Sequence Diagram**

Shows method call order:

```python
from repomind.tools.generate_diagrams import generate_sequence_diagram

result = generate_sequence_diagram(
    query="user login flow",
    repo_name="my-app",
    depth=5
)
```

**Output (Mermaid):**
```mermaid
sequenceDiagram
    %% Sequence Diagram: user login flow
    participant AuthController
    participant AuthService
    participant UserRepository
    participant TokenService
    
    AuthController->>+AuthService: authenticate
    AuthService->>+UserRepository: findByEmail
    UserRepository-->>-AuthService: user
    AuthService->>+TokenService: generateToken
    TokenService-->>-AuthService: token
    AuthService-->>-AuthController: authResult
```

**6. Dependency Diagram**

Shows module import relationships:

```python
from repomind.tools.generate_diagrams import generate_dependency_diagram

result = generate_dependency_diagram(
    query="authentication modules",
    repo_name="my-app"
)
```

---

### Output Formats

**Mermaid (Default)**
- Best for: GitHub, GitLab, Notion, Obsidian
- Renders directly in markdown
- Most common choice

**PlantUML**
- Best for: IDE plugins, documentation generators
- Widely supported in enterprise tools
- More customization options

**DOT (Graphviz)**
- Best for: Custom rendering, large graphs
- Can be converted to many image formats
- Good for complex diagrams

**ASCII**
- Best for: Terminal output, plain text docs
- Works everywhere
- Simple representation

---

### Using the Unified Function

For convenience, use `generate_diagram` for any diagram type:

```python
from repomind.tools.generate_diagrams import generate_diagram

# Architecture diagram
result = generate_diagram(
    query="user authentication",
    diagram_type="architecture",
    output_format="mermaid"
)

# Data flow diagram
result = generate_diagram(
    query="payment processing",
    diagram_type="dataflow",
    output_format="plantuml"
)

# Sequence diagram
result = generate_diagram(
    query="checkout flow",
    diagram_type="sequence",
    output_format="mermaid",
    depth=5
)
```

---

### MCP Integration

Use diagram generation in VS Code with Copilot:

```
You: @repomind generate an architecture diagram for user authentication

Copilot: Here's an architecture diagram for user authentication:

```mermaid
flowchart TD
    subgraph API Layer
        AuthController[AuthController]
    end
    subgraph Service Layer
        AuthService[AuthService]
        TokenService[TokenService]
    end
    subgraph Repository Layer
        UserRepository[UserRepository]
    end
    AuthController --> AuthService
    AuthService --> TokenService
    AuthService --> UserRepository
```

The diagram shows:
- **API Layer**: AuthController handles HTTP requests
- **Service Layer**: AuthService orchestrates authentication, TokenService manages JWT
- **Repository Layer**: UserRepository accesses user data

Would you like me to generate a sequence diagram showing the login flow?
```

---

### Common Patterns

**Pattern 1: Understand Before Modifying**

```python
# Before changing auth code, understand the structure
arch = generate_architecture_diagram("authentication", "my-app")
flow = generate_sequence_diagram("login flow", "my-app")

# Now you know:
# - Which components are involved
# - What order things happen
# - What might break
```

**Pattern 2: Document API Endpoints**

```python
# Generate diagrams for each major feature
features = ["user registration", "checkout", "notifications"]

for feature in features:
    result = generate_dataflow_diagram(feature, "my-app")
    with open(f"docs/diagrams/{feature.replace(' ', '_')}.md", "w") as f:
        f.write(f"# {feature.title()} Data Flow\n\n")
        f.write("```mermaid\n")
        f.write(result["diagram"])
        f.write("\n```\n")
```

**Pattern 3: Onboarding Documentation**

```python
# Generate overview diagrams for new developers
generate_architecture_diagram("main application", "my-app", max_nodes=50)
generate_class_diagram("domain models", "my-app", max_nodes=30)
generate_dependency_diagram("all modules", "my-app", max_modules=25)
```

---

### Testing Diagram Generation

```python
# tests/test_diagrams.py
import pytest
from repomind.tools.generate_diagrams import (
    generate_diagram,
    generate_architecture_diagram,
    generate_dataflow_diagram,
    DiagramFormat,
    DiagramType,
)

def test_architecture_diagram():
    """Test generating architecture diagram."""
    result = generate_architecture_diagram(
        query="service layer",
        repo_name="test-repo",
        output_format="mermaid"
    )
    
    assert "diagram" in result
    assert "flowchart" in result["diagram"]
    assert result["format"] == "mermaid"
    assert result["type"] == "architecture"

def test_diagram_formats():
    """Test different output formats."""
    for fmt in ["mermaid", "plantuml", "dot", "ascii"]:
        result = generate_diagram(
            query="test",
            diagram_type="architecture",
            output_format=fmt
        )
        assert result["format"] == fmt

def test_diagram_types():
    """Test different diagram types."""
    types = ["architecture", "dataflow", "callflow", "class", "sequence", "dependency"]
    
    for diagram_type in types:
        result = generate_diagram(
            query="test",
            diagram_type=diagram_type
        )
        assert result["type"] == diagram_type

def test_empty_results():
    """Test handling of queries with no results."""
    result = generate_architecture_diagram(
        query="nonexistent feature xyz123",
        repo_name="my-app"
    )
    
    # Should return empty diagram, not error
    assert "diagram" in result
    assert result["statistics"]["nodes"] == 0
```

---

## Glossary

### Core Concepts

**AST (Abstract Syntax Tree):** A tree representation of code structure

**Call Graph:** Database of "who calls who" relationships between functions

**Caller:** A function that calls another function

**Callee:** A function that is being called

**Chunk:** A meaningful piece of code (function, class, method)

**ChromaDB:** The vector database we use for similarity search

**Embedding:** A list of numbers representing text meaning

**Entry Point:** A function called by external code (main, API routes, tests)

**MCP (Model Context Protocol):** Standard for connecting AI tools

**Parser:** Code that reads source code and extracts structure

**Semantic Search:** Finding by meaning, not exact text

**Symbol Table:** Fast lookup database for code symbols

**Tree-sitter:** Library for parsing multiple programming languages

**Vector Database:** Database optimized for similarity search

**Virtual Environment:** Isolated Python environment for a project

### Advanced Features

**API Call:** An outgoing HTTP request from frontend code to an API

**ApiCall Object:** Data model representing a frontend API call with normalized path

**Base64 Secret:** Encoded credentials that may appear in code

**Call Type:** Category of function call (direct, virtual, callback, polymorphic)

**Circular Dependency:** When code depends on itself (A → B → A)

**CODEOWNERS:** File defining code ownership and review responsibilities

**Cognitive Complexity:** Measure of how hard code is to understand (includes nesting penalties)

**Confidence Score:** 0-1 value indicating match quality or detection certainty

**Contributor Stats:** Metrics about a developer's contributions (commits, lines, expertise)

**Cyclomatic Complexity:** Number of independent paths through code

**Dead Code:** Code that exists but is never executed

**Dependency Injection:** Framework automatically provides object instances

**Entry Point Analysis:** Finding functions that can't be reached from main/API routes

**Golden File:** Exemplary code file that follows best practices

**Halstead Metrics:** Scientific measurements (volume, difficulty, effort) of code complexity

**Hotspot:** Code that is called very frequently or has high complexity

**Hybrid Search:** Combining semantic search with keyword matching

**Interface:** Contract that multiple classes can implement

**LOC (Lines of Code):** Size metrics including total, source, comments, blanks

**Maintainability Index:** Overall quality score combining multiple metrics

**Momentum:** Trend direction for patterns (rising, stable, declining)

**Normalized Route:** Unified representation of API paths across frameworks

**Ownership Rule:** Pattern matching rule from CODEOWNERS file

**Parameter Source:** Where a route parameter comes from (path, query, body, header)

**Pattern Analysis:** Detecting coding conventions and library usage

**Pattern Category:** Type of pattern (DI, error handling, async, etc.)

**Polymorphic Call:** Call to an interface that could resolve to multiple implementations

**Query Expansion:** Adding related terms to search queries

**Reachability Analysis:** Finding code that can be reached from entry points

**Route Conflict:** Multiple endpoints with the same path

**Route Endpoint:** Backend API route handler

**RouteMatch:** Link between frontend API call and backend endpoint

**SCC (Strongly Connected Component):** A cycle in a graph

**Secret Type:** Category of sensitive data (API key, token, password, etc.)

**Security Finding:** Detected security issue with severity and recommendation

**Severity:** Importance level (critical, high, medium, low, info)

**Symbol:** A code entity (function, class, method) in the symbol table

**Tarjan's Algorithm:** Algorithm for finding cycles in a graph

**Transitive Caller:** Indirect caller (A calls B, B calls C, so A is a transitive caller of C)

**Virtual Call:** Call through an interface or polymorphic reference

### Framework & Language Terms

**@Autowired:** Spring annotation for dependency injection

**@GetMapping/@PostMapping:** Spring annotations for HTTP routes

**@app.route:** Flask/FastAPI decorator for defining routes

**axios:** Popular JavaScript HTTP client library

**Constructor Injection:** Passing dependencies through class constructor

**Express:** Node.js web framework

**FastAPI:** Modern Python web framework

**fetch:** Browser/Node.js API for HTTP requests

**Field Injection:** Injecting dependencies directly into class fields

**Jest:** JavaScript testing framework

**NestJS:** TypeScript web framework

**pytest:** Python testing framework

**React Query:** Data fetching library for React

**Spring Boot:** Java web framework

**Stripe:** Payment processing API

### Diagram Terms

**Architecture Diagram:** Visual showing system components grouped by layers

**ASCII Diagram:** Plain text representation of a diagram

**Call Flow Diagram:** Shows function/method call relationships

**Class Diagram:** UML diagram showing classes, interfaces, and inheritance

**Data Flow Diagram (DFD):** Shows how data moves through a system

**Dependency Diagram:** Shows module/package import relationships

**DOT Format:** Graph description language used by Graphviz

**Mermaid:** Markdown-based diagram syntax for GitHub/GitLab

**Participant:** An actor in a sequence diagram (class/module)

**PlantUML:** Text-based diagram description language

**Processor (DFD):** Component that transforms data

**Sequence Diagram:** UML diagram showing method call order

**Sink (DFD):** Where data ends up (save, send, export)

**Source (DFD):** Where data originates (fetch, read, query)

**Subgraph:** A grouped section of a diagram (layer, module)

### Technical Terms

**B-tree Index:** Database index structure for fast lookups

**BFS (Breadth-First Search):** Graph traversal algorithm

**FTS5 (Full-Text Search):** SQLite extension for text search

**O(log n):** Logarithmic time complexity (very fast)

**Qualified Name:** Full name including class/module (e.g., "UserService.save")

**Row Factory:** SQLite feature for accessing results by column name

**SQLite:** Embedded relational database

**Trigger:** Automatic database action when data changes

### Metrics & Quality

**Comment Ratio:** Percentage of lines that are comments

**Complexity Rating:** Quality level (low, moderate, high, very_high)

**Decision Point:** Code branch that creates multiple paths (if, for, while)

**Effort (Halstead):** Estimated time to write/understand code

**Nesting Depth:** How many levels deep code is nested

**Quality Gate:** Threshold that code must meet to pass CI/CD

**SLOC (Source Lines of Code):** Non-blank, non-comment lines

**Technical Debt:** Accumulated complexity and quality issues

**Vocabulary (Halstead):** Unique operators + operands

**Volume (Halstead):** Information content of code

### Ownership & Review

**Blame Data:** Git information about who last modified each line

**Bus Factor:** Number of people who could leave before project fails

**Contributor:** Developer who has committed code

**Expertise Area:** Topic a developer has contributed to frequently

**Pattern Matching:** Wildcard rules in CODEOWNERS (*.py, /src/*)

**Reviewer Suggestion:** Recommended person to review a change with confidence score

**Top Contributor:** Person with most commits in a file or area

---

## Quick Reference

### Common CLI Commands

```bash
# Indexing
repomind index /path/to/repo --repo my-app
repomind index /path/to/repo --incremental  # Only changed files
repomind index /path/to/repo --embedding-model bge-base

# Searching
repomind search "user authentication" --repo my-app
repomind search "database connection" --limit 10 --min-score 0.8

# Context Retrieval
repomind context UserService.save --repo my-app --include-callers
repomind context src/api/users.py --lines 45-67

# Call Graph
repomind find-callers Database.execute --depth 3
repomind find-callees UserController.register --depth 2
repomind detect-cycles --repo my-app --group-by-file

# Dead Code Detection
repomind find-dead-code --repo my-app
repomind find-unreachable --repo my-app

# Route Analysis
repomind analyze-routes --repo my-app
repomind find-route-conflicts --repo backend-api
repomind link-routes --frontend frontend-app --backend backend-api

# Diagram Generation
repomind diagram "user authentication" --type architecture --format mermaid
repomind diagram "payment processing" --type dataflow --format plantuml
repomind diagram "login flow" --type sequence --format ascii

# Security
repomind scan-secrets /path/to/repo
repomind scan-secrets . --severity critical

# Metrics
repomind analyze-complexity --repo my-app
repomind find-hotspots --repo my-app --top 20

# Code Ownership
repomind find-owners src/auth.py
repomind suggest-reviewers --files "*.py"
```

### Common Python API Patterns

**Indexing:**
```python
from repomind.tools.index_repo import index_repository

# Index repository
result = index_repository(
    path="/path/to/repo",
    repo_name="my-app",
    incremental=True
)
```

**Searching:**
```python
from repomind.tools.semantic_grep import semantic_grep

# Semantic search
results = semantic_grep(
    query="user authentication logic",
    repo_name="my-app",
    limit=10,
    min_score=0.7
)
```

**Call Graph:**
```python
from repomind.services.call_graph import CallGraphService

call_graph = CallGraphService()

# Find callers
callers = call_graph.find_callers("UserService.save", max_depth=3)

# Find callees
callees = call_graph.find_callees("UserController.register", max_depth=2)

# Detect cycles
cycles = call_graph.detect_cycles(group_by_file=True)

# Find dead code
entry_points = call_graph.identify_entry_points(chunks)
dead = call_graph.find_unreachable_from_entry_points(entry_points)
```

**Route Analysis:**
```python
from repomind.services.route_registry import RouteRegistry
from repomind.services.route_linker import RouteLinker

registry = RouteRegistry()
linker = RouteLinker()

# Register endpoint
registry.register_endpoint(endpoint)

# Find matches
matches = linker.find_endpoint_matches(api_call)

# Detect conflicts
conflicts = registry.detect_conflicts(repo_name="my-app")
```

**Security Scanning:**
```python
from repomind.services.security_scanner import SecurityScanner

scanner = SecurityScanner()

# Scan file
findings = scanner.scan_file("config.py")

# Scan repository
report = scanner.scan_repository("/path/to/repo")
```

**Metrics:**
```python
from repomind.services.metrics import MetricsService

metrics = MetricsService()

# Analyze function
result = metrics.analyze_function(code, language="python")

# Analyze repository
report = metrics.analyze_repository("my-app")
```

**Ownership:**
```python
from repomind.services.ownership import OwnershipService

ownership = OwnershipService("/path/to/repo")

# Get owners
owners = ownership.get_owners("src/auth.py")

# Suggest reviewers
reviewers = ownership.suggest_reviewers(["file1.py", "file2.py"])
```

**Diagram Generation:**
```python
from repomind.tools.generate_diagrams import (
    generate_diagram,
    generate_architecture_diagram,
    generate_dataflow_diagram,
    generate_callflow_diagram,
    generate_class_diagram,
    generate_sequence_diagram,
    generate_dependency_diagram,
)

# Architecture diagram
result = generate_architecture_diagram(
    query="user authentication",
    repo_name="my-app",
    output_format="mermaid",
    max_nodes=30
)
print(result["diagram"])

# Data flow diagram
result = generate_dataflow_diagram(
    query="order processing",
    repo_name="my-app",
    depth=3
)

# Sequence diagram
result = generate_sequence_diagram(
    query="login flow",
    repo_name="my-app",
    max_participants=10
)

# Class diagram
result = generate_class_diagram(
    query="domain models",
    repo_name="my-app"
)

# Unified function for any type
result = generate_diagram(
    query="payment flow",
    diagram_type="dataflow",  # architecture, dataflow, callflow, class, sequence, dependency
    output_format="plantuml"  # mermaid, plantuml, dot, ascii
)
```

### Common Test Patterns

**Setting Up Tests:**
```python
import pytest
from repomind.services.storage import StorageService
from repomind.services.call_graph import CallGraphService

@pytest.fixture
def storage():
    """Create a fresh storage service for testing."""
    return StorageService()

@pytest.fixture
def call_graph():
    """Create a fresh call graph for testing."""
    return CallGraphService()
```

**Testing Indexing:**
```python
def test_indexing(storage):
    """Test indexing a code file."""
    from repomind.tools.index_repo import index_repository
    
    result = index_repository(
        path="./tests/fixtures/sample_repo",
        repo_name="test-repo"
    )
    
    assert result["chunks_indexed"] > 0
    
    # Verify chunks are stored
    chunks = storage.get_all_chunks(repo_name="test-repo")
    assert len(chunks) > 0
```

**Testing Call Graph:**
```python
def test_call_graph(call_graph):
    """Test call graph relationships."""
    from repomind.services.call_graph import CallRelation
    
    # Add test data
    call_graph.add_call(CallRelation(
        caller="A",
        callee="B",
        caller_file="test.py",
        caller_line=1,
        repo_name="test"
    ))
    
    # Verify
    callers = call_graph.find_callers("B", repo_name="test")
    assert len(callers) == 1
    assert callers[0].caller == "A"
```

### Environment Variables

```bash
# Embedding Configuration
EMBEDDING_MODEL=bge-base  # bge-base, bge-large, codebert, all-MiniLM-L6-v2
EMBEDDING_DIMENSION=768   # Match model dimension

# API Keys (optional)
VOYAGE_API_KEY=your_key   # For Voyage AI embeddings
OPENAI_API_KEY=your_key   # For OpenAI embeddings

# Storage Configuration
INDEX_DIR=~/.repomind     # Where to store indexes
METADATA_DIR=~/.repomind/metadata  # Where to store SQLite DBs

# Performance Tuning
CHUNK_SIZE=1000           # Max tokens per chunk
BATCH_SIZE=32             # Embeddings per batch
MAX_WORKERS=4             # Parallel processing threads
```

### Configuration File (repomind.yaml)

```yaml
# Indexing configuration
index:
  chunk_size: 1000
  overlap: 100
  batch_size: 32
  metadata_dir: ~/.repomind/metadata

# Embedding configuration
embedding:
  provider: local  # local, voyage, openai
  model: bge-base
  dimension: 768
  device: cpu  # cpu, cuda, mps

# Search configuration
search:
  default_limit: 20
  min_score: 0.7
  hybrid_alpha: 0.5  # Balance between semantic and keyword

# Call graph configuration
call_graph:
  max_depth: 5
  track_virtual_calls: true
  enable_di_resolution: true

# Security scanning
security:
  enabled: true
  severity_threshold: medium
  exclude_patterns:
    - "*/node_modules/*"
    - "*/venv/*"
    - "*.test.js"

# Metrics
metrics:
  cyclomatic_threshold: 10
  cognitive_threshold: 15
  comment_ratio_min: 0.1
```

### Debugging Tips

**Enable Debug Logging:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("repomind")
logger.setLevel(logging.DEBUG)
```

**Check Database Contents:**
```python
import sqlite3

# Check call graph
conn = sqlite3.connect("~/.repomind/metadata/callgraph.db")
cursor = conn.execute("SELECT COUNT(*) FROM calls")
print(f"Total call relationships: {cursor.fetchone()[0]}")

# Check symbols
conn = sqlite3.connect("~/.repomind/metadata/symbols.db")
cursor = conn.execute("SELECT COUNT(*) FROM symbols")
print(f"Total symbols: {cursor.fetchone()[0]}")
```

**Verify Embeddings:**
```python
from repomind.services.storage import StorageService

storage = StorageService()
chunks = storage.get_all_chunks(repo_name="my-app")

print(f"Total chunks: {len(chunks)}")
if chunks:
    print(f"Sample chunk: {chunks[0].name}")
    print(f"File: {chunks[0].file_path}")
```

**Performance Profiling:**
```python
import time

start = time.time()
result = semantic_grep("your query", repo_name="my-app")
elapsed = time.time() - start

print(f"Search took {elapsed:.2f}s")
print(f"Results: {len(result)}")
print(f"Results per second: {len(result)/elapsed:.1f}")
```

### Common Troubleshooting

**Problem: "No results found"**
- Check if repository is indexed: `repomind list-repos`
- Try broader query: Use fewer specific terms
- Lower min_score: `--min-score 0.5`
- Re-index: `repomind index /path --repo name`

**Problem: "Call graph empty"**
- Re-index with fresh database
- Check parser supports your language
- Verify function calls are being extracted

**Problem: "Slow searches"**
- Reduce result limit: `--limit 10`
- Use keyword filtering: `--file-pattern "*.py"`
- Consider using smaller embedding model
- Index on SSD, not network drive

**Problem: "Out of memory"**
- Reduce batch size in config
- Use smaller embedding model (all-MiniLM-L6-v2)
- Process fewer files at once
- Increase system swap space

**Problem: "Import errors"**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall: `pip install -e .`
- Check Python version: `python --version` (need 3.11+)

---

## Next Steps

Congratulations on reading this guide! Here's what to do next:

### Week 1: Getting Comfortable
- [ ] Set up development environment
- [ ] Index a small test repository
- [ ] Try all CLI commands
- [ ] Read through key source files

### Week 2: First Contribution
- [ ] Pick a "good first issue"
- [ ] Make a small change
- [ ] Write a test
- [ ] Submit a PR

### Week 3: Deep Dive
- [ ] Trace code flow for indexing
- [ ] Trace code flow for searching
- [ ] Understand embedding process
- [ ] Understand hybrid search

### Week 4: Bigger Contribution
- [ ] Pick a medium-sized feature
- [ ] Design your approach
- [ ] Implement with tests
- [ ] Submit and iterate on feedback

### Week 5: Advanced Call Graph Features
- [ ] Understand call graph architecture
- [ ] Implement custom dead code detection rules
- [ ] Create cycle detection reports
- [ ] Add polymorphic resolution improvements

### Week 6: Integration and Optimization
- [ ] Use call graph in impact analysis
- [ ] Create custom MCP tools using call graph
- [ ] Build automated refactoring suggestions
- [ ] Optimize performance for large codebases

---

## Additional Resources

### Learning Resources

**Python:**
- [Official Python Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/) - Great tutorials

**Git:**
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Learn Git Branching](https://learngitbranching.js.org/) - Interactive

**AI/ML Concepts:**
- [What are embeddings?](https://platform.openai.com/docs/guides/embeddings)
- [Vector databases explained](https://www.pinecone.io/learn/vector-database/)

**Our Tech Stack:**
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [MCP Protocol](https://modelcontextprotocol.io/)

### Project Documentation
- `README.md` - User guide
- `CONTRIBUTING.md` - Contribution guidelines (if exists)
- Code comments - Inline documentation

---

## Feedback

This guide is for you! If something is:
- Unclear
- Missing
- Wrong
- Too technical
- Not technical enough

Please let us know! Open an issue or ask in chat.

---

**Happy Coding! 🚀**

*Last Updated: January 2026*
*Maintained by: RepoMind Team*
