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

## Glossary

**AST (Abstract Syntax Tree):** A tree representation of code structure

**Chunk:** A meaningful piece of code (function, class, method)

**ChromaDB:** The vector database we use for similarity search

**Embedding:** A list of numbers representing text meaning

**Hybrid Search:** Combining semantic search with keyword matching

**MCP (Model Context Protocol):** Standard for connecting AI tools

**Parser:** Code that reads source code and extracts structure

**Query Expansion:** Adding related terms to search queries

**Semantic Search:** Finding by meaning, not exact text

**Tree-sitter:** Library for parsing multiple programming languages

**Vector Database:** Database optimized for similarity search

**Virtual Environment:** Isolated Python environment for a project

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
