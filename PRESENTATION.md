# SmellAI system design presentation

This directory contains a Slidev presentation describing the multi-agent system for code smell detection, prioritization, and refactoring.

## Running the presentation

### Install dependencies

```bash
npm install
```

### Start development server

```bash
npm run dev
```

This will open the presentation in your browser at http://localhost:3030

### Build for production

```bash
npm run build
```

The static site will be generated in the `dist/` directory.

### Export to PDF

```bash
npm run export
```

This will generate a PDF version of the presentation.

## Presentation structure

The presentation covers:

1. System overview
2. Multi-agent architecture (6 agents)
3. Smell prioritization algorithm and dependency rules
4. Visualization of dependency graphs
5. Evaluation framework and datasets
6. Code metrics and impact analysis
7. Implementation technologies
8. Complete workflow execution

## Key visualizations

The presentation includes:

- `smell_priority_graph.png` - Overall priority and dependency graph with different shapes for smell types
- `smell_deps_OrderProcessor.png` - File-level dependency view for OrderProcessor.java
- `smell_deps_ReportGenerator.png` - File-level dependency view for ReportGenerator.java

These visualizations demonstrate how the system models positive and negative dependencies between code smells and uses this information to determine optimal refactoring sequences.

## Navigation

- Use arrow keys to navigate between slides
- Press `f` for fullscreen mode
- Press `o` for overview mode
- Press `d` for dark mode toggle

## Customization

Edit `slides.md` to modify the presentation content. The presentation uses Slidev's default theme with Markdown syntax and supports:

- Mermaid diagrams
- LaTeX math equations
- Code highlighting
- Two-column layouts
- Custom CSS classes
