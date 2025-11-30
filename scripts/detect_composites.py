#!/usr/bin/env python3
"""
Script to detect and analyze composite refactorings from RefactoringMiner data.

This script analyzes refactoring operations to identify composite refactorings,
which are higher-level refactoring patterns composed of multiple atomic refactorings.
It supports detecting:
- Class Decomposition (multiple moves from one class)
- Method Decomposition (extracting multiple methods from one)
- Method Composition (combining methods, etc.)

Usage:
    uv run python scripts/rminer_composite.py /path/to/data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src directory to Python path
# Add repo root to Python path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from smellai.data.rminer_utils import load_rminer_data
from smellai.models.refactoring import RMinerCommit, Refactoring

import networkx as nx

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


EXTRACT_METHOD = "Extract Method"
MOVE_METHOD = "Move Method"
INLINE_METHOD = "Inline Method"
RENAME_METHOD = "Rename Method"
PULL_UP_METHOD = "Pull Up Method"
PUSH_DOWN_METHOD = "Push Down Method"
MOVE_CLASS = "Move Class"
RENAME_CLASS = "Rename Class"

class RefactoringParser:
    """Parses RefactoringMiner descriptions to extract source and target entities."""

    def __init__(self):
        # Regex patterns for common refactorings
        # Note: These are heuristics and might not cover all cases perfectly
        self.patterns = {
            EXTRACT_METHOD: re.compile(r"Extract Method\s+(.+?)\s+extracted from\s+(.+?)\s+in class\s+(.+)"),
            MOVE_METHOD: re.compile(r"Move Method\s+(.+?)\s+from class\s+(.+?)\s+to\s+(.+?)\s+from class\s+(.+)"),
            INLINE_METHOD: re.compile(r"Inline Method\s+(.+?)\s+inlined to\s+(.+?)\s+in class\s+(.+)"),
            RENAME_METHOD: re.compile(r"Rename Method\s+(.+?)\s+renamed to\s+(.+?)\s+in class\s+(.+)"),
            PULL_UP_METHOD: re.compile(r"Pull Up Method\s+(.+?)\s+from class\s+(.+?)\s+to\s+(.+?)\s+from class\s+(.+)"),
            PUSH_DOWN_METHOD: re.compile(r"Push Down Method\s+(.+?)\s+from class\s+(.+?)\s+to\s+(.+?)\s+from class\s+(.+)"),
            MOVE_CLASS: re.compile(r"Move Class\s+(.+?)\s+moved to\s+(.+)"),
            RENAME_CLASS: re.compile(r"Rename Class\s+(.+?)\s+renamed to\s+(.+)"),
        }

    def parse(self, refactoring: Refactoring) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract source and target entity names from refactoring description.
        
        Returns:
            Tuple of (source_entity, target_entity). Returns (None, None) if parsing fails.
        """
        # If location data is available, use it (preferred)
        if refactoring.left_side_locations and refactoring.right_side_locations:
            # Simplified: take the first location's code element or file path
            source = self._get_location_name(refactoring.left_side_locations[0])
            target = self._get_location_name(refactoring.right_side_locations[0])
            return source, target

        # Fallback to description parsing
        pattern = self.patterns.get(refactoring.type)
        if not pattern:
            return None, None

        match = pattern.search(refactoring.description)
        if not match:
            return None, None

        groups = match.groups()
        
        if refactoring.type == EXTRACT_METHOD:
            # 1: extracted method, 2: source method, 3: class
            # Source is the original method, Target is the new extracted method
            # But for "Decomposition", we view the original method as the source of the split
            extracted_method = groups[0]
            source_method = groups[1]
            class_name = groups[2]
            return f"{class_name}.{source_method}", f"{class_name}.{extracted_method}"

        elif refactoring.type in [MOVE_METHOD, PULL_UP_METHOD, PUSH_DOWN_METHOD]:
            # 1: method name (might change?), 2: source class, 3: method name (after), 4: target class
            method_before = groups[0]
            source_class = groups[1]
            method_after = groups[2]
            target_class = groups[3]
            return f"{source_class}.{method_before}", f"{target_class}.{method_after}"

        elif refactoring.type == INLINE_METHOD:
            # 1: inlined method, 2: target method, 3: class
            inlined_method = groups[0]
            target_method = groups[1]
            class_name = groups[2]
            return f"{class_name}.{inlined_method}", f"{class_name}.{target_method}"
            
        elif refactoring.type == RENAME_METHOD:
            method_before = groups[0]
            method_after = groups[1]
            class_name = groups[2]
            return f"{class_name}.{method_before}", f"{class_name}.{method_after}"

        elif refactoring.type in [MOVE_CLASS, RENAME_CLASS]:
            return groups[0], groups[1]

        return None, None

    def _get_location_name(self, loc) -> str:
        if loc.code_element:
            return f"{loc.file_path}:{loc.code_element}"
        return f"{loc.file_path}:{loc.start_line}"


class CompositeAnalyzer:
    """Analyzes refactorings to find composites."""

    def __init__(self, commits: List[RMinerCommit]):
        self.commits = commits
        self.parser = RefactoringParser()

    def analyze(self) -> Dict[str, Any]:
        results = {
            "class_decomposition": [],
            "method_decomposition": [],
            "method_composition": [],
            "stats": defaultdict(int)
        }

        # Analyze per commit (atomic composites)
        for commit in self.commits:
            self._analyze_commit(commit, results)

        return results

    def _analyze_commit(self, commit: RMinerCommit, results: Dict[str, Any]):
        # Build a graph for this commit
        G = nx.MultiDiGraph()
        
        refactorings_by_type = defaultdict(list)

        for ref in commit.refactorings:
            source, target = self.parser.parse(ref)
            if source and target:
                G.add_edge(source, target, type=ref.type, refactoring=ref)
                refactorings_by_type[ref.type].append((source, target, ref))

        self._detect_class_decomposition(commit, refactorings_by_type, results)
        self._detect_method_decomposition(commit, refactorings_by_type, results)
        self._detect_method_composition(commit, refactorings_by_type, results)

    def _detect_class_decomposition(self, commit, refactorings_by_type, results):
        # 1. Class Decomposition (Move Method)
        # One source class -> Multiple target classes (or multiple methods moved from one class)
        # Heuristic: Multiple Move Method operations from the same source class
        move_methods = refactorings_by_type[MOVE_METHOD]
        if move_methods:
            source_classes = defaultdict(list)
            for src, tgt, ref in move_methods:
                # Extract class name from method signature (simplified)
                src_class = src.split('(')[0].rpartition('.')[0]
                source_classes[src_class].append(ref)
            
            for src_class, refs in source_classes.items():
                if len(refs) >= 2: # Threshold for decomposition
                    results["class_decomposition"].append({
                        "repository": commit.repository,
                        "commit": commit.sha1,
                        "source_class": src_class,
                        "refactorings": [r.description for r in refs],
                        "count": len(refs)
                    })
                    results["stats"]["class_decomposition"] += 1

    def _detect_method_decomposition(self, commit, refactorings_by_type, results):
        # 2. Method Decomposition (Extract Method)
        # One source method -> Multiple extracted methods
        extract_methods = refactorings_by_type[EXTRACT_METHOD]
        if extract_methods:
            source_methods = defaultdict(list)
            for src, tgt, ref in extract_methods:
                source_methods[src].append(ref)
            
            for src_method, refs in source_methods.items():
                if len(refs) >= 2:
                    results["method_decomposition"].append({
                        "repository": commit.repository,
                        "commit": commit.sha1,
                        "source_method": src_method,
                        "refactorings": [r.description for r in refs],
                        "count": len(refs)
                    })
                    results["stats"]["method_decomposition"] += 1

    def _detect_method_composition(self, commit, refactorings_by_type, results):
        # 3. Method Composition (Inline Method)
        # Multiple methods inlined into one target method
        inline_methods = refactorings_by_type[INLINE_METHOD]
        if inline_methods:
            target_methods = defaultdict(list)
            for src, tgt, ref in inline_methods:
                target_methods[tgt].append(ref)
            
            for tgt_method, refs in target_methods.items():
                if len(refs) >= 2:
                    results["method_composition"].append({
                        "repository": commit.repository,
                        "commit": commit.sha1,
                        "target_method": tgt_method,
                        "refactorings": [r.description for r in refs],
                        "count": len(refs)
                    })
                    results["stats"]["method_composition"] += 1


def visualize_composites(results: Dict[str, Any]):
    """Print a text-based visualization of composite refactorings."""
    print("\n" + "="*60)
    print("COMPOSITE REFACTORING VISUALIZATION")
    print("="*60)

    # Class Decomposition
    if results["class_decomposition"]:
        print(f"\n[Class Decomposition] Found {len(results['class_decomposition'])} instances")
        for i, item in enumerate(results["class_decomposition"], 1):
            print(f"\n{i}. Source Class: \033[1m{item['source_class']}\033[0m")
            print(f"   Repo: {item['repository']} | Commit: {item['commit'][:7]}")
            for ref in item["refactorings"]:
                print(f"   ├── {ref}")

    # Method Decomposition
    if results["method_decomposition"]:
        print(f"\n[Method Decomposition] Found {len(results['method_decomposition'])} instances")
        for i, item in enumerate(results["method_decomposition"], 1):
            print(f"\n{i}. Source Method: \033[1m{item['source_method']}\033[0m")
            print(f"   Repo: {item['repository']} | Commit: {item['commit'][:7]}")
            for ref in item["refactorings"]:
                print(f"   ├── {ref}")

    # Method Composition
    if results["method_composition"]:
        print(f"\n[Method Composition] Found {len(results['method_composition'])} instances")
        for i, item in enumerate(results["method_composition"], 1):
            print(f"\n{i}. Target Method: \033[1m{item['target_method']}\033[0m")
            print(f"   Repo: {item['repository']} | Commit: {item['commit'][:7]}")
            for ref in item["refactorings"]:
                print(f"   ├── {ref}")
    
    print("\n" + "="*60 + "\n")


def save_composite_diagrams(results: Dict[str, Any], output_dir: Path):
    """Generates and saves separate diagrams for each detected composite refactoring."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.error("matplotlib is required for diagram generation. Please install it.")
        return

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Collect all items
    items = []
    if results["class_decomposition"]:
        items.extend([("Class_Decomposition", item) for item in results["class_decomposition"]])
    if results["method_decomposition"]:
        items.extend([("Method_Decomposition", item) for item in results["method_decomposition"]])
    if results["method_composition"]:
        items.extend([("Method_Composition", item) for item in results["method_composition"]])

    if not items:
        LOGGER.warning("No composites to visualize.")
        return

    LOGGER.info(f"Generating {len(items)} diagrams in {output_dir}...")

    for i, (type_name, item) in enumerate(items, 1):
        plt.figure(figsize=(10, 6))
        G = nx.DiGraph()
        
        # Add nodes and edges based on type
        if type_name == "Class_Decomposition":
            center = item['source_class'].split('.')[-1]
            G.add_node(center, color='lightblue', node_type='Source Class')
            for ref in item['refactorings']:
                # Heuristic parsing for visualization
                target = "Target"
                if " to " in ref and " from class " in ref:
                    parts = ref.split(" from class ")
                    if len(parts) > 2:
                        target = parts[-1].strip().split('.')[-1]
                
                method_name = "method"
                if "Move Method" in ref:
                    try:
                        method_name = ref.split("Move Method")[1].split(" from class")[0].strip().split('(')[0]
                    except IndexError:
                        pass

                G.add_edge(center, target, label=method_name)
                
        elif type_name == "Method_Decomposition":
            center = item['source_method'].split('(')[0].split('.')[-1]
            G.add_node(center, color='lightblue', node_type='Source Method')
            for ref in item['refactorings']:
                extracted = "Extracted"
                if "Extract Method" in ref:
                    try:
                        extracted = ref.split("Extract Method")[1].split(" extracted from")[0].strip().split('(')[0].split(' ')[-1]
                    except IndexError:
                        pass
                G.add_edge(center, extracted, label="extracts")

        elif type_name == "Method_Composition":
            center = item['target_method'].split('(')[0].split('.')[-1]
            G.add_node(center, color='lightblue', node_type='Target Method')
            for ref in item['refactorings']:
                inlined = "Inlined"
                if "Inline Method" in ref:
                    try:
                        inlined = ref.split("Inline Method")[1].split(" inlined to")[0].strip().split('(')[0].split(' ')[-1]
                    except IndexError:
                        pass
                G.add_edge(inlined, center, label="inlines")

        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold', arrows=True, edge_color='gray')
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        commit_short = item['commit'][:7]
        plt.title(f"{type_name.replace('_', ' ')}\nRepo: {item['repository'].split('/')[-1]} | Commit: {commit_short}", fontsize=10)
        
        filename = f"{type_name}_{commit_short}_{i}.png"
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()
        
    LOGGER.info(f"Saved {len(items)} diagrams to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detect composite refactorings")
    parser.add_argument("data_json", type=Path, help="Path to RefactoringMiner data.json")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--visualize", action="store_true", help="Visualize composites in terminal")
    parser.add_argument("--save-diagrams-dir", type=Path, help="Directory to save diagrams (e.g. composites_viz)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if not args.data_json.exists():
        LOGGER.error(f"File not found: {args.data_json}")
        sys.exit(1)

    LOGGER.info(f"Loading data from {args.data_json}")
    commits = load_rminer_data(str(args.data_json))
    
    LOGGER.info(f"Analyzing {len(commits)} commits for composite refactorings...")
    analyzer = CompositeAnalyzer(commits)
    results = analyzer.analyze()

    LOGGER.info("Analysis complete.")
    LOGGER.info("Composite Refactorings Found:")
    for key, count in results["stats"].items():
        LOGGER.info(f"  {key}: {count}")

    # Visualize composites in terminal
    if args.visualize:
        visualize_composites(results)

    if args.save_diagrams_dir:
        save_composite_diagrams(results, args.save_diagrams_dir)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"Results saved to {args.output}")
    else:
        # Print sample results if no output file
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
