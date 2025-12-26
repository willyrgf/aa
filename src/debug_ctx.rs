use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A single debug log entry with level, text, and timestamp
#[derive(Clone, Debug)]
pub struct DbgLine {
    pub level: u8,
    pub text: String,
    pub timestamp: Instant,
}

/// Thread-safe hierarchical debug context
#[derive(Clone, Debug)]
pub struct ExpDebugCtx {
    inner: Arc<Mutex<ExpDebugCtxInner>>,
    created_at: Instant,
}

#[derive(Debug)]
struct ExpDebugCtxInner {
    label: String,
    lines: Vec<DbgLine>,
    children: Vec<ExpDebugCtx>,
}

impl ExpDebugCtx {
    /// Create a new root context
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ExpDebugCtxInner {
                label: label.into(),
                lines: Vec::new(),
                children: Vec::new(),
            })),
            created_at: Instant::now(),
        }
    }

    /// Log a message at a specific verbosity level
    pub fn log(&self, level: u8, text: impl Into<String>) {
        let mut inner = self.inner.lock().unwrap();
        inner.lines.push(DbgLine {
            level,
            text: text.into(),
            timestamp: Instant::now(),
        });
    }

    /// Conditionally log based on global verbosity level
    pub fn vlog(&self, level: u8, text: impl Into<String>) {
        if crate::verbosity::enabled(level) {
            self.log(level, text);
        }
    }

    /// Create and return a child context
    pub fn child(&self, label: impl Into<String>) -> ExpDebugCtx {
        let child = ExpDebugCtx::new(label);
        let mut inner = self.inner.lock().unwrap();
        inner.children.push(child.clone());
        child
    }

    /// Render the tree structure respecting max_level and max_depth
    pub fn render(&self, max_level: u8, max_depth: usize) -> String {
        let inner = self.inner.lock().unwrap();
        let mut output = String::new();
        self.render_recursive(&inner, max_level, max_depth, 0, &mut output);
        output
    }

    fn render_recursive(
        &self,
        inner: &ExpDebugCtxInner,
        max_level: u8,
        max_depth: usize,
        current_depth: usize,
        output: &mut String,
    ) {
        if current_depth > max_depth {
            return;
        }

        // Render label with indentation and elapsed time
        let indent = "  ".repeat(current_depth);
        let elapsed = self.created_at.elapsed().as_secs_f64();
        output.push_str(&format!("{}[{}] (t={:.2}s)\n", indent, inner.label, elapsed));

        // Render lines that meet verbosity threshold
        for line in &inner.lines {
            if line.level <= max_level {
                let line_elapsed = line.timestamp.duration_since(self.created_at).as_secs_f64();
                output.push_str(&format!(
                    "{}  +{:.2}s: {}\n",
                    indent, line_elapsed, line.text
                ));
            }
        }

        // Render children
        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_recursive(
                    &child_inner,
                    max_level,
                    max_depth,
                    current_depth + 1,
                    output,
                );
            }
        }
    }

    /// Get summary statistics for this context
    pub fn summary(&self) -> DebugSummary {
        let inner = self.inner.lock().unwrap();
        DebugSummary {
            label: inner.label.clone(),
            num_lines: inner.lines.len(),
            num_children: inner.children.len(),
            elapsed_secs: self.created_at.elapsed().as_secs_f64(),
        }
    }

    /// Recursively count all log lines across the tree
    pub fn total_lines(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        let own = inner.lines.len();
        let children_total: usize = inner.children.iter().map(|c| c.total_lines()).sum();
        own + children_total
    }

    /// Get maximum depth of the tree
    pub fn max_depth(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        if inner.children.is_empty() {
            1
        } else {
            1 + inner
                .children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(0)
        }
    }

    /// Render a summary showing only structure and counts, no detailed log lines
    pub fn render_summary(&self, max_depth: usize) -> String {
        let inner = self.inner.lock().unwrap();
        let mut output = String::new();
        self.render_summary_recursive(&inner, max_depth, 0, &mut output);
        output
    }

    fn render_summary_recursive(
        &self,
        inner: &ExpDebugCtxInner,
        max_depth: usize,
        current_depth: usize,
        output: &mut String,
    ) {
        if current_depth > max_depth {
            return;
        }

        let indent = "  ".repeat(current_depth);
        let elapsed = self.created_at.elapsed().as_secs_f64();
        let summary = format!(
            "{}[{}] ({} lines, {} children, {:.2}s)\n",
            indent,
            inner.label,
            inner.lines.len(),
            inner.children.len(),
            elapsed
        );
        output.push_str(&summary);

        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_summary_recursive(&child_inner, max_depth, current_depth + 1, output);
            }
        }
    }

    /// Render with sampling: show first N and last M lines from each context
    pub fn render_sampled(&self, max_level: u8, max_depth: usize, first: usize, last: usize) -> String {
        let inner = self.inner.lock().unwrap();
        let mut output = String::new();
        self.render_sampled_recursive(&inner, max_level, max_depth, first, last, 0, &mut output);
        output
    }

    fn render_sampled_recursive(
        &self,
        inner: &ExpDebugCtxInner,
        max_level: u8,
        max_depth: usize,
        first: usize,
        last: usize,
        current_depth: usize,
        output: &mut String,
    ) {
        if current_depth > max_depth {
            return;
        }

        let indent = "  ".repeat(current_depth);
        let elapsed = self.created_at.elapsed().as_secs_f64();
        output.push_str(&format!("{}[{}] (t={:.2}s)\n", indent, inner.label, elapsed));

        // Filter lines by level
        let visible_lines: Vec<_> = inner
            .lines
            .iter()
            .filter(|line| line.level <= max_level)
            .collect();

        let total = visible_lines.len();
        if total == 0 {
            // No lines to show
        } else if total <= first + last {
            // Show all lines
            for line in visible_lines {
                let line_elapsed = line.timestamp.duration_since(self.created_at).as_secs_f64();
                output.push_str(&format!(
                    "{}  +{:.2}s: {}\n",
                    indent, line_elapsed, line.text
                ));
            }
        } else {
            // Show first N lines
            for line in visible_lines.iter().take(first) {
                let line_elapsed = line.timestamp.duration_since(self.created_at).as_secs_f64();
                output.push_str(&format!(
                    "{}  +{:.2}s: {}\n",
                    indent, line_elapsed, line.text
                ));
            }

            // Show ellipsis
            let skipped = total - first - last;
            output.push_str(&format!("{}  ... ({} lines omitted) ...\n", indent, skipped));

            // Show last M lines
            for line in visible_lines.iter().skip(total - last) {
                let line_elapsed = line.timestamp.duration_since(self.created_at).as_secs_f64();
                output.push_str(&format!(
                    "{}  +{:.2}s: {}\n",
                    indent, line_elapsed, line.text
                ));
            }
        }

        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_sampled_recursive(
                    &child_inner,
                    max_level,
                    max_depth,
                    first,
                    last,
                    current_depth + 1,
                    output,
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugSummary {
    pub label: String,
    pub num_lines: usize,
    pub num_children: usize,
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_single_thread_logging() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "line1");
        ctx.log(2, "line2");

        let summary = ctx.summary();
        assert_eq!(summary.num_lines, 2);
        assert_eq!(summary.label, "test");
    }

    #[test]
    fn test_multi_thread_logging() {
        let ctx = ExpDebugCtx::new("root");
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let child = ctx.child(format!("thread_{}", i));
                thread::spawn(move || {
                    for j in 0..10 {
                        child.log(1, format!("line_{}", j));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(ctx.summary().num_children, 4);

        // Verify each child has 10 lines
        let inner = ctx.inner.lock().unwrap();
        for child in &inner.children {
            assert_eq!(child.summary().num_lines, 10);
        }
    }

    #[test]
    fn test_render_respects_max_level() {
        let ctx = ExpDebugCtx::new("root");
        ctx.log(1, "level1");
        ctx.log(3, "level3");

        let output_level1 = ctx.render(1, 10);
        assert!(output_level1.contains("level1"));
        assert!(!output_level1.contains("level3"));

        let output_level3 = ctx.render(3, 10);
        assert!(output_level3.contains("level1"));
        assert!(output_level3.contains("level3"));
    }

    #[test]
    fn test_render_respects_max_depth() {
        let root = ExpDebugCtx::new("root");
        let child1 = root.child("child1");
        let child2 = child1.child("child2");
        child2.log(1, "deep");

        let shallow = root.render(10, 1);
        assert!(shallow.contains("root"));
        assert!(shallow.contains("child1"));
        assert!(!shallow.contains("child2"));

        let deep = root.render(10, 3);
        assert!(deep.contains("root"));
        assert!(deep.contains("child1"));
        assert!(deep.contains("child2"));
        assert!(deep.contains("deep"));
    }

    #[test]
    fn test_summary_stats() {
        let root = ExpDebugCtx::new("root");
        root.log(1, "root_line");

        let child1 = root.child("child1");
        child1.log(1, "child1_line1");
        child1.log(1, "child1_line2");

        let child2 = root.child("child2");
        child2.log(1, "child2_line");

        // Test total_lines
        assert_eq!(root.total_lines(), 4); // 1 + 2 + 1

        // Test max_depth
        assert_eq!(root.max_depth(), 2); // root -> child

        // Test summary
        let summary = root.summary();
        assert_eq!(summary.num_lines, 1);
        assert_eq!(summary.num_children, 2);
    }

    #[test]
    fn test_nested_max_depth() {
        let root = ExpDebugCtx::new("root");
        let c1 = root.child("c1");
        let c2 = c1.child("c2");
        let c3 = c2.child("c3");
        c3.log(1, "deep");

        assert_eq!(root.max_depth(), 4);
        assert_eq!(c1.max_depth(), 3);
        assert_eq!(c2.max_depth(), 2);
        assert_eq!(c3.max_depth(), 1);
    }

    #[test]
    fn test_timestamps_increase() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "first");
        thread::sleep(std::time::Duration::from_millis(10));
        ctx.log(1, "second");

        let inner = ctx.inner.lock().unwrap();
        assert!(inner.lines[1].timestamp > inner.lines[0].timestamp);
    }
}
