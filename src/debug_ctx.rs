use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::usize;

/// a single debug log entry with level, text, and timestamp
#[derive(Clone, Debug)]
pub struct DbgLine {
    pub level: u8,
    pub text: String,
    pub timestamp: Instant,
}

/// thread-safe hierarchical debug context
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

#[derive(Debug, Clone)]
pub struct DebugSummary {
    pub label: String,
    pub num_lines: usize,
    pub num_children: usize,
    pub elapsed_secs: f64,
}

impl ExpDebugCtx {
    /// create a new root context
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

    /// log a message at a specific verbosity level
    pub fn log(&self, level: u8, text: impl Into<String>) {
        let mut inner = self.inner.lock().unwrap();
        inner.lines.push(DbgLine {
            level,
            text: text.into(),
            timestamp: Instant::now(),
        });
    }

    /// conditionally log based on global verbosity level
    pub fn vlog(&self, level: u8, text: impl Into<String>) {
        if crate::verbosity::enabled(level) {
            self.log(level, text);
        }
    }

    /// create and return a child context
    pub fn child(&self, label: impl Into<String>) -> ExpDebugCtx {
        let child = ExpDebugCtx::new(label);
        let mut inner = self.inner.lock().unwrap();
        inner.children.push(child.clone());
        child
    }

    /// render the tree structure respecting max_level and max_depth
    ///   max_depth < 0 means no limit
    pub fn render(&self, max_level: u8, max_depth: isize) -> String {
        let max_depth = usize::try_from(max_depth).unwrap_or(usize::MAX);
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

        // render label with indentation
        let indent = "  ".repeat(current_depth);

        // calculate display time: duration for this context (creation to last activity)
        let last_time = self.get_end_time(&inner);
        let display_time_ms = last_time.duration_since(self.created_at).as_millis();

        output.push_str(&format!(
            "{}[{}] (t={}ms)\n",
            indent, inner.label, display_time_ms
        ));

        // render lines that meet verbosity threshold,
        // time relative to this context creation
        for line in &inner.lines {
            if line.level <= max_level {
                let line_time_ms = line.timestamp.duration_since(self.created_at).as_millis();
                output.push_str(&format!("{}  +{}ms: {}\n", indent, line_time_ms, line.text));
            }
        }

        // render children:
        // pass this context's creation time as parent_time
        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_recursive_with_parent(
                    &child_inner,
                    max_level,
                    max_depth,
                    current_depth + 1,
                    output,
                    self.created_at, // pass parent's creation time
                );
            }
        }
    }

    fn render_recursive_with_parent(
        &self,
        inner: &ExpDebugCtxInner,
        max_level: u8,
        max_depth: usize,
        current_depth: usize,
        output: &mut String,
        _parent_time: Instant,
    ) {
        if current_depth > max_depth {
            return;
        }

        let indent = "  ".repeat(current_depth);

        // show duration for this context
        let last_time = self.get_end_time(&inner);
        let display_time_ms = last_time.duration_since(self.created_at).as_millis();

        output.push_str(&format!(
            "{}[{}] (t={}ms)\n",
            indent, inner.label, display_time_ms
        ));

        // render lines that meet verbosity threshold
        // time relative to this context creation
        for line in &inner.lines {
            if line.level <= max_level {
                let line_time_ms = line.timestamp.duration_since(self.created_at).as_millis();
                output.push_str(&format!("{}  +{}ms: {}\n", indent, line_time_ms, line.text));
            }
        }

        // render children,
        // pass this context's creation time as parent_time
        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_recursive_with_parent(
                    &child_inner,
                    max_level,
                    max_depth,
                    current_depth + 1,
                    output,
                    self.created_at,
                );
            }
        }
    }

    /// get the latest activity time in this context or any child
    fn get_end_time(&self, inner: &ExpDebugCtxInner) -> Instant {
        // start with the last log line in this context, or creation time if no logs
        let own_end = inner
            .lines
            .last()
            .map(|line| line.timestamp)
            .unwrap_or(self.created_at);

        // check all children for their end times
        let children_end = inner
            .children
            .iter()
            .map(|child| {
                let child_inner = child.inner.lock().unwrap();
                child.get_end_time(&child_inner)
            })
            .max()
            .unwrap_or(self.created_at);

        // return the maximum
        own_end.max(children_end)
    }

    /// get summary statistics for this context
    pub fn summary(&self) -> DebugSummary {
        let inner = self.inner.lock().unwrap();
        DebugSummary {
            label: inner.label.clone(),
            num_lines: inner.lines.len(),
            num_children: inner.children.len(),
            elapsed_secs: self.created_at.elapsed().as_secs_f64(),
        }
    }

    /// recursively count all log lines across the tree
    pub fn total_lines(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        let own = inner.lines.len();
        let children_total: usize = inner.children.iter().map(|c| c.total_lines()).sum();
        own + children_total
    }

    /// get maximum depth of the tree
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

    /// render a summary showing only structure and counts,
    /// no detailed log lines
    pub fn render_summary(&self, max_depth: usize) -> String {
        let inner = self.inner.lock().unwrap();
        let mut output = String::new();
        let root_time = self.created_at;
        self.render_summary_recursive(&inner, max_depth, 0, &mut output, root_time);
        output
    }

    fn render_summary_recursive(
        &self,
        inner: &ExpDebugCtxInner,
        max_depth: usize,
        current_depth: usize,
        output: &mut String,
        root_time: Instant,
    ) {
        if current_depth > max_depth {
            return;
        }

        let indent = "  ".repeat(current_depth);

        // all contexts show their duration:
        // creation to last activity
        let last_time = self.get_end_time(&inner);
        let display_time_ms = last_time.duration_since(self.created_at).as_millis();

        let summary = format!(
            "{}[{}] ({} lines, {} children, t={}ms)\n",
            indent,
            inner.label,
            inner.lines.len(),
            inner.children.len(),
            display_time_ms
        );
        output.push_str(&summary);

        if current_depth < max_depth {
            for child in &inner.children {
                let child_inner = child.inner.lock().unwrap();
                child.render_summary_recursive(
                    &child_inner,
                    max_depth,
                    current_depth + 1,
                    output,
                    root_time, // keep passing root_time for consistency, though not used
                );
            }
        }
    }

    /// render with sampling: show first n and last m lines
    /// from each context
    pub fn render_sampled(
        &self,
        max_level: u8,
        max_depth: usize,
        first: usize,
        last: usize,
    ) -> String {
        let inner = self.inner.lock().unwrap();
        let mut output = String::new();
        let root_time = self.created_at;
        self.render_sampled_recursive(
            &inner,
            max_level,
            max_depth,
            first,
            last,
            0,
            &mut output,
            root_time,
        );
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
        root_time: Instant,
    ) {
        if current_depth > max_depth {
            return;
        }

        let indent = "  ".repeat(current_depth);

        // all contexts show their duration:
        // creation to last activity
        let last_time = self.get_end_time(&inner);
        let display_time_ms = last_time.duration_since(self.created_at).as_millis();

        output.push_str(&format!(
            "{}[{}] (t={}ms)\n",
            indent, inner.label, display_time_ms
        ));

        // filter lines by level
        let visible_lines: Vec<_> = inner
            .lines
            .iter()
            .filter(|line| line.level <= max_level)
            .collect();

        let total = visible_lines.len();
        if total == 0 {
            // no lines to show
        } else if total <= first + last {
            // show all lines
            for line in visible_lines {
                let line_time_ms = line.timestamp.duration_since(self.created_at).as_millis();
                output.push_str(&format!("{}  +{}ms: {}\n", indent, line_time_ms, line.text));
            }
        } else {
            // show first n lines
            for line in visible_lines.iter().take(first) {
                let line_time_ms = line.timestamp.duration_since(self.created_at).as_millis();
                output.push_str(&format!("{}  +{}ms: {}\n", indent, line_time_ms, line.text));
            }

            // show ellipsis
            let skipped = total - first - last;
            output.push_str(&format!(
                "{}  ... ({} lines omitted) ...\n",
                indent, skipped
            ));

            // show last m lines
            for line in visible_lines.iter().skip(total - last) {
                let line_time_ms = line.timestamp.duration_since(self.created_at).as_millis();
                output.push_str(&format!("{}  +{}ms: {}\n", indent, line_time_ms, line.text));
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
                    root_time, // keep passing root_time for consistency
                );
            }
        }
    }
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

    #[test]
    fn test_get_end_time_no_activity() {
        // Context with no logs and no children should return creation time
        let ctx = ExpDebugCtx::new("empty");
        let inner = ctx.inner.lock().unwrap();
        let end_time = ctx.get_end_time(&inner);

        // End time should be the creation time (0 duration)
        assert_eq!(end_time, ctx.created_at);
    }

    #[test]
    fn test_get_end_time_with_logs() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "first");
        thread::sleep(std::time::Duration::from_millis(5));
        ctx.log(1, "second");
        thread::sleep(std::time::Duration::from_millis(5));
        ctx.log(1, "third");

        let inner = ctx.inner.lock().unwrap();
        let end_time = ctx.get_end_time(&inner);

        // End time should be the timestamp of the last log line
        assert_eq!(end_time, inner.lines[2].timestamp);
    }

    #[test]
    fn test_get_end_time_with_children() {
        let root = ExpDebugCtx::new("root");
        root.log(1, "root_log");

        thread::sleep(std::time::Duration::from_millis(10));
        let child = root.child("child");
        thread::sleep(std::time::Duration::from_millis(5));
        child.log(1, "child_log");

        let root_inner = root.inner.lock().unwrap();
        let end_time = root.get_end_time(&root_inner);

        // End time should be from the child's log (the latest activity)
        let child_inner = child.inner.lock().unwrap();
        let child_end = child.get_end_time(&child_inner);
        assert_eq!(end_time, child_end);
    }

    #[test]
    fn test_duration_calculation() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "start");
        thread::sleep(std::time::Duration::from_millis(20));
        ctx.log(1, "end");

        let inner = ctx.inner.lock().unwrap();
        let last_time = ctx.get_end_time(&inner);
        let duration_ms = last_time.duration_since(ctx.created_at).as_millis();

        // Duration should be at least 20ms
        assert!(duration_ms >= 20);
    }

    #[test]
    fn test_hierarchical_duration() {
        let root = ExpDebugCtx::new("root");

        let child1 = root.child("child1");
        thread::sleep(std::time::Duration::from_millis(10));
        child1.log(1, "child1_log");

        thread::sleep(std::time::Duration::from_millis(10));
        let child2 = root.child("child2");
        thread::sleep(std::time::Duration::from_millis(10));
        child2.log(1, "child2_log");

        // Root duration should include all children
        let root_inner = root.inner.lock().unwrap();
        let root_end = root.get_end_time(&root_inner);
        let root_duration = root_end.duration_since(root.created_at).as_millis();

        // Child2's log is the last activity, so root duration should be at least 30ms
        assert!(root_duration >= 30);
    }

    #[test]
    fn test_render_summary_structure() {
        let root = ExpDebugCtx::new("root");
        root.log(1, "root_log");

        let child1 = root.child("child1");
        child1.log(1, "child1_log1");
        child1.log(1, "child1_log2");

        let child2 = root.child("child2");
        child2.log(1, "child2_log");

        let summary = root.render_summary(10);

        // Should contain all context names
        assert!(summary.contains("[root]"));
        assert!(summary.contains("[child1]"));
        assert!(summary.contains("[child2]"));

        // Should show counts
        assert!(summary.contains("1 lines")); // root has 1 line
        assert!(summary.contains("2 lines")); // child1 has 2 lines
        assert!(summary.contains("2 children")); // root has 2 children

        // Should NOT contain actual log text
        assert!(!summary.contains("root_log"));
        assert!(!summary.contains("child1_log1"));
    }

    #[test]
    fn test_render_sampled_all_lines() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "line1");
        ctx.log(1, "line2");
        ctx.log(1, "line3");

        // Request first 2 and last 2, but only 3 lines total
        let output = ctx.render_sampled(1, 10, 2, 2);

        // Should show all lines (no ellipsis)
        assert!(output.contains("line1"));
        assert!(output.contains("line2"));
        assert!(output.contains("line3"));
        assert!(!output.contains("omitted"));
    }

    #[test]
    fn test_render_sampled_with_ellipsis() {
        let ctx = ExpDebugCtx::new("test");
        for i in 0..10 {
            ctx.log(1, format!("line{}", i));
        }

        // Request first 2 and last 2 from 10 lines
        let output = ctx.render_sampled(1, 10, 2, 2);

        // Should show first 2
        assert!(output.contains("line0"));
        assert!(output.contains("line1"));

        // Should show ellipsis
        assert!(output.contains("6 lines omitted"));

        // Should show last 2
        assert!(output.contains("line8"));
        assert!(output.contains("line9"));

        // Should NOT show middle lines
        assert!(!output.contains("line5"));
    }

    #[test]
    fn test_render_respects_verbosity() {
        let ctx = ExpDebugCtx::new("test");
        ctx.log(1, "verbose_level_1");
        ctx.log(2, "verbose_level_2");
        ctx.log(3, "verbose_level_3");

        // Render with max_level=1
        let output = ctx.render(1, 10);
        assert!(output.contains("verbose_level_1"));
        assert!(!output.contains("verbose_level_2"));
        assert!(!output.contains("verbose_level_3"));

        // Render with max_level=2
        let output = ctx.render(2, 10);
        assert!(output.contains("verbose_level_1"));
        assert!(output.contains("verbose_level_2"));
        assert!(!output.contains("verbose_level_3"));
    }

    #[test]
    fn test_vlog_conditional() {
        // This test can't easily verify vlog behavior since it depends on
        // the global verbosity level, but we can at least call it
        let ctx = ExpDebugCtx::new("test");
        ctx.vlog(1, "conditional_log");

        // The log may or may not be present depending on global verbosity
        // Just verify the context still works
        assert_eq!(ctx.summary().label, "test");
    }

    #[test]
    fn test_deeply_nested_contexts() {
        let root = ExpDebugCtx::new("root");
        let l1 = root.child("level1");
        let l2 = l1.child("level2");
        let l3 = l2.child("level3");
        let l4 = l3.child("level4");
        l4.log(1, "deep_log");

        // Test max_depth calculation
        assert_eq!(root.max_depth(), 5);

        // Test rendering with depth limit
        let output = root.render(10, 2);
        assert!(output.contains("[root]"));
        assert!(output.contains("[level1]"));
        assert!(output.contains("[level2]"));
        assert!(!output.contains("[level3]")); // Depth limit
    }

    #[test]
    fn test_concurrent_child_creation() {
        let root = ExpDebugCtx::new("root");

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let root_clone = root.clone();
                thread::spawn(move || {
                    let child = root_clone.child(format!("thread_{}", i));
                    for j in 0..5 {
                        child.log(1, format!("log_{}", j));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should have 8 children
        assert_eq!(root.summary().num_children, 8);

        // Total lines should be 8 * 5 = 40
        assert_eq!(root.total_lines(), 40);
    }

    #[test]
    fn test_render_with_isize_max_depth() {
        let root = ExpDebugCtx::new("root");
        let child = root.child("child");
        child.log(1, "test");

        // Test with negative depth (should convert to usize::MAX, no limit)
        let output = root.render(10, -1);
        assert!(output.contains("[root]"));
        assert!(output.contains("[child]"));
        assert!(output.contains("test"));
    }

    #[test]
    fn test_empty_context_render() {
        let ctx = ExpDebugCtx::new("empty");

        let output = ctx.render(10, 10);
        assert!(output.contains("[empty]"));
        assert!(output.contains("t=0ms")); // No activity, duration is 0

        let summary = ctx.render_summary(10);
        assert!(summary.contains("0 lines"));
        assert!(summary.contains("0 children"));
    }

    #[test]
    fn test_context_with_only_children() {
        let root = ExpDebugCtx::new("root");
        let child = root.child("child");
        child.log(1, "child_log");

        // Root has no direct logs, only child activity
        let root_inner = root.inner.lock().unwrap();
        assert_eq!(root_inner.lines.len(), 0);

        // But root duration should include child's activity
        let end_time = root.get_end_time(&root_inner);
        assert!(end_time > root.created_at);
    }

    #[test]
    fn test_render_time_offsets() {
        let ctx = ExpDebugCtx::new("test");
        thread::sleep(std::time::Duration::from_millis(5));
        ctx.log(1, "log1");
        thread::sleep(std::time::Duration::from_millis(10));
        ctx.log(1, "log2");

        let output = ctx.render(10, 10);

        // Should contain time offsets for log lines
        assert!(output.contains("+"));
        assert!(output.contains("ms:"));

        // Parse and verify timing relationships
        assert!(output.contains("log1"));
        assert!(output.contains("log2"));
    }
}
