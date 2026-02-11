use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::borrow::Cow;
use std::error::Error;
use std::io;
use std::io::Stdout;
use std::ops::Deref;
use std::time::Duration;

use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use ratatui::prelude::Stylize;
use ratatui::widgets::Block;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Style,
    text::Line,
    widgets::{Paragraph, Sparkline, Wrap},
};

pub struct Ui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Ui {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        enable_raw_mode()?;

        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    pub fn inference(&mut self, text: &str, durations: &[Duration], token_count: usize) -> Result<(), Box<dyn Error>> {
        let total_duration: Duration = durations.iter().sum();
        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    return Err("interrupted".into());
                }
            }
        }

        let values: Vec<u64> = durations.iter().map(|d| d.as_millis() as u64).collect();
        let max = values.iter().copied().max().unwrap_or(0);
        let latest = values.last().copied().unwrap_or(0);

        self.terminal.draw(|f| {
            let [text_pane, graph_pane] = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),    // wrapped text
                    Constraint::Length(4), // graph
                ])
                .areas(f.area());

            f.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }), text_pane);

            // Graph
            let width = graph_pane.width as usize;
            let data = if values.len() <= width {
                Cow::Borrowed(&values)
            } else {
                // simple bucket max
                let step = values.len() as f64 / width as f64;
                let buckets = (0..width)
                    .map(|i| {
                        let start = (i as f64 * step) as usize;
                        let end = ((i + 1) as f64 * step) as usize;
                        values[start..end].iter().copied().max().unwrap_or(0)
                    })
                    .collect();
                Cow::Owned(buckets)
            };

            // Sparkline (height = 4 by layout)
            let spark = Sparkline::default()
                .cyan()
                .data(data.deref())
                .max(max)
                .block(Block::bordered().blue());
            f.render_widget(spark, graph_pane);

            // Max label aligned to top row
            f.render_widget(
                Paragraph::new(Line::from(format!("┌ max: {max} ms / latest: {latest} ms ")))
                    .style(Style::default())
                    .left_aligned(),
                graph_pane,
            );
            f.render_widget(
                Paragraph::new(Line::from(format!(
                    " total: {total_duration:?} ({token_count} tokens) ┐",
                )))
                .style(Style::default())
                .right_aligned(),
                graph_pane,
            );
        })?;

        Ok(())
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        fn drop_resulty(me: &mut Ui) -> Result<(), Box<dyn Error>> {
            disable_raw_mode()?;
            execute!(me.terminal.backend_mut(), LeaveAlternateScreen)?;
            me.terminal.show_cursor()?;
            Ok(())
        }
        drop_resulty(self).expect("couldn't restore terminal");
    }
}
