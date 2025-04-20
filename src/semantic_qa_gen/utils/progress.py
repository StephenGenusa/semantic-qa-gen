"""Optimized progress reporting utilities."""

import sys
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Union

try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    LOADING = "Loading document"
    CHUNKING = "Chunking document"
    ANALYSIS = "Analyzing chunks"
    QUESTION_GENERATION = "Generating questions"
    VALIDATION = "Validating questions"
    OUTPUT = "Formatting output"
    COMPLETE = "Processing complete"


class ProgressReporter:
    """
    Reporter for processing progress with rich visual feedback.
    """
    
    def __init__(self, show_progress_bar: bool = True):
        """
        Initialize the progress reporter.
        
        Args:
            show_progress_bar: Whether to show progress bars.
        """
        self.show_progress_bar = show_progress_bar
        self.logger = logging.getLogger(__name__)
        self.current_stage = ProcessingStage.LOADING
        
        # Rich-based progress reporting
        self.rich_enabled = RICH_AVAILABLE and show_progress_bar
        
        if self.rich_enabled:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn()
            )
            self.task_id = None
            self.live_display = None
            self.statistics = {}
        else:
            self.last_progress_print = 0
    
    def update_stage(self, stage: ProcessingStage) -> None:
        """
        Update the current processing stage with visual feedback.
        
        Args:
            stage: New processing stage.
        """
        self.current_stage = stage
        self.logger.info(f"Processing stage: {stage.value}")
        
        if self.rich_enabled:
            # Create a new progress bar for this stage
            if self.live_display:
                self.live_display.stop()
                
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]{stage.value}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn()
            )
            self.task_id = self.progress.add_task(stage.value, total=100)
            self.live_display = Live(self.progress, console=self.console, refresh_per_second=10)
            self.live_display.start()
        else:
            # Simple console output
            if self.show_progress_bar:
                print(f"\n{stage.value}...")
    
    def update_progress(self, completed: int, total: int, 
                      extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress for the current stage with rich visuals.
        
        Args:
            completed: Number of completed items.
            total: Total number of items.
            extra_info: Optional extra information to display.
        """
        if extra_info:
            self.logger.debug(f"Progress: {completed}/{total} - {extra_info}")
            
            if isinstance(extra_info, dict):
                self.statistics.update(extra_info)
        
        if self.rich_enabled:
            # Update the progress bar
            if self.task_id is not None:
                percentage = (completed / total) * 100 if total > 0 else 0
                self.progress.update(self.task_id, completed=percentage, total=100)
                
                # Update description with extra info if provided
                if extra_info:
                    desc_parts = [f"[bold blue]{self.current_stage.value}"]
                    
                    for key, value in extra_info.items():
                        if isinstance(value, (int, float, str)):
                            desc_parts.append(f"[green]{key}[/green]=[yellow]{value}[/yellow]")
                            
                    self.progress.update(self.task_id, description=" ".join(desc_parts))
        else:
            # Simple console progress reporting (limit updates to avoid console spam)
            current_time = time.time()
            if self.show_progress_bar and (current_time - self.last_progress_print >= 0.5 or completed == total):
                progress_percent = (completed / total) * 100 if total > 0 else 0
                
                # Create a simple progress bar
                bar_length = 40
                filled_length = int(bar_length * completed / total) if total > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Add extra info if provided
                extra = ""
                if extra_info and isinstance(extra_info, dict):
                    parts = []
                    for key, value in extra_info.items():
                        if isinstance(value, (int, float, str)):
                            parts.append(f"{key}={value}")
                            
                    if parts:
                        extra = " | " + ", ".join(parts)
                
                print(f"\r{self.current_stage.value}: [{bar}] {progress_percent:.1f}%{extra}", end="")
                sys.stdout.flush()
                self.last_progress_print = current_time
                
                # Print newline when complete
                if completed == total:
                    print()
    
    def complete(self, final_stats: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark processing as complete with a rich summary panel.
        
        Args:
            final_stats: Optional final statistics to display.
        """
        if final_stats:
            self.statistics.update(final_stats)
            
        self.logger.info("Processing complete")
        
        if self.rich_enabled:
            # Stop the live display
            if self.live_display:
                self.live_display.stop()
                
            # Show final statistics
            if self.statistics:
                table = Table(title="Processing Results")
                table.add_column("Statistic", style="cyan")
                table.add_column("Value", style="green")
                
                # Sort statistics by key for consistent display
                for key in sorted(self.statistics.keys()):
                    value = self.statistics[key]
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for sub_key, sub_value in value.items():
                            table.add_row(f"{key} {sub_key}", str(sub_value))
                    else:
                        table.add_row(key, str(value))
                
                self.console.print(Panel(table, title="[bold green]Processing Complete", border_style="green"))
        else:
            if self.show_progress_bar and self.statistics:
                print("\nProcessing complete!")
                for key, value in self.statistics.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for sub_key, sub_value in value.items():
                            print(f"  - {key} {sub_key}: {sub_value}")
                    else:
                        print(f"  - {key}: {value}")
