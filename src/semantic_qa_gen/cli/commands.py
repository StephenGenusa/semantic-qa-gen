# filename: semantic_qa_gen/cli/commands.py (updated for project structure)

"""Command-line interface for SemanticQAGen."""

import sys
import os
import argparse
import logging
from typing import Dict, Any, Optional, List
import time
import platform
import re

# Check if rich is available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Confirm, Prompt
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
    from rich.syntax import Syntax

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from semantic_qa_gen import SemanticQAGen, __version__
from semantic_qa_gen.utils.error import SemanticQAGenError
from semantic_qa_gen.utils.project import ProjectManager


def print_rich_info(title: str, data: Dict[str, Any]) -> None:
    """
    Print information using rich formatting.

    Args:
        title: Title for the panel.
        data: Data to display.
    """
    if not RICH_AVAILABLE:
        # Fallback to plain text
        print(f"\n{title}:")
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    - {sub_key}: {sub_value}")
            else:
                print(f"  - {key}: {value}")
        return

    # Create a rich table
    console = Console()
    table = Table(show_header=False)
    table.add_column("Property", style="cyan", width=30)
    table.add_column("Value", style="green")

    for key, value in data.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            table.add_row(key, "")
            for sub_key, sub_value in value.items():
                table.add_row(f"  {sub_key}", str(sub_value))
        else:
            table.add_row(key, str(value))

    console.print(Panel(table, title=f"[bold blue]{title}"))


def display_system_info() -> None:
    """Display system information for diagnostics."""
    if not RICH_AVAILABLE:
        print("\nSystem Information:")
        print(f"  - Python: {platform.python_version()}")
        print(f"  - OS: {platform.system()} {platform.release()}")
        print(f"  - SemanticQAGen: {__version__}")
        return

    console = Console()
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="green")

    # Python details
    table.add_row("Python", f"{platform.python_version()} ({platform.python_implementation()})")

    # OS details
    table.add_row("Operating System", f"{platform.system()} {platform.release()}")

    # SemanticQAGen details
    table.add_row("SemanticQAGen", __version__)

    # Check for available libraries
    libraries = {
        "rich": "✓ Installed" if RICH_AVAILABLE else "✗ Not Found",
    }

    try:
        import nltk
        libraries["nltk"] = f"✓ Installed ({nltk.__version__})"
    except (ImportError, AttributeError):
        libraries["nltk"] = "✗ Not Found"

    try:
        import pymupdf
        libraries["pymupdf"] = f"✓ Installed ({pymupdf.__version__})"
    except (ImportError, AttributeError):
        libraries["pymupdf"] = "✗ Not Found"

    try:
        import docx
        libraries["python-docx"] = f"✓ Installed"
    except ImportError:
        libraries["python-docx"] = "✗ Not Found"

    for lib_name, status in libraries.items():
        table.add_row(f"Library: {lib_name}", status)

    console.print(Panel(table))


def interactive_config() -> Dict[str, Any]:
    """
    Interactive configuration creator.

    Returns:
        Configuration dictionary.
    """
    if not RICH_AVAILABLE:
        print("Interactive configuration requires 'rich' package.")
        print("Install with: pip install rich")
        return {}

    console = Console()
    console.print(Panel("[bold green]SemanticQAGen Interactive Configuration", border_style="green"))

    config = {}

    # LLM Services
    console.print("\n[bold cyan]LLM Services Configuration[/bold cyan]")

    # Remote LLM
    use_remote = Confirm.ask("Use remote LLM service (like OpenAI)?", default=True)
    if use_remote:
        remote_config = {}
        remote_config["provider"] = Prompt.ask(
            "Provider",
            choices=["openai", "azure", "anthropic", "other"],
            default="openai"
        )

        # For Azure, ask for Azure-specific settings
        if remote_config["provider"] == "azure":
            azure_endpoint = Prompt.ask("Azure OpenAI Endpoint URL", default="https://your-resource.openai.azure.com/")
            remote_config["api_base"] = azure_endpoint

            api_version = Prompt.ask("API Version", default="2023-12-01-preview")
            remote_config["api_version"] = api_version

            use_azure_ad = Confirm.ask("Use Azure AD authentication?", default=False)
            if not use_azure_ad:
                # Get API key with masked input
                api_key = Prompt.ask("API Key", password=True)
                if api_key:
                    remote_config["api_key"] = api_key

            # Azure deployment name as model
            model = Prompt.ask("Azure Deployment Name", default="gpt-4")
            remote_config["model"] = model
        else:
            # Get API key with masked input
            api_key = Prompt.ask("API Key", password=True)
            if api_key:
                remote_config["api_key"] = api_key

            # Model selection based on provider
            if remote_config["provider"] == "openai":
                model_choices = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
                default_model = "gpt-4"
            elif remote_config["provider"] == "anthropic":
                model_choices = ["claude-3-opus", "claude-3-sonnet", "claude-2.1"]
                default_model = "claude-3-sonnet"
            else:
                model_choices = None
                default_model = ""

            remote_config["model"] = Prompt.ask(
                "Model name",
                choices=model_choices,
                default=default_model
            )

        # Task routing
        console.print("\nWhich tasks should use the remote LLM service?")
        tasks = []
        if Confirm.ask("Use for analysis?", default=True):
            tasks.append("analysis")
        if Confirm.ask("Use for question generation?", default=True):
            tasks.append("generation")
        if Confirm.ask("Use for validation?", default=False):
            tasks.append("validation")

        remote_config["preferred_tasks"] = tasks

        # Add to config
        config["llm_services"] = {"remote": remote_config}

    # Local LLM
    use_local = Confirm.ask("Use local LLM service (like Ollama)?", default=True)
    if use_local:
        local_config = {}
        local_config["url"] = Prompt.ask("API URL", default="http://localhost:11434/api")
        local_config["model"] = Prompt.ask("Model name", default="mistral:7b")

        # Task routing
        console.print("\nWhich tasks should use the local LLM service?")
        tasks = []
        if Confirm.ask("Use for analysis?", default=not use_remote):
            tasks.append("analysis")
        if Confirm.ask("Use for validation?", default=True):
            tasks.append("validation")
        if Confirm.ask("Use for question generation?", default=not use_remote):
            tasks.append("generation")

        local_config["preferred_tasks"] = tasks

        # Add to config or update existing
        if "llm_services" not in config:
            config["llm_services"] = {}
        config["llm_services"]["local"] = local_config

    # Chunking
    console.print("\n[bold cyan]Chunking Configuration[/bold cyan]")
    chunking_config = {}
    chunking_config["strategy"] = Prompt.ask(
        "Chunking strategy",
        choices=["semantic", "fixed_size", "hybrid"],
        default="semantic"
    )
    chunking_config["target_chunk_size"] = int(Prompt.ask("Target chunk size (tokens)", default="1500"))
    chunking_config["preserve_headings"] = Confirm.ask("Preserve headings in chunks?", default=True)

    config["chunking"] = chunking_config

    # Question Generation
    console.print("\n[bold cyan]Question Generation Configuration[/bold cyan]")
    qgen_config = {}
    qgen_config["max_questions_per_chunk"] = int(Prompt.ask("Maximum questions per chunk", default="10"))

    # Question category distribution
    console.print("\nQuestion category distribution:")
    category_config = {}

    factual_min = int(Prompt.ask("Minimum factual questions per chunk", default="2"))
    inferential_min = int(Prompt.ask("Minimum inferential questions per chunk", default="1"))
    conceptual_min = int(Prompt.ask("Minimum conceptual questions per chunk", default="1"))

    category_config["factual"] = {"min_questions": factual_min, "weight": 1.0}
    category_config["inferential"] = {"min_questions": inferential_min, "weight": 1.2}
    category_config["conceptual"] = {"min_questions": conceptual_min, "weight": 1.5}

    qgen_config["categories"] = category_config
    qgen_config["adaptive_generation"] = Confirm.ask(
        "Enable adaptive question generation based on content density?",
        default=True
    )

    config["question_generation"] = qgen_config

    # Output
    console.print("\n[bold cyan]Output Configuration[/bold cyan]")
    output_config = {}
    output_config["format"] = Prompt.ask("Default output format", choices=["json", "csv"], default="json")
    output_config["include_metadata"] = Confirm.ask("Include metadata in output?", default=True)
    output_config["include_statistics"] = Confirm.ask("Include statistics in output?", default=True)

    config["output"] = output_config

    # Processing
    console.print("\n[bold cyan]Processing Configuration[/bold cyan]")
    proc_config = {}
    proc_config["concurrency"] = int(Prompt.ask("Concurrency level (1-5)", default="3"))
    proc_config["enable_checkpoints"] = Confirm.ask("Enable processing checkpoints?", default=True)

    # Set log level based on verbosity
    verbosity = Prompt.ask("Verbosity level", choices=["quiet", "normal", "verbose", "debug"], default="normal")
    if verbosity == "quiet":
        proc_config["log_level"] = "WARNING"
    elif verbosity == "normal":
        proc_config["log_level"] = "INFO"
    elif verbosity == "verbose":
        proc_config["log_level"] = "DEBUG"
    elif verbosity == "debug":
        proc_config["log_level"] = "DEBUG"
        proc_config["debug_mode"] = True

    config["processing"] = proc_config

    # Show summary
    console.print("\n[bold green]Configuration Summary[/bold green]")

    summary_table = Table()
    summary_table.add_column("Section", style="cyan")
    summary_table.add_column("Setting", style="green")

    # Add LLM services
    if "llm_services" in config:
        if "remote" in config["llm_services"]:
            remote = config["llm_services"]["remote"]
            model = remote.get("model", "not specified")
            summary_table.add_row("Remote LLM", f"{remote.get('provider', 'unknown')} ({model})")
        if "local" in config["llm_services"]:
            local = config["llm_services"]["local"]
            summary_table.add_row("Local LLM", f"{local.get('model', 'not specified')}")

    # Add chunking
    if "chunking" in config:
        chunking = config["chunking"]
        summary_table.add_row("Chunking", f"{chunking.get('strategy')} (size: {chunking.get('target_chunk_size')})")

    # Add question generation
    if "question_generation" in config:
        qgen = config["question_generation"]
        summary_table.add_row("Questions", f"Max {qgen.get('max_questions_per_chunk')} per chunk")

    # Add output
    if "output" in config:
        output = config["output"]
        summary_table.add_row("Output", f"{output.get('format', 'json').upper()} format")

    # Show the summary
    console.print(summary_table)

    # Confirm
    if not Confirm.ask("\nSave this configuration?", default=True):
        console.print("[yellow]Configuration cancelled.[/yellow]")
        return {}

    return config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="SemanticQAGen - Generate high-quality question-answer pairs from text documents"
    )

    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create project command
    create_parser = subparsers.add_parser('create-project', help='Create a new QAGenProject structure')
    create_parser.add_argument('path', nargs='?', help='Path for the new project (default: current directory)')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process a document and generate questions')
    process_parser.add_argument('document', help='Path to the document file')
    process_parser.add_argument('-o', '--output', help='Path for output file')
    process_parser.add_argument('-f', '--format', choices=['json', 'csv'],
                                help='Output format (default: from config)')
    process_parser.add_argument('-c', '--config', help='Path to config file')
    process_parser.add_argument('-p', '--project', help='Path to QAGenProject directory')
    process_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    # Init config command
    config_parser = subparsers.add_parser('init-config', help='Create a default configuration file')
    config_parser.add_argument('output', help='Path for the config file')
    config_parser.add_argument('-i', '--interactive', action='store_true',
                               help='Create config interactively')
    config_parser.add_argument('-p', '--project', help='Path to QAGenProject directory')

    # Interactive config command
    subparsers.add_parser('interactive', help='Run in interactive mode')

    # List supported file formats
    subparsers.add_parser('formats', help='List supported file formats')

    # System info command
    subparsers.add_parser('info', help='Show system information')

    # Version command
    subparsers.add_parser('version', help='Show the version and exit')

    return parser.parse_args()


def list_supported_formats() -> None:
    """Display supported file formats."""
    formats = {
        "Text": [".txt", ".text"],
        "Markdown": [".md", ".markdown"],
        "PDF": [".pdf"],
        "DOCX": [".docx"]
    }

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Supported File Formats")
        table.add_column("Format Type", style="cyan")
        table.add_column("Extensions", style="green")
        table.add_column("Requirements", style="yellow")

        # Add rows for each format
        table.add_row("Text", ", ".join(formats["Text"]), "Built-in")
        table.add_row("Markdown", ", ".join(formats["Markdown"]), "commonmark")

        # Check if PDF support is available
        try:
            import pymupdf
            pdf_req = "pymupdf ✓"
        except ImportError:
            pdf_req = "pymupdf ✗ (not installed)"
        table.add_row("PDF", ", ".join(formats["PDF"]), pdf_req)

        # Check if DOCX support is available
        try:
            import docx
            docx_req = "python-docx ✓"
        except ImportError:
            docx_req = "python-docx ✗ (not installed)"
        table.add_row("DOCX", ", ".join(formats["DOCX"]), docx_req)

        console.print(table)
    else:
        print("\nSupported File Formats:")
        for format_name, extensions in formats.items():
            print(f"  - {format_name}: {', '.join(extensions)}")
        print("\nNote: Some formats may require additional libraries:")
        print("  - PDF: requires pymupdf")
        print("  - DOCX: requires python-docx")


def main() -> int:
    """
    Main entry point for the CLI interface.

    Returns:
        Exit code.
    """
    # Create a Rich console if available
    console = Console() if RICH_AVAILABLE else None

    args = parse_arguments()

    # Default command if none specified
    if args.command is None:
        if console:
            console.print("[bold red]Error:[/bold red] No command specified.")
            console.print("Run with [bold]--help[/bold] for usage information.")
        else:
            print("Error: No command specified.")
            print("Run with --help for usage information.")
        return 1

    # Version command
    if args.command == 'version':
        if console:
            console.print(f"[bold green]SemanticQAGen[/bold green] version [bold yellow]{__version__}[/bold yellow]")
        else:
            print(f"SemanticQAGen version {__version__}")
        return 0

    # System info command
    if args.command == 'info':
        display_system_info()
        return 0

    # List formats command
    if args.command == 'formats':
        list_supported_formats()
        return 0

    # Create project command
    if args.command == 'create-project':
        try:
            project_manager = ProjectManager()
            project_path = args.path

            # Create the project
            created_path = project_manager.create_project_structure(project_path)

            if console:
                console.print(f"[green]Project created successfully at:[/green] {created_path}")
                console.print("\nProject structure:")
                console.print("  - [cyan]config/[/cyan]: Configuration files")
                console.print("  - [cyan]config/prompts/[/cyan]: Prompt templates")
                console.print("  - [cyan]input/[/cyan]: Place input documents here")
                console.print("  - [cyan]output/[/cyan]: Generated output files")
                console.print("  - [cyan]logs/[/cyan]: Log files")
                console.print("  - [cyan]temp/[/cyan]: Temporary files and checkpoints")
            else:
                print(f"Project created successfully at: {created_path}")
                print("\nProject structure:")
                print("  - config/: Configuration files")
                print("  - config/prompts/: Prompt templates")
                print("  - input/: Place input documents here")
                print("  - output/: Generated output files")
                print("  - logs/: Log files")
                print("  - temp/: Temporary files and checkpoints")

            return 0
        except Exception as e:
            if console:
                console.print(f"[bold red]Error creating project:[/bold red] {e}")
            else:
                print(f"Error creating project: {e}")
            return 1

    # Interactive mode
    if args.command == 'interactive':
        if not RICH_AVAILABLE:
            print("Interactive mode requires 'rich' package.")
            print("Install with: pip install rich")
            return 1

        # This would be a full interactive CLI - we'll implement a basic version
        console.print(Panel("[bold green]SemanticQAGen Interactive Mode", border_style="green"))

        # First, find or create project
        project_manager = ProjectManager()
        existing_project = project_manager.find_project_root()

        if existing_project:
            console.print(f"[green]Using existing project:[/green] {existing_project}")
            project_path = existing_project
        else:
            console.print("[yellow]No project structure found.[/yellow]")
            if Confirm.ask("Create a new project?", default=True):
                project_dir = Prompt.ask("Project directory", default="QAGenProject")
                project_path = project_manager.create_project_structure(project_dir)
                console.print(f"[green]Project created at:[/green] {project_path}")
            else:
                console.print("[red]Project required for operation.[/red]")
                return 1

        # Get action
        action = Prompt.ask(
            "What would you like to do?",
            choices=["process", "config", "info", "exit"],
            default="process"
        )

        if action == "process":
            # Let user browse for document
            document = Prompt.ask("Document path (relative to project/input or absolute)")

            # Check various paths
            abs_doc_path = os.path.abspath(document)
            project_input_path = os.path.join(project_path, "input", document)

            if os.path.exists(abs_doc_path):
                document_path = abs_doc_path
            elif os.path.exists(project_input_path):
                document_path = project_input_path
            else:
                console.print(f"[bold red]Error:[/bold red] Document not found at either:\n"
                              f"  - {abs_doc_path}\n"
                              f"  - {project_input_path}")
                return 1

            # Get output path
            base_name = os.path.basename(document_path)
            name_without_ext = os.path.splitext(base_name)[0]
            default_output = name_without_ext
            output = Prompt.ask("Output filename (without extension)", default=default_output)

            # Get format
            format = Prompt.ask("Output format", choices=["json", "csv"], default="json")

            # Process document
            try:
                # Initialize with project path
                qa_gen = SemanticQAGen(verbose=True, project_path=project_path)

                with Progress(
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                        console=console
                ) as progress:
                    task = progress.add_task("[cyan]Processing document...", total=100)

                    # Process the document
                    result = qa_gen.process_document(document_path)
                    progress.update(task, completed=100)

                # Save the questions
                saved_path = qa_gen.save_questions(result, output, format)

                # Print statistics
                stats = result.get('statistics', {})
                total_questions = stats.get('total_valid_questions', 0)
                chunks_processed = stats.get('total_chunks', 0)

                stats_data = {
                    "Document": document_path,
                    "Chunks processed": chunks_processed,
                    "Questions generated": total_questions,
                    "Output saved to": saved_path
                }

                # Add category breakdown if available
                if 'categories' in stats:
                    stats_data["Question categories"] = stats['categories']

                print_rich_info("Processing Complete", stats_data)

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                return 1

        elif action == "config":
            # Interactive config
            config = interactive_config()
            if config:
                output = Prompt.ask("Config filename", default="system.yaml")
                config_path = os.path.join(project_path, "config", output)
                try:
                    qa_gen = SemanticQAGen(config_dict=config, project_path=project_path)
                    qa_gen.create_default_config_file(config_path)
                    console.print(f"[green]Configuration saved to:[/green] {config_path}")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    return 1

        elif action == "info":
            display_system_info()

        return 0

    # Init config command
    if args.command == 'init-config':
        try:
            # Determine project path
            project_path = args.project
            if not project_path:
                # Try to find existing project
                project_manager = ProjectManager()
                detected_project = project_manager.find_project_root()
                if detected_project:
                    project_path = detected_project

            if args.interactive and RICH_AVAILABLE:
                # Interactive config creation
                config = interactive_config()
                qa_gen = SemanticQAGen(config_dict=config, project_path=project_path)
                qa_gen.create_default_config_file(args.output)

                if console:
                    console.print(f"[green]Configuration created at:[/green] {args.output}")
                else:
                    print(f"Configuration created at: {args.output}")
            else:
                # Default config creation
                qa_gen = SemanticQAGen(project_path=project_path)
                qa_gen.create_default_config_file(args.output)

                if console:
                    console.print(f"[green]Default configuration created at:[/green] {args.output}")
                else:
                    print(f"Default configuration created at: {args.output}")
            return 0
        except SemanticQAGenError as e:
            if console:
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                print(f"Error: {e}")
            return 1

    # Process command
    if args.command == 'process':
        try:
            if console:
                console.print(Panel(f"[bold]Processing document:[/bold] {args.document}", title="SemanticQAGen"))
            else:
                print(f"Processing document: {args.document}")

            # Initialize SemanticQAGen with project path if provided
            qa_gen = SemanticQAGen(
                config_path=args.config,
                verbose=args.verbose,
                project_path=args.project
            )

            # Process the document
            result = qa_gen.process_document(args.document)

            # Determine output path
            output_path = args.output
            if not output_path:
                base_name = os.path.basename(args.document)
                name_without_ext = os.path.splitext(base_name)[0]
                output_path = f"{name_without_ext}_questions"

            # Save the questions
            saved_path = qa_gen.save_questions(result, output_path, args.format)

            # Print statistics
            stats = result.get('statistics', {})
            total_questions = stats.get('total_valid_questions', 0)
            chunks_processed = stats.get('total_chunks', 0)

            stats_data = {
                "Document": args.document,
                "Chunks processed": chunks_processed,
                "Questions generated": total_questions,
                "Output saved to": saved_path
            }

            # Add category breakdown if available
            if 'categories' in stats:
                stats_data["Question categories"] = stats['categories']

            # Add processing time if available
            if 'performance' in stats:
                perf = stats['performance']
                if 'total_seconds' in perf:
                    stats_data["Processing time"] = f"{perf['total_seconds']:.1f} seconds"

            if console:
                print_rich_info("Processing Complete", stats_data)
            else:
                print(f"Processing complete!")
                for key, value in stats_data.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    - {sub_key}: {sub_value}")
                    else:
                        print(f"  - {key}: {value}")

            return 0

        except SemanticQAGenError as e:
            if console:
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                print(f"Error: {e}")
            return 1
        except Exception as e:
            if console:
                console.print(f"[bold red]Unexpected error:[/bold red] {e}")
            else:
                print(f"Unexpected error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
