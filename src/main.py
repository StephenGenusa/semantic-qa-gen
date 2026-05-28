"""
Main entry point for SemanticQAGen CLI with enhanced help and error reporting.

This module provides the command-line interface for SemanticQAGen,
allowing users to process documents, create configurations, and
get information about the library.
"""

from semantic_qa_gen.cli.commands import main

if __name__ == "__main__":
    import sys
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(2)
