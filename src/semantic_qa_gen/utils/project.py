# filename: semantic_qa_gen/utils/project.py

"""Project structure utilities for SemanticQAGen."""

import os
import logging
import shutil
from typing import Optional, List, Dict, Any
import yaml

from semantic_qa_gen.utils.error import ConfigurationError


class ProjectManager:
    """
    Manages SemanticQAGen project structure and files.

    Handles creation of project directories, configuration files,
    and templates in a standardized structure.
    """

    DEFAULT_PROJECT_NAME = "QAGenProject"

    # Standard directory structure
    DIRECTORIES = {
        "config": "Configuration files",
        "config/prompts": "Prompt templates",
        "input": "Input documents",
        "output": "Generated output files",
        "logs": "Log files",
        "temp": "Temporary files and checkpoints"
    }

    def __init__(self):
        """Initialize the ProjectManager."""
        self.logger = logging.getLogger(__name__)

    def create_project_structure(self, project_path: Optional[str] = None) -> str:
        """
        Create a standard QAGenProject directory structure.

        Args:
            project_path: Path where the project will be created.
                          If None, creates in current directory.

        Returns:
            Absolute path to the created project directory.

        Raises:
            ConfigurationError: If project creation fails.
        """
        try:
            # Determine project directory
            if project_path is None:
                project_dir = os.path.join(os.getcwd(), self.DEFAULT_PROJECT_NAME)
            else:
                project_dir = os.path.abspath(project_path)

            self.logger.info(f"Creating SemanticQAGen project at: {project_dir}")

            # Create main directory and subdirectories
            for subdir, description in self.DIRECTORIES.items():
                dir_path = os.path.join(project_dir, subdir)
                os.makedirs(dir_path, exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path} ({description})")

            # Copy default templates to project
            self._copy_default_templates(project_dir)

            # Create default system configuration
            self._create_default_system_config(project_dir)

            self.logger.info(f"Project structure created successfully at {project_dir}")
            return project_dir

        except PermissionError as e:
            raise ConfigurationError(f"Permission denied creating project: {e}")
        except OSError as e:
            raise ConfigurationError(f"Error creating project structure: {e}")

    # filename: semantic_qa_gen/utils/project.py
    # Fix for _copy_default_templates method

    def _copy_default_templates(self, project_dir: str) -> None:
        """
        Copy default prompt templates to project config/prompts directory.
        ONLY if they don't already exist.

        Args:
            project_dir: Project directory path.
        """
        # Locate the package's template directory
        package_templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "llm", "prompts", "templates"
        )

        target_dir = os.path.join(project_dir, "config", "prompts")

        if not os.path.exists(package_templates_dir):
            self.logger.warning("Default templates directory not found in package.")
            return

        try:
            for template_file in os.listdir(package_templates_dir):
                if not template_file.endswith(('.yaml', '.yml')):
                    continue

                source = os.path.join(package_templates_dir, template_file)
                destination = os.path.join(target_dir, template_file)

                # CHECK IF FILE EXISTS FIRST - DON'T OVERWRITE!
                if os.path.exists(destination):
                    self.logger.debug(f"Template {template_file} already exists. Not overwriting.")
                    continue

                shutil.copy2(source, destination)
                self.logger.debug(f"Copied new template: {template_file}")
        except Exception as e:
            self.logger.error(f"Error copying templates: {e}")

    def _create_default_system_config(self, project_dir: str) -> None:
        """
        Create a default system configuration file in the project.
        ONLY if it doesn't already exist.

        Args:
            project_dir: Project directory path.
        """
        # Define the path for the config file
        config_path = os.path.join(project_dir, "config", "system.yaml")

        # CHECK IF FILE EXISTS FIRST - DON'T OVERWRITE!
        if os.path.exists(config_path):
            self.logger.info(f"Config file {config_path} already exists. Not overwriting existing configuration.")
            return

        try:
            from semantic_qa_gen.config.manager import ConfigManager

            # Only create new config if doesn't exist
            config_manager = ConfigManager()

            # Update paths to match project structure
            config = config_manager.config

            # Update default paths to match project structure (relative to project dir)
            config.processing.checkpoint_dir = "./temp/checkpoints"
            config.processing.log_level = "INFO"
            config.output.output_dir = "./output"

            # Ensure log file path exists and is set
            os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)

            # Save the modified config only for NEW configs
            if not os.path.isfile(config_path):
                config_manager.save_config(config_path, include_comments=True)

            self.logger.debug(f"Created new default system configuration at {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default system configuration: {e}")

    def find_project_root(self, start_path: Optional[str] = None) -> Optional[str]:
        """
        Find the root of a QAGenProject by searching for the standard structure.

        Args:
            start_path: Path to start searching from. Uses current directory if None.

        Returns:
            Path to project root if found, None otherwise.
        """
        if start_path is None:
            start_path = os.getcwd()

        current_path = os.path.abspath(start_path)

        # Check if this is a project directory
        if self._is_project_directory(current_path):
            return current_path

        # Check parent directories (stop at filesystem root)
        prev_path = None
        while current_path != prev_path:
            if self._is_project_directory(current_path):
                return current_path

            # Move up one directory
            prev_path = current_path
            current_path = os.path.dirname(current_path)

        # Project root not found
        return None

    def _is_project_directory(self, path: str) -> bool:
        """
        Check if a directory has the QAGenProject structure.

        Args:
            path: Directory path to check.

        Returns:
            True if path follows project structure.
        """
        # Check for key directories and files that would indicate a project
        required_dirs = ["config", "input", "output"]
        config_file = os.path.join(path, "config", "system.yaml")
        prompts_dir = os.path.join(path, "config", "prompts")

        # Directory must exist and contain required subdirectories
        if not os.path.isdir(path):
            return False

        for req_dir in required_dirs:
            if not os.path.isdir(os.path.join(path, req_dir)):
                return False

        # Check if either config file or prompts directory exists
        return os.path.isfile(config_file) or os.path.isdir(prompts_dir)

    def resolve_project_path(self, path: str, base_dir: str = "") -> str:
        """
        Resolve a project-relative path to an absolute path.

        Args:
            path: Path, potentially relative to project root.
            base_dir: Base directory within project (e.g., "input", "output").

        Returns:
            Absolute path.
        """
        # If path is already absolute, return it
        if os.path.isabs(path):
            return path

        # If path starts with ./ or ../, it's relative to current directory
        if path.startswith(('./', '../')):
            return os.path.abspath(path)

        # Find project root
        project_root = self.find_project_root()
        if not project_root:
            # No project found, treat as relative to current directory
            return os.path.abspath(path)

        # Path is relative to project or specified base directory
        if base_dir:
            return os.path.join(project_root, base_dir, path)
        else:
            return os.path.join(project_root, path)
