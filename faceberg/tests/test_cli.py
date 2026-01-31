"""Tests for CLI commands."""

from click.testing import CliRunner

from faceberg import config as cfg
from faceberg.cli import main


def test_list_command_with_tree_view(tmp_path):
    """Test list command uses CatalogTreeView for rich display."""
    # Create a local catalog
    catalog_dir = tmp_path / "test_catalog"
    catalog_dir.mkdir()

    # Create config with some tables
    config = cfg.Config()
    config["default"] = cfg.Namespace()
    config["default"]["imdb"] = cfg.Dataset(repo="stanfordnlp/imdb", config="plain_text")
    config["default"]["squad"] = cfg.Dataset(repo="squad", config="plain_text")
    config["analytics"] = cfg.Namespace()
    config["analytics"]["aggregated"] = cfg.Table()

    # Save config
    config.to_yaml(catalog_dir / "faceberg.yml")

    # Run list command
    runner = CliRunner()
    result = runner.invoke(main, [str(catalog_dir), "list"])

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify output contains catalog name in rich format
    output = result.output
    assert "📁" in output or "test_catalog" in output

    # Verify namespaces are shown
    assert "default" in output
    assert "analytics" in output

    # Verify dataset nodes are shown with their icons
    assert "imdb" in output
    assert "squad" in output
    assert "aggregated" in output

    # Verify dataset metadata is shown (repo info)
    assert "stanfordnlp/imdb" in output or "🤗" in output


def test_list_command_empty_catalog(tmp_path):
    """Test list command with empty catalog."""
    catalog_dir = tmp_path / "empty_catalog"
    catalog_dir.mkdir()

    # Create empty config
    config = cfg.Config()
    config.to_yaml(catalog_dir / "faceberg.yml")

    runner = CliRunner()
    result = runner.invoke(main, [str(catalog_dir), "list"])

    # Command should succeed even with empty catalog
    assert result.exit_code == 0
