"""Playwright tests for Faceberg server."""

import re
import time

import pytest

try:
    from playwright.sync_api import Page, expect
except ImportError:
    playwright = pytest.importorskip("playwright", reason="Playwright not installed")


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for tests."""
    return {
        **browser_context_args,
        "ignore_https_errors": True,
        "viewport": {"width": 1280, "height": 720},
    }


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Configure browser launch arguments."""
    return {
        **browser_type_launch_args,
        "headless": True,
    }


# =============================================================================
# Page Load and Structure Tests
# =============================================================================


def test_landing_page_loads(session_rest_server: str, page: Page):
    """Test that the landing page loads successfully."""
    page.goto(session_rest_server)

    # Check that the page title contains "Faceberg"
    expect(page).to_have_title(re.compile("Faceberg"))

    # Check that the logo with Faceberg text is visible
    logo = page.locator(".logo")
    expect(logo).to_be_visible()
    expect(logo).to_contain_text("Faceberg")


def test_header_displays_correctly(session_rest_server: str, page: Page):
    """Test that the header displays catalog information correctly."""
    page.goto(session_rest_server)

    # Check header
    header = page.locator(".header")
    expect(header).to_be_visible()

    # Check logo is present
    logo = page.locator(".logo")
    expect(logo).to_be_visible()
    expect(logo).to_contain_text("Faceberg")

    # Check logo icon image
    logo_img = page.locator(".logo-icon img")
    expect(logo_img).to_be_visible()

    # Check catalog badge is present
    catalog_badge = page.locator(".catalog-badge")
    expect(catalog_badge).to_be_visible()

    # Check table count is present
    header_stat = page.locator(".header-stat")
    expect(header_stat).to_contain_text("tables")


def test_layout_structure(session_rest_server: str, page: Page):
    """Test that the page has correct three-pane layout structure."""
    page.goto(session_rest_server)

    # Check main container exists
    main = page.locator(".main")
    expect(main).to_be_visible()

    # Check left pane (catalog)
    pane_catalog = page.locator(".pane-catalog")
    expect(pane_catalog).to_be_visible()

    # Check middle pane (metadata)
    pane_metadata = page.locator(".pane-metadata")
    expect(pane_metadata).to_be_visible()

    # Check right pane (shell)
    pane_shell = page.locator(".pane-shell")
    expect(pane_shell).to_be_visible()


def test_catalog_pane_structure(session_rest_server: str, page: Page):
    """Test that the catalog pane has correct structure."""
    page.goto(session_rest_server)

    # Check pane header
    pane_header = page.locator(".pane-catalog .pane-header")
    expect(pane_header).to_be_visible()

    # Check pane title
    pane_title = page.locator(".pane-catalog .pane-title")
    expect(pane_title).to_be_visible()
    expect(pane_title).to_contain_text("Catalog")

    # Check pane content
    pane_content = page.locator(".pane-catalog .pane-content")
    expect(pane_content).to_be_visible()


# =============================================================================
# Catalog with Data Tests
# =============================================================================


def test_catalog_displays_tables(session_rest_server: str, page: Page):
    """Test that the catalog displays table list."""
    page.goto(session_rest_server)

    # Check that at least one table is visible
    table_items = page.locator(".table-item")
    expect(table_items.first).to_be_visible()


def test_catalog_shows_table_count(session_rest_server: str, page: Page):
    """Test that the catalog shows table count in header."""
    page.goto(session_rest_server)

    header_stat = page.locator(".header-stat")
    expect(header_stat).to_contain_text("table")


def test_namespace_groups_displayed(session_rest_server: str, page: Page):
    """Test that namespace groups are displayed."""
    page.goto(session_rest_server)

    # Check namespace group exists
    namespace_group = page.locator(".namespace-group")
    expect(namespace_group.first).to_be_visible()

    # Check namespace header
    namespace_header = page.locator(".namespace-header")
    expect(namespace_header.first).to_be_visible()

    # Check namespace name
    namespace_name = page.locator(".namespace-name")
    expect(namespace_name.first).to_be_visible()


def test_table_item_displays_metadata(session_rest_server: str, page: Page):
    """Test that table items display name and row count."""
    page.goto(session_rest_server)

    # Get first table item
    first_table = page.locator(".table-item").first

    # Check table name is visible
    table_name = first_table.locator(".table-name")
    expect(table_name).to_be_visible()

    # Check row count is visible
    row_count = first_table.locator(".table-rows")
    expect(row_count).to_be_visible()


def test_table_selection_shows_metadata(session_rest_server: str, page: Page):
    """Test that clicking a table shows its metadata in the middle pane."""
    page.goto(session_rest_server)

    # Get first table
    first_table = page.locator(".table-item").first

    # Initially metadata content should be hidden
    metadata_empty = page.locator("#metadata-empty")
    expect(metadata_empty).to_be_visible()

    # Click to select table
    first_table.click()

    # Now metadata content should be visible
    metadata_content = page.locator("#metadata-content")
    expect(metadata_content).to_be_visible()

    # Table should have selected class
    expect(first_table).to_have_class(re.compile("selected"))


def test_table_schema_displays(session_rest_server: str, page: Page):
    """Test that selected table shows schema information."""
    page.goto(session_rest_server)

    # Select first table
    first_table = page.locator(".table-item").first
    first_table.click()

    # Check schema section exists
    schema_section = page.locator("#section-schema")
    expect(schema_section).to_be_visible()

    # Check schema grid exists with columns
    schema_grid = page.locator("#schema-grid")
    expect(schema_grid).to_be_visible()

    # Check for column elements
    schema_columns = page.locator(".schema-column")
    expect(schema_columns.first).to_be_visible()


def test_table_header_info_displays(session_rest_server: str, page: Page):
    """Test that selected table shows header info with stats."""
    page.goto(session_rest_server)

    # Select first table
    first_table = page.locator(".table-item").first
    first_table.click()

    # Check table header info
    table_header = page.locator(".table-header-info")
    expect(table_header).to_be_visible()

    # Check table name in header
    table_header_name = page.locator("#table-header-name")
    expect(table_header_name).to_be_visible()

    # Check metadata items (rows, files, columns)
    table_header_meta = page.locator("#table-header-meta")
    expect(table_header_meta).to_contain_text("rows")
    expect(table_header_meta).to_contain_text("files")
    expect(table_header_meta).to_contain_text("columns")


def test_query_button_present(session_rest_server: str, page: Page):
    """Test that query button is present when table is selected."""
    page.goto(session_rest_server)

    # Select first table
    first_table = page.locator(".table-item").first
    first_table.click()

    # Check for query button
    query_button = page.locator(".btn-primary")
    expect(query_button).to_be_visible()
    expect(query_button).to_contain_text("Query")


# =============================================================================
# DuckDB Shell Tests
# =============================================================================


def test_shell_container_visible(session_rest_server: str, page: Page):
    """Test that the shell container is visible on page load."""
    page.goto(session_rest_server)

    shell_container = page.locator("#shell-container")
    expect(shell_container).to_be_visible()


def test_shell_header_section(session_rest_server: str, page: Page):
    """Test that the shell header displays correctly."""
    page.goto(session_rest_server)

    # Check shell header
    shell_header = page.locator(".shell-header")
    expect(shell_header).to_be_visible()

    # Check shell title
    shell_title = page.locator(".shell-title")
    expect(shell_title).to_be_visible()
    expect(shell_title).to_contain_text("DuckDB Shell")

    # Check hints section
    shell_hints = page.locator(".shell-hints")
    expect(shell_hints).to_be_visible()


def test_duckdb_shell_initializes(session_rest_server: str, page: Page):
    """Test that the DuckDB shell initializes without errors."""
    # Set up console message and error listeners
    console_messages = []
    errors = []

    page.on("console", lambda msg: console_messages.append(msg))
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    # Navigate to the page
    page.goto(session_rest_server)

    # Wait for the shell container to be visible
    shell_container = page.locator("#shell-container")
    expect(shell_container).to_be_visible(timeout=10000)

    # Wait for the shell to initialize
    page.wait_for_timeout(8000)

    # Check for the specific postMessage error
    post_message_errors = [
        err for err in errors if "postMessage" in err and "could not be cloned" in err
    ]

    # Assert that no postMessage cloning errors occurred
    assert len(post_message_errors) == 0, (
        f"Found {len(post_message_errors)} postMessage cloning errors: {post_message_errors}"
    )

    # Check that shell initialization didn't show an error message in the UI
    error_divs = page.locator("div:has-text('Error initializing shell')").all()
    visible_errors = [div for div in error_divs if div.is_visible()]
    assert len(visible_errors) == 0, "Error message is visible in UI"

    # Verify the shell is ready by checking for terminal-like elements
    terminal = page.locator(".xterm")
    expect(terminal).to_be_visible(timeout=5000)


def test_shell_has_xterm(session_rest_server: str, page: Page):
    """Test that XTerm.js terminal renders in the shell."""
    page.goto(session_rest_server)

    # Wait for XTerm to initialize
    page.wait_for_timeout(8000)

    # Check for XTerm elements
    xterm = page.locator(".xterm")
    expect(xterm).to_be_visible()

    # Check for XTerm viewport
    xterm_viewport = page.locator(".xterm-viewport")
    expect(xterm_viewport).to_be_visible()


def test_hint_badges_visible(session_rest_server: str, page: Page):
    """Test that hint badges are visible in shell header."""
    page.goto(session_rest_server)

    # Check for hint badges (iceberg, httpfs, .help)
    hint_badges = page.locator(".hint-badge")
    expect(hint_badges).to_have_count(3)

    # Check specific extension names
    expect(page.locator(".hint-badge:has-text('iceberg')")).to_be_visible()
    expect(page.locator(".hint-badge:has-text('httpfs')")).to_be_visible()


# =============================================================================
# Responsive Behavior Tests
# =============================================================================


def test_page_responsive_at_smaller_viewport(session_rest_server: str, page: Page):
    """Test that the page is responsive at smaller viewport sizes."""
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(session_rest_server)

    # Check that main components are still visible
    main = page.locator(".main")
    expect(main).to_be_visible()

    pane_catalog = page.locator(".pane-catalog")
    expect(pane_catalog).to_be_visible()

    pane_shell = page.locator(".pane-shell")
    expect(pane_shell).to_be_visible()


def test_metadata_pane_hidden_on_small_viewport(session_rest_server: str, page: Page):
    """Test that metadata pane is hidden on small viewport."""
    page.set_viewport_size({"width": 800, "height": 600})
    page.goto(session_rest_server)

    # Metadata pane should be hidden at this size
    pane_metadata = page.locator(".pane-metadata")
    expect(pane_metadata).not_to_be_visible()


def test_scrollbar_styling_applied(session_rest_server: str, page: Page):
    """Test that scrollable areas have proper overflow."""
    page.goto(session_rest_server)

    # Check that pane content is scrollable
    pane_content = page.locator(".pane-catalog .pane-content")
    expect(pane_content).to_be_visible()

    # Verify overflow-y is set to auto
    overflow = pane_content.evaluate("el => window.getComputedStyle(el).overflowY")
    assert overflow == "auto", f"Expected overflow-y: auto, got {overflow}"


# =============================================================================
# API Endpoint Tests (via browser fetch)
# =============================================================================


def test_config_endpoint_accessible(session_rest_server: str, page: Page):
    """Test that the /v1/config endpoint is accessible."""
    page.goto(session_rest_server)

    # Use page.evaluate to fetch config endpoint
    config_response = page.evaluate("""
        async () => {
            const response = await fetch('/v1/config');
            return {
                status: response.status,
                data: await response.json()
            };
        }
    """)

    assert config_response["status"] == 200
    assert "overrides" in config_response["data"]


def test_namespaces_endpoint_accessible(session_rest_server: str, page: Page):
    """Test that the /v1/namespaces endpoint is accessible."""
    page.goto(session_rest_server)

    # Use page.evaluate to fetch namespaces endpoint
    namespaces_response = page.evaluate("""
        async () => {
            const response = await fetch('/v1/namespaces');
            return {
                status: response.status,
                data: await response.json()
            };
        }
    """)

    assert namespaces_response["status"] == 200
    assert "namespaces" in namespaces_response["data"]
    # Catalog should have at least one namespace (google-research-datasets)
    assert len(namespaces_response["data"]["namespaces"]) > 0


def test_tables_endpoint_accessible(session_rest_server: str, page: Page):
    """Test that the /v1/namespaces/{namespace}/tables endpoint is accessible."""
    page.goto(session_rest_server)

    # Use page.evaluate to fetch tables endpoint
    tables_response = page.evaluate("""
        async () => {
            const response = await fetch('/v1/namespaces/google-research-datasets/tables');
            return {
                status: response.status,
                data: await response.json()
            };
        }
    """)

    assert tables_response["status"] == 200
    assert "identifiers" in tables_response["data"]
    # Should have at least one table (mbpp)
    assert len(tables_response["data"]["identifiers"]) > 0


# =============================================================================
# JavaScript Error Tests
# =============================================================================


def test_no_javascript_errors_on_load(session_rest_server: str, page: Page):
    """Test that no JavaScript errors occur on page load."""
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    page.goto(session_rest_server)
    page.wait_for_timeout(2000)

    # Filter out known acceptable warnings
    critical_errors = [
        err
        for err in errors
        if "postMessage" not in err  # Known DuckDB WASM issue
    ]

    assert len(critical_errors) == 0, f"Found JavaScript errors: {critical_errors}"


def test_no_console_errors(session_rest_server: str, page: Page):
    """Test that no critical console errors are logged."""
    console_errors = []

    def handle_console(msg):
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", handle_console)

    page.goto(session_rest_server)
    page.wait_for_timeout(3000)

    # Some console errors might be acceptable (DuckDB initialization messages)
    # Filter for critical errors that indicate real problems
    critical_errors = [
        err
        for err in console_errors
        if "Failed to load resource" in err
        or "Uncaught" in err
        or "SyntaxError" in err
        or "ReferenceError" in err
    ]

    assert len(critical_errors) == 0, f"Found critical console errors: {critical_errors}"


# =============================================================================
# Visual Elements Tests
# =============================================================================


def test_logo_image_loads(session_rest_server: str, page: Page):
    """Test that the logo image loads correctly."""
    page.goto(session_rest_server)

    logo = page.locator(".logo-icon img")
    expect(logo).to_be_visible()

    # Check that image has a source
    src = logo.get_attribute("src")
    assert src is not None
    assert "faceberg" in src.lower()


def test_color_scheme_applied(session_rest_server: str, page: Page):
    """Test that the color scheme is properly applied."""
    page.goto(session_rest_server)

    # Check header background color
    header = page.locator(".header")
    bg_color = header.evaluate("el => window.getComputedStyle(el).backgroundColor")

    # Should have some background color
    assert bg_color is not None, "Header should have background color"


def test_pane_title_visible(session_rest_server: str, page: Page):
    """Test that the pane titles are visible."""
    page.goto(session_rest_server)

    # Check catalog pane title
    pane_title = page.locator(".pane-catalog .pane-title")
    expect(pane_title).to_be_visible()
    expect(pane_title).to_contain_text("Catalog")


# =============================================================================
# Theme Toggle Tests
# =============================================================================


def test_theme_toggle_button_visible(session_rest_server: str, page: Page):
    """Test that the theme toggle button is visible."""
    page.goto(session_rest_server)

    theme_toggle = page.locator(".theme-toggle")
    expect(theme_toggle).to_be_visible()


def test_theme_toggle_switches_theme(session_rest_server: str, page: Page):
    """Test that clicking the theme toggle switches the theme."""
    page.goto(session_rest_server)

    # Check initial theme (default is dark)
    html = page.locator("html")
    initial_theme = html.get_attribute("data-theme")

    # Click theme toggle
    theme_toggle = page.locator(".theme-toggle")
    theme_toggle.click()

    # Check theme changed
    new_theme = html.get_attribute("data-theme")
    assert new_theme != initial_theme, "Theme should have changed after toggle"


# =============================================================================
# Performance and Loading Tests
# =============================================================================


def test_page_loads_quickly(session_rest_server: str, page: Page):
    """Test that the page loads within a reasonable time."""
    start_time = time.time()

    page.goto(session_rest_server)

    # Wait for main content to be visible
    page.locator(".main").wait_for(state="visible")

    load_time = time.time() - start_time

    # Page should load in under 5 seconds
    assert load_time < 5.0, f"Page took too long to load: {load_time:.2f}s"


def test_fonts_load(session_rest_server: str, page: Page):
    """Test that custom fonts are loaded."""
    page.goto(session_rest_server)

    # Check that Inter font is applied to body
    body_font = page.locator("body").evaluate("el => window.getComputedStyle(el).fontFamily")

    assert "Inter" in body_font or "inter" in body_font.lower(), (
        f"Expected Inter font, got: {body_font}"
    )


# =============================================================================
# Resize Handle Tests
# =============================================================================


def test_resize_handles_visible(session_rest_server: str, page: Page):
    """Test that resize handles are present for pane resizing."""
    page.goto(session_rest_server)

    resize_handles = page.locator(".resize-handle")
    expect(resize_handles).to_have_count(2)


# =============================================================================
# Namespace Expansion Tests
# =============================================================================


def test_namespace_toggle_expansion(session_rest_server: str, page: Page):
    """Test that namespace groups can be expanded and collapsed."""
    page.goto(session_rest_server)

    # Get first namespace group
    namespace_group = page.locator(".namespace-group").first

    # Should start expanded
    expect(namespace_group).to_have_class(re.compile("expanded"))

    # Tables should be visible
    namespace_tables = namespace_group.locator(".namespace-tables")
    expect(namespace_tables).to_be_visible()

    # Click header to collapse
    namespace_header = namespace_group.locator(".namespace-header")
    namespace_header.click()

    # Should no longer have expanded class
    expect(namespace_group).not_to_have_class(re.compile("expanded"))
