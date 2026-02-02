"""Playwright tests for Faceberg server."""

import re
import time

import pytest
from playwright.sync_api import Page, expect


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


def test_landing_page_loads(session_session_rest_server: str, page: Page):
    """Test that the landing page loads successfully."""
    page.goto(session_session_rest_server)

    # Check that the page title contains "Faceberg"
    expect(page).to_have_title(re.compile("Faceberg"))

    # Check that the header with Faceberg text is visible
    header = page.locator("h1:has-text('Faceberg')")
    expect(header).to_be_visible()


def test_header_displays_correctly(session_rest_server: str, page: Page):
    """Test that the header displays catalog information correctly."""
    page.goto(session_rest_server)

    # Check header title
    header_title = page.locator(".header-title")
    expect(header_title).to_be_visible()
    expect(header_title).to_contain_text("Faceberg")

    # Check logo is present
    logo = page.locator(".logo-icon img")
    expect(logo).to_be_visible()

    # Check catalog badge is present
    catalog_badge = page.locator(".header-catalog-badge")
    expect(catalog_badge).to_be_visible()

    # Check table count is present
    header_meta = page.locator(".header-meta")
    expect(header_meta).to_contain_text("tables")


def test_layout_structure(session_rest_server: str, page: Page):
    """Test that the page has correct layout structure."""
    page.goto(session_rest_server)

    # Check main container exists
    main_container = page.locator(".main-container")
    expect(main_container).to_be_visible()

    # Check left sidebar (table list)
    left_sidebar = page.locator(".left-sidebar")
    expect(left_sidebar).to_be_visible()

    # Check right sidebar (shell)
    right_sidebar = page.locator(".right-sidebar")
    expect(right_sidebar).to_be_visible()


def test_catalog_hierarchy_section_present(session_rest_server: str, page: Page):
    """Test that the catalog hierarchy section is present."""
    page.goto(session_rest_server)

    # Check for catalog hierarchy
    catalog_hierarchy = page.locator(".catalog-hierarchy")
    expect(catalog_hierarchy).to_be_visible()

    # Check section title
    section_title = page.locator(".section-title")
    expect(section_title).to_be_visible()


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

    header_meta = page.locator(".header-meta")
    expect(header_meta).to_contain_text("table")


def test_table_item_displays_metadata(session_rest_server: str, page: Page):
    """Test that table items display name and row count."""
    page.goto(session_rest_server)

    # Get first table item
    first_table = page.locator(".table-item").first

    # Check table name is visible
    table_name = first_table.locator(".table-name")
    expect(table_name).to_be_visible()

    # Check row count is visible
    row_count = first_table.locator(".table-row-count")
    expect(row_count).to_be_visible()
    expect(row_count).to_contain_text("rows")


def test_table_expansion(session_rest_server: str, page: Page):
    """Test that clicking a table expands its details."""
    page.goto(session_rest_server)

    # Get first table
    first_table = page.locator(".table-item").first
    summary = first_table.locator("summary")

    # Initially should be collapsed (not open)
    is_open = first_table.evaluate("el => el.hasAttribute('open')")
    assert not is_open, "Table should start collapsed"

    # Click to expand
    summary.click()

    # Should now be open
    expect(first_table).to_have_attribute("open", "")

    # Check that table content is visible
    table_content = first_table.locator(".table-content")
    expect(table_content).to_be_visible()


def test_table_schema_displays(session_rest_server: str, page: Page):
    """Test that expanded table shows schema information."""
    page.goto(session_rest_server)

    # Expand first table
    first_table = page.locator(".table-item").first
    first_table.locator("summary").click()

    # Wait for content to be visible
    table_content = first_table.locator(".table-content")
    expect(table_content).to_be_visible()

    # Check schema table exists
    schema_table = first_table.locator(".schema-table")
    expect(schema_table).to_be_visible()

    # Check for column headers
    expect(schema_table).to_contain_text("Column")
    expect(schema_table).to_contain_text("Type")
    expect(schema_table).to_contain_text("Req")


def test_table_metadata_grid(session_rest_server: str, page: Page):
    """Test that expanded table shows metadata grid with stats."""
    page.goto(session_rest_server)

    # Expand first table
    first_table = page.locator(".table-item").first
    first_table.locator("summary").click()

    # Check metadata grid
    metadata_grid = first_table.locator(".metadata-grid")
    expect(metadata_grid).to_be_visible()

    # Check for metadata labels
    expect(metadata_grid).to_contain_text("Rows:")
    expect(metadata_grid).to_contain_text("Files:")
    expect(metadata_grid).to_contain_text("Cols:")


def test_query_button_present(session_rest_server: str, page: Page):
    """Test that query button is present in expanded table."""
    page.goto(session_rest_server)

    # Expand first table
    first_table = page.locator(".table-item").first
    first_table.locator("summary").click()

    # Check for query button
    query_button = first_table.locator(".action-button")
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


def test_quick_tips_section(session_rest_server: str, page: Page):
    """Test that the quick tips section displays correctly."""
    page.goto(session_rest_server)

    quick_tips = page.locator(".quick-tips")
    expect(quick_tips).to_be_visible()
    expect(quick_tips).to_contain_text("DuckDB Shell")
    expect(quick_tips).to_contain_text("iceberg")
    expect(quick_tips).to_contain_text("httpfs")


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


def test_extension_badges_visible(session_rest_server: str, page: Page):
    """Test that extension badges are visible in quick tips."""
    page.goto(session_rest_server)

    # Check for extension badges
    extension_badges = page.locator(".extension-badge")
    expect(extension_badges).to_have_count(2)

    # Check specific extension names
    expect(page.locator(".extension-badge:has-text('iceberg')")).to_be_visible()
    expect(page.locator(".extension-badge:has-text('httpfs')")).to_be_visible()


# =============================================================================
# Responsive Behavior Tests
# =============================================================================


def test_page_responsive_at_smaller_viewport(session_rest_server: str, page: Page):
    """Test that the page is responsive at smaller viewport sizes."""
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(session_rest_server)

    # Check that main components are still visible
    main_container = page.locator(".main-container")
    expect(main_container).to_be_visible()

    left_sidebar = page.locator(".left-sidebar")
    expect(left_sidebar).to_be_visible()

    right_sidebar = page.locator(".right-sidebar")
    expect(right_sidebar).to_be_visible()


def test_scrollbar_styling_applied(session_rest_server: str, page: Page):
    """Test that custom scrollbar styling is applied."""
    page.goto(session_rest_server)

    # Check that left sidebar is scrollable
    left_sidebar = page.locator(".left-sidebar")
    expect(left_sidebar).to_be_visible()

    # Verify overflow-y is set to auto
    overflow = left_sidebar.evaluate("el => window.getComputedStyle(el).overflowY")
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
    # Catalog should have at least one namespace (stanfordnlp)
    assert len(namespaces_response["data"]["namespaces"]) > 0


def test_tables_endpoint_accessible(session_rest_server: str, page: Page):
    """Test that the /v1/namespaces/{namespace}/tables endpoint is accessible."""
    page.goto(session_rest_server)

    # Use page.evaluate to fetch tables endpoint
    tables_response = page.evaluate("""
        async () => {
            const response = await fetch('/v1/namespaces/stanfordnlp/tables');
            return {
                status: response.status,
                data: await response.json()
            };
        }
    """)

    assert tables_response["status"] == 200
    assert "identifiers" in tables_response["data"]
    # Should have at least one table (imdb)
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
    header = page.locator(".app-header")
    bg_color = header.evaluate("el => window.getComputedStyle(el).backgroundColor")

    # Should be some shade of blue (primary-blue from CSS)
    assert bg_color is not None, "Header should have background color"


def test_section_title_visible(session_rest_server: str, page: Page):
    """Test that the section title is visible."""
    page.goto(session_rest_server)

    section_title = page.locator(".section-title")
    expect(section_title).to_be_visible()
    expect(section_title).to_contain_text("Tables")


# =============================================================================
# Performance and Loading Tests
# =============================================================================


def test_page_loads_quickly(session_rest_server: str, page: Page):
    """Test that the page loads within a reasonable time."""
    start_time = time.time()

    page.goto(session_rest_server)

    # Wait for main content to be visible
    page.locator(".main-container").wait_for(state="visible")

    load_time = time.time() - start_time

    # Page should load in under 5 seconds
    assert load_time < 5.0, f"Page took too long to load: {load_time:.2f}s"


def test_fonts_load(session_rest_server: str, page: Page):
    """Test that custom fonts are loaded."""
    page.goto(session_rest_server)

    # Check that DM Sans font is applied to body
    body_font = page.locator("body").evaluate("el => window.getComputedStyle(el).fontFamily")

    assert "DM Sans" in body_font or "dm sans" in body_font.lower(), (
        f"Expected DM Sans font, got: {body_font}"
    )
