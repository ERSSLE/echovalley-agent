import gc
import os
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import Optional,Dict,List,Any,Callable,Coroutine,ParamSpec,TypeVar
from playwright.async_api import Page,ElementHandle,FrameLocator
from functools import cached_property,wraps
import time
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class HashedDomElement:
	"""
	Hash of the dom element to be used as a unique identifier
	"""

	branch_path_hash: str
	attributes_hash: str
	xpath_hash: str
	# text_hash: str


class Coordinates(BaseModel):
	x: int
	y: int


class CoordinateSet(BaseModel):
	top_left: Coordinates
	top_right: Coordinates
	bottom_left: Coordinates
	bottom_right: Coordinates
	center: Coordinates
	width: int
	height: int


class ViewportInfo(BaseModel):
	scroll_x: int
	scroll_y: int
	width: int
	height: int


@dataclass
class DOMHistoryElement:
	tag_name: str
	xpath: str
	highlight_index: Optional[int]
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False
	css_selector: Optional[str] = None
	page_coordinates: Optional[CoordinateSet] = None
	viewport_coordinates: Optional[CoordinateSet] = None
	viewport_info: Optional[ViewportInfo] = None

	def to_dict(self) -> dict:
		page_coordinates = self.page_coordinates.model_dump() if self.page_coordinates else None
		viewport_coordinates = self.viewport_coordinates.model_dump() if self.viewport_coordinates else None
		viewport_info = self.viewport_info.model_dump() if self.viewport_info else None

		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
			'css_selector': self.css_selector,
			'page_coordinates': page_coordinates,
			'viewport_coordinates': viewport_coordinates,
			'viewport_info': viewport_info,
		}

# Define generic type variables for return type and parameters
R = TypeVar('R')
P = ParamSpec('P')


def time_execution_sync(additional_text: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:
	def decorator(func: Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = func(*args, **kwargs)
			execution_time = time.time() - start_time
			logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
			return result

		return wrapper

	return decorator


def time_execution_async(
	additional_text: str = '',
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
	def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
		@wraps(func)
		async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = await func(*args, **kwargs)
			execution_time = time.time() - start_time
			logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
			return result

		return wrapper

	return decorator


def singleton(cls):
	instance = [None]

	def wrapper(*args, **kwargs):
		if instance[0] is None:
			instance[0] = cls(*args, **kwargs)
		return instance[0]

	return wrapper


@dataclass(frozen=False)
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
	text: str
	type: str = 'TEXT_NODE'

	def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			# stop if the element has a highlight index (will be handled separately)
			if current.highlight_index is not None:
				return True

			current = current.parent
		return False

	def is_parent_in_viewport(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_in_viewport

	def is_parent_top_element(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_top_element


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
	"""
	xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
	To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
	"""

	tag_name: str
	xpath: str
	attributes: Dict[str, str]
	children: List[DOMBaseNode]
	is_interactive: bool = False
	is_top_element: bool = False
	is_in_viewport: bool = False
	shadow_root: bool = False
	highlight_index: Optional[int] = None
	viewport_coordinates: Optional[CoordinateSet] = None
	page_coordinates: Optional[CoordinateSet] = None
	viewport_info: Optional[ViewportInfo] = None

	def __repr__(self) -> str:
		tag_str = f'<{self.tag_name}'

		# Add attributes
		for key, value in self.attributes.items():
			tag_str += f' {key}="{value}"'
		tag_str += '>'

		# Add extra info
		extras = []
		if self.is_interactive:
			extras.append('interactive')
		if self.is_top_element:
			extras.append('top')
		if self.shadow_root:
			extras.append('shadow-root')
		if self.highlight_index is not None:
			extras.append(f'highlight:{self.highlight_index}')
		if self.is_in_viewport:
			extras.append('in-viewport')

		if extras:
			tag_str += f' [{", ".join(extras)}]'

		return tag_str

	@cached_property
	def hash(self) -> HashedDomElement:
		from browser_use.dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)

		return HistoryTreeProcessor._hash_dom_element(self)

	def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
		text_parts = []

		def collect_text(node: DOMBaseNode, current_depth: int) -> None:
			if max_depth != -1 and current_depth > max_depth:
				return

			# Skip this branch if we hit a highlighted element (except for the current node)
			if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child, current_depth + 1)

		collect_text(self, 0)
		return '\n'.join(text_parts).strip()

	@time_execution_sync('--clickable_elements_to_string')
	def clickable_elements_to_string(self, include_attributes: list[str] = []) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		def process_node(node: DOMBaseNode, depth: int) -> None:
			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					attributes_str = ''
					text = node.get_all_text_till_next_clickable_element()
					if include_attributes:
						attributes = list(
							set(
								[
									str(value)
									for key, value in node.attributes.items()
									if key in include_attributes and value != node.tag_name
								]
							)
						)
						if text in attributes:
							attributes.remove(text)
						attributes_str = ';'.join(attributes)
					line = f'[{node.highlight_index}]<{node.tag_name} '
					if attributes_str:
						line += f'{attributes_str}'
					if text:
						if attributes_str:
							line += f'>{text}'
						else:
							line += f'{text}'
					line += '/>'
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, depth + 1)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if not node.has_parent_with_highlight_index() and node.is_visible:  # and node.is_parent_top_element()
					formatted_text.append(f'{node.text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)

	def get_file_upload_element(self, check_siblings: bool = True) -> Optional['DOMElementNode']:
		# Check if current element is a file input
		if self.tag_name == 'input' and self.attributes.get('type') == 'file':
			return self

		# Check children
		for child in self.children:
			if isinstance(child, DOMElementNode):
				result = child.get_file_upload_element(check_siblings=False)
				if result:
					return result

		# Check siblings only for the initial call
		if check_siblings and self.parent:
			for sibling in self.parent.children:
				if sibling is not self and isinstance(sibling, DOMElementNode):
					result = sibling.get_file_upload_element(check_siblings=False)
					if result:
						return result
		return None


@dataclass
class DOMState:
	element_tree: DOMElementNode
	selector_map: Dict[int, DOMElementNode]

#==========================================

@dataclass
class ViewportInfo:
	width: int
	height: int

class DomService:
	def __init__(self, page: 'Page'):
		self.page = page
		self.xpath_cache = {}

		self.js_code = resources.read_text('dom', 'buildDomTree.js')

	# region - Clickable elements
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		highlight_elements: bool = True,
		focus_element: int = -1,
		viewport_expansion: int = 0,
	) -> DOMState:
		element_tree, selector_map = await self._build_dom_tree(highlight_elements, focus_element, viewport_expansion)
		return DOMState(element_tree=element_tree, selector_map=selector_map)

	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, Dict[int, DOMElementNode]]:
		if await self.page.evaluate('1+1') != 2:
			raise ValueError('The page cannot evaluate javascript code properly')

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.
		debug_mode = logger.getEffectiveLevel() == logging.DEBUG
		args = {
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			eval_page = await self.page.evaluate(self.js_code, args)
		except Exception as e:
			logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Only log performance metrics in debug mode
		if debug_mode and 'perfMetrics' in eval_page:
			logger.debug('DOM Tree Building Performance Metrics:\n%s', json.dumps(eval_page['perfMetrics'], indent=2))

		return await self._construct_dom_tree(eval_page)

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, Dict[int, DOMElementNode]]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']

		selector_map = {}
		node_map = {}

		for id, node_data in js_node_map.items():
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node

			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):
				for child_id in children_ids:
					if child_id not in node_map:
						continue

					child_node = node_map[child_id]

					child_node.parent = node
					node.children.append(child_node)

		html_to_dict = node_map[str(js_root_id)]

		del node_map
		del js_node_map
		del js_root_id

		gc.collect()

		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(
		self,
		node_data: dict,
	) -> tuple[Optional[DOMBaseNode], list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []

		# Process coordinates if they exist for element nodes

		viewport_info = None

		if 'viewport' in node_data:
			viewport_info = ViewportInfo(
				width=node_data['viewport']['width'],
				height=node_data['viewport']['height'],
			)

		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
		)

		children_ids = node_data.get('children', [])

		return element_node, children_ids
	
async def remove_highlights(page: Page):
    """
    Removes all highlight overlays and labels created by the highlightElement function.
    Handles cases where the page might be closed or inaccessible.
    """
    try:
        await page.evaluate(
            """
            try {
                // Remove the highlight container and all its contents
                const container = document.getElementById('playwright-highlight-container');
                if (container) {
                    container.remove();
                }

                // Remove highlight attributes from elements
                const highlightedElements = document.querySelectorAll('[browser-user-highlight-id^="playwright-highlight-"]');
                highlightedElements.forEach(el => {
                    el.removeAttribute('browser-user-highlight-id');
                });
            } catch (e) {
                console.error('Failed to remove highlights:', e);
            }
            """
        )
    except Exception as e:
        logger.debug(f'Failed to remove highlights (this is usually ok): {str(e)}')
        # Don't raise the error since this is not critical functionality
        pass
	
#==============================================================================
import re

def _convert_simple_xpath_to_css_selector(xpath: str) -> str:
    """Converts simple XPath expressions to CSS selectors."""
    if not xpath:
        return ''

    # Remove leading slash if present
    xpath = xpath.lstrip('/')

    # Split into parts
    parts = xpath.split('/')
    css_parts = []

    for part in parts:
        if not part:
            continue

        # Handle index notation [n]
        if '[' in part:
            base_part = part[: part.find('[')]
            index_part = part[part.find('[') :]

            # Handle multiple indices
            indices = [i.strip('[]') for i in index_part.split(']')[:-1]]

            for idx in indices:
                try:
                    # Handle numeric indices
                    if idx.isdigit():
                        index = int(idx) - 1
                        base_part += f':nth-of-type({index + 1})'
                    # Handle last() function
                    elif idx == 'last()':
                        base_part += ':last-of-type'
                    # Handle position() functions
                    elif 'position()' in idx:
                        if '>1' in idx:
                            base_part += ':nth-of-type(n+2)'
                except ValueError:
                    continue

            css_parts.append(base_part)
        else:
            css_parts.append(part)

    base_selector = ' > '.join(css_parts)
    return base_selector


def _enhanced_css_selector_for_element(element: DOMElementNode, include_dynamic_attributes: bool = True) -> str:
    """
    Creates a CSS selector for a DOM element, handling various edge cases and special characters.

    Args:
            element: The DOM element to create a selector for

    Returns:
            A valid CSS selector string
    """
    try:
        # Get base selector from XPath
        css_selector = _convert_simple_xpath_to_css_selector(element.xpath)

        # Handle class attributes
        if 'class' in element.attributes and element.attributes['class'] and include_dynamic_attributes:
            # Define a regex pattern for valid class names in CSS
            valid_class_name_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_-]*$')

            # Iterate through the class attribute values
            classes = element.attributes['class'].split()
            for class_name in classes:
                # Skip empty class names
                if not class_name.strip():
                    continue

                # Check if the class name is valid
                if valid_class_name_pattern.match(class_name):
                    # Append the valid class name to the CSS selector
                    css_selector += f'.{class_name}'
                else:
                    # Skip invalid class names
                    continue

        # Expanded set of safe attributes that are stable and useful for selection
        SAFE_ATTRIBUTES = {
            # Data attributes (if they're stable in your application)
            'id',
            # Standard HTML attributes
            'name',
            'type',
            'placeholder',
            # Accessibility attributes
            'aria-label',
            'aria-labelledby',
            'aria-describedby',
            'role',
            # Common form attributes
            'for',
            'autocomplete',
            'required',
            'readonly',
            # Media attributes
            'alt',
            'title',
            'src',
            # Custom stable attributes (add any application-specific ones)
            'href',
            'target',
        }

        if include_dynamic_attributes:
            dynamic_attributes = {
                'data-id',
                'data-qa',
                'data-cy',
                'data-testid',
            }
            SAFE_ATTRIBUTES.update(dynamic_attributes)

        # Handle other attributes
        for attribute, value in element.attributes.items():
            if attribute == 'class':
                continue

            # Skip invalid attribute names
            if not attribute.strip():
                continue

            if attribute not in SAFE_ATTRIBUTES:
                continue

            # Escape special characters in attribute names
            safe_attribute = attribute.replace(':', r'\:')

            # Handle different value cases
            if value == '':
                css_selector += f'[{safe_attribute}]'
            elif any(char in value for char in '"\'<>`\n\r\t'):
                # Use contains for values with special characters
                # Regex-substitute *any* whitespace with a single space, then strip.
                collapsed_value = re.sub(r'\s+', ' ', value).strip()
                # Escape embedded double-quotes.
                safe_value = collapsed_value.replace('"', '\\"')
                css_selector += f'[{safe_attribute}*="{safe_value}"]'
            else:
                css_selector += f'[{safe_attribute}="{value}"]'

        return css_selector

    except Exception:
        # Fallback to a more basic selector if something goes wrong
        tag_name = element.tag_name or '*'
        return f"{tag_name}[highlight_index='{element.highlight_index}']"


async def get_locate_element(page: Page, element: DOMElementNode) -> Optional[ElementHandle]:
    current_frame = page
    # Start with the target element and collect all parents
    parents: list[DOMElementNode] = []
    current = element
    while current.parent is not None:
        parent = current.parent
        parents.append(parent)
        current = parent

    # Reverse the parents list to process from top to bottom
    parents.reverse()

    # Process all iframe parents in sequence
    iframes = [item for item in parents if item.tag_name == 'iframe']
    for parent in iframes:
        css_selector = _enhanced_css_selector_for_element(
            parent,
            include_dynamic_attributes=True,
        )
        current_frame = current_frame.frame_locator(css_selector)

    css_selector = _enhanced_css_selector_for_element(
        element, include_dynamic_attributes=True
    )

    try:
        if isinstance(current_frame, FrameLocator):
            element_handle = await current_frame.locator(css_selector).element_handle()
            return element_handle
        else:
            # Try to scroll into view if hidden
            element_handle = await current_frame.query_selector(css_selector)
            if element_handle:
                await element_handle.scroll_into_view_if_needed()
                return element_handle
            return None
    except Exception as e:
        logger.error(f'Failed to locate element: {str(e)}')
        return None
	
class BrowserError(Exception):
	"""Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""



async def input_text_element_node(page: Page, element_node: DOMElementNode, text: str):
    """
    Input text into an element with proper error handling and state management.
    Handles different types of input fields and ensures proper element state before input.
    """
    try:
        # Highlight before typing
        # if element_node.highlight_index is not None:
        # 	await self._update_state(focus_element=element_node.highlight_index)

        element_handle = await get_locate_element(page,element_node)

        if element_handle is None:
            raise BrowserError(f'Element: {repr(element_node)} not found')

        # Ensure element is ready for input
        try:
            await element_handle.wait_for_element_state('stable', timeout=1000)
            await element_handle.scroll_into_view_if_needed(timeout=1000)
        except Exception:
            pass

        # Get element properties to determine input method
        is_contenteditable = await element_handle.get_property('isContentEditable')

        # Different handling for contenteditable vs input fields
        if await is_contenteditable.json_value():
            await element_handle.evaluate('el => el.textContent = ""')
            await element_handle.type(text, delay=5)
        else:
            await element_handle.fill(text)

    except Exception as e:
        logger.debug(f'Failed to input text into element: {repr(element_node)}. Error: {str(e)}')
        raise BrowserError(f'Failed to input text into index {element_node.highlight_index}')

async def _get_unique_filename(directory, filename):
    """Generate a unique filename by appending (1), (2), etc., if a file already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f'{base} ({counter}){ext}'
        counter += 1
    return new_filename

def is_url_allowed(url: str,allowed_domains: List[str] | None = None) -> bool:
    """Check if a URL is allowed based on the whitelist configuration."""
    if not allowed_domains:
        return True
    try:
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Remove port number if present
        if ':' in domain:
            domain = domain.split(':')[0]

        # Check if domain matches any allowed domain pattern
        return any(
            domain == allowed_domain.lower() or domain.endswith('.' + allowed_domain.lower())
            for allowed_domain in allowed_domains
        )
    except Exception as e:
        logger.error(f'Error checking URL allowlist: {str(e)}')
        return False

async def go_back(page):
    """Navigate back in history"""
    try:
        # 10 ms timeout
        await page.go_back(timeout=10, wait_until='domcontentloaded')
        # await self._wait_for_page_and_frames_load(timeout_overwrite=1.0)
    except Exception as e:
        # Continue even if its not fully loaded, because we wait later for the page to load
        logger.debug(f'During go_back: {e}')

async def _check_and_handle_navigation(page: Page) -> None:
    """Check if current page URL is allowed and handle if not."""
    if not is_url_allowed(page.url):
        logger.warning(f'Navigation to non-allowed URL detected: {page.url}')
        try:
            await go_back()
        except Exception as e:
            logger.error(f'Failed to go back after detecting non-allowed URL: {str(e)}')
        raise URLNotAllowedError(f'Navigation to non-allowed URL: {page.url}')

async def click_element_node(page:Page, element_node: DOMElementNode, save_downloads_path:str | None = None) -> Optional[str]:
    """
    Optimized method to click an element using xpath.
    """
    try:
        # Highlight before clicking
        # if element_node.highlight_index is not None:
        # 	await self._update_state(focus_element=element_node.highlight_index)

        element_handle = await get_locate_element(page,element_node)
        print(element_handle)
        if element_handle is None:
            raise Exception(f'Element: {repr(element_node)} not found')

        async def perform_click(click_func):
            """Performs the actual click, handling both download
            and navigation scenarios."""
            if save_downloads_path:
                try:
                    # Try short-timeout expect_download to detect a file download has been been triggered
                    async with page.expect_download(timeout=5000) as download_info:
                        await click_func()
                    download = await download_info.value
                    # Determine file path
                    suggested_filename = download.suggested_filename
                    unique_filename = await _get_unique_filename(save_downloads_path, suggested_filename)
                    download_path = os.path.join(save_downloads_path, unique_filename)
                    await download.save_as(download_path)
                    logger.debug(f'Download triggered. Saved file to: {download_path}')
                    return download_path
                except TimeoutError:
                    # If no download is triggered, treat as normal click
                    logger.debug('No download triggered within timeout. Checking navigation...')
                    await page.wait_for_load_state()
                    await _check_and_handle_navigation(page)
            else:
                # Standard click logic if no download is expected
                await click_func()
                await page.wait_for_load_state()
                await _check_and_handle_navigation(page)

        try:
            return await perform_click(lambda: element_handle.click(timeout=1500))
        except URLNotAllowedError as e:
            raise e
        except Exception:
            try:
                return await perform_click(lambda: page.evaluate('(el) => el.click()', element_handle))
            except URLNotAllowedError as e:
                raise e
            except Exception as e:
                raise Exception(f'Failed to click element: {str(e)}')

    except URLNotAllowedError as e:
        raise e
    except Exception as e:
        raise Exception(f'Failed to click element: {repr(element_node)}. Error: {str(e)}')

