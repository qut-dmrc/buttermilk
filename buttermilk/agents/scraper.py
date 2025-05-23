"""Provides web scraping capabilities using both basic HTTP requests and Selenium.

This module defines two main classes:
- `WebScraperRequests`: A base class for scraping using the `requests` library.
- `WebScraperSelenium`: Extends `WebScraperRequests` to use Selenium WebDriver
  for dynamic content scraping that may require JavaScript execution.

The classes are designed to be modular, allowing subclasses to override methods
for specific extraction logic and handling of pagination or recursion.
They incorporate retry logic for network requests and can be configured with
request intervals.

Note:
    This file appears to use a global `gc.logger` and `gc.upload_text`/`gc.upload_binary`
    which are not defined within this module. It's assumed `gc` is a globally available
    object providing these functionalities (e.g., a Google Cloud client or similar utility).
    The `bm` object for saving is used from `buttermilk.bm` import.
    Also, `selenium`, `By` from `selenium.webdriver.common.by`, and
    `NoSuchElementException` from `selenium.common.exceptions` are used but not
    imported directly in this file, which would cause runtime errors if not available
    in the execution environment where these classes are used.
    The `new_webdriver()` function is called by `WebScraperSelenium` but not defined
    here, implying it must be provided from an external source.

"""

import logging  # Standard logging
import random
import time
from typing import Any  # For type hinting

import requests  # For HTTP requests

# Attempt to import Selenium types for type hinting if available
try:
    from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement
except ImportError:
    WebDriver = Any  # type: ignore
    WebElement = Any  # type: ignore
    By = Any  # type: ignore
    NoSuchElementException = Exception  # type: ignore
    WebDriverException = Exception  # type: ignore
    StaleElementReferenceException = Exception  # type: ignore


import urllib3  # Underlying HTTP library used by requests
from pydantic import PrivateAttr  # Pydantic for private attributes
from tenacity import (  # Retry library components
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from buttermilk import buttermilk as bm  # Global Buttermilk instance for saving
from buttermilk._core.agent import Agent  # Buttermilk base Agent class
from buttermilk._core.exceptions import FatalError, RateLimit  # Custom exceptions
from buttermilk._core.log import logger  # Buttermilk logger
from buttermilk._core.retry import TooManyRequests  # Custom exception for rate limits
from buttermilk._core.types import Record  # Buttermilk Record type

# Placeholder for gc object if it's meant to be globally available or configured elsewhere.
# For now, calls to gc.logger will be replaced by buttermilk's logger.
# Calls to gc.upload_* will be replaced by bm.save or noted as placeholders.

REQUEST_INTERVAL = 20
"""Base delay in seconds between HTTP requests. The actual delay will be a
random integer between `REQUEST_INTERVAL` and `REQUEST_INTERVAL * 2`.
Used in the synchronous `run` method of `WebScraperSelenium`.
"""


class WebScraperRequests(Agent):
    """A base web scraper agent that uses the `requests` library for fetching content.

    This agent provides basic functionality for making HTTP GET requests,
    handling retries for common network issues, and extracting data from
    responses. It is designed to be subclassed for specific scraping tasks
    by overriding methods like `extract` and `extract_next`.

    It does not use Selenium and thus cannot handle JavaScript-rendered content.

    Attributes:
        _session (requests.Session): A `requests.Session` object used to persist
            parameters (like cookies and headers) across requests. Initialized on first use.

    """

    _session: requests.Session = PrivateAttr(default_factory=requests.Session)
    """A `requests.Session` object for making HTTP requests.
    It's initialized with a default factory.
    """

    def add_cookies(self, cookies: dict[str, str] | None = None) -> None:
        """Adds cookies to the current requests session.

        Note:
            This method is not implemented in the base class and should be
            overridden by subclasses if cookie management is needed for the
            `requests` session. For Selenium, see `WebScraperSelenium.add_cookies`.

        Args:
            cookies (dict[str, str] | None): A dictionary of cookies to add.

        Raises:
            NotImplementedError: Always, as this method must be overridden.

        """
        raise NotImplementedError("Subclasses should implement add_cookies for requests session if needed.")

    def _process(self, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Processes a single record, typically representing a URL to scrape.

        Fetches the URL specified in `record["url"]` using the `get` method,
        then calls the `extract` method to parse the response. The extracted data
        is used to update the input `record` dictionary. Errors during fetching
        or processing are caught and stored in the `record["error"]` field.

        Args:
            record (dict[str, Any]): A dictionary containing at least a "url" key
                with the URL to scrape. It may also contain other contextual
                information like "parent_id".
            **kwargs: Additional keyword arguments passed to the `get` method
                      (and potentially to `extract` if it's overridden to accept them).

        Returns:
            dict[str, Any]: The input `record` dictionary, updated with extracted
            data under keys defined by the `extract` method, or an "error" key
            if an HTTP error or other processing exception occurred.

        """
        url = record.get("url")
        if not url:
            logger.error(f"WebScraperRequests ('{self.agent_id}')._process: Record missing 'url'. Record: {record}")
            record["error"] = "Missing 'url' in record."
            return record

        logger.debug(f"WebScraperRequests ('{self.agent_id}'): Fetching URL: {url}, context: parent_id {record.get('parent_id')}")

        try:
            response = self.get(url, **kwargs)  # Call the get method with retry logic
            extracted_data = self.extract(response)
            if isinstance(extracted_data, dict):
                record.update(extracted_data)
            else:
                logger.warning(f"WebScraperRequests ('{self.agent_id}'): extract() method did not return a dict for URL {url}. Got: {type(extracted_data)}")
                record["extracted_data_raw"] = extracted_data  # Store raw if not dict
        except requests.exceptions.HTTPError as e:
            error_message = str(e)
            record["error"] = error_message
            logger.warning(f"WebScraperRequests ('{self.agent_id}'): Received HTTP error for {url}: {error_message}.")
        except (RateLimit, FatalError) as e:  # Buttermilk specific exceptions
            logger.error(f"WebScraperRequests ('{self.agent_id}'): Critical error while processing {url}: {e!s} {e.args=}")
            record["error"] = str(e)
        except Exception as e:  # Catch any other unexpected errors
            logger.error(f"WebScraperRequests ('{self.agent_id}'): Unexpected error processing {url}: {e!s}", exc_info=True)
            record["error"] = f"Unexpected error: {e!s}"
        return record

    @retry(
        retry=retry_if_exception_type(
            (
                RateLimit,
                TooManyRequests,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                urllib3.exceptions.HTTPError,
            ),
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=1),  # Wait 1s, then up to 30s
    )
    def get(self, url: str, headers: dict[str, str] | None = None, **parameters: Any) -> requests.Response:
        """Makes an HTTP GET request to the specified URL with retry logic.

        Ensures a `requests.Session` exists (`self._session`) and updates its
        headers if `headers` are provided for this specific call. After a successful
        request, it calls `self.try_save_page` to attempt saving the response text.
        It will raise `requests.exceptions.HTTPError` for bad status codes (4xx or 5xx)
        after retries are exhausted.

        Args:
            url (str): The URL to fetch.
            headers (dict[str, str] | None): Optional dictionary of HTTP headers
                to include specifically for this request (merged with session headers).
            **parameters: Additional keyword arguments passed as URL parameters to
                `self._session.get()`.

        Returns:
            requests.Response: The response object from the HTTP GET request.

        Raises:
            requests.exceptions.HTTPError: If the response status code is 4xx or 5xx
                after all retry attempts.
            requests.exceptions.ConnectionError: For network connectivity issues after retries.
            requests.exceptions.Timeout: If the request times out after retries.

        """
        logger.debug(f"WebScraperRequests ('{self.agent_id}'): Fetching URL: {url} with parameters: {parameters}")

        if not hasattr(self, "_session") or self._session is None:
             self._session = requests.Session()

        current_session_headers = self._session.headers.copy()
        if headers:
            current_session_headers.update(headers)

        response = self._session.get(url, headers=current_session_headers, params=parameters)
        # Create a filename prefix from the URL for saving
        filename_prefix = re.sub(r"[^\w\-_\.]", "_", url)  # Sanitize URL for filename
        self.try_save_page(source=response.text, filename_prefix=filename_prefix[:100])  # Limit prefix length
        response.raise_for_status()
        return response

    def extract(self, response: requests.Response) -> dict[str, Any]:
        """Extracts relevant information from an HTTP response object.

        This base method simply returns the entire response text under the key "text".
        Subclasses should override this method to implement specific data extraction
        logic tailored to the content of the websites they are scraping (e.g.,
        parsing HTML with BeautifulSoup, extracting JSON data, etc.).

        Args:
            response (requests.Response): The HTTP response object from which to
                extract data.

        Returns:
            dict[str, Any]: A dictionary containing the extracted data.
            The base implementation returns `{"text": response.text}`.

        """
        return {"text": response.text}

    def extract_next(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Identifies and returns information for scraping subsequent pages (e.g., for pagination).

        This method is intended to be overridden by subclasses if the scraping task
        involves navigating through multiple pages or following links found on the
        current page. The base implementation provides an example logic that assumes
        the `record` might contain a "level" for recursion depth and a "nextPage"
        URL. It also assumes a `self.base_url` attribute might be defined in a subclass
        for constructing full URLs.

        Args:
            record (dict[str, Any]): The current record (a dictionary, typically
                containing data from a scraped page) from which next pages/links
                might be identified.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a subsequent record to be scraped. Each should contain
            necessary information like "url", "parent_id", and an updated "level".
            Returns an empty list if no further pages are identified or if the
            recursion depth (based on "level") is met.

        """
        current_level = record.get("level", 0)
        if current_level > 0 and record.get("nextPage"):
            base_url_template = getattr(self, "base_url", None)
            next_page_val = record["nextPage"]

            if base_url_template and isinstance(base_url_template, str):
                try:
                    next_url = base_url_template.format(nextPage=next_page_val)
                except KeyError:  # If nextPage is not a valid key for format
                    logger.warning(f"WebScraperRequests ('{self.agent_id}'): 'nextPage' key not found in base_url_template: {base_url_template}. Using nextPage value as full URL.")
                    next_url = str(next_page_val)
            else:
                next_url = str(next_page_val)

            next_record_info = record.copy()
            next_record_info.update({
                "parent_id": record.get("id", record.get("url")),
                "url": next_url,
                "level": current_level - 1,
            })
            next_record_info.pop("nextPage", None)
            next_record_info.pop("id", None)  # New record should get its own ID if applicable
            return [next_record_info]
        return []

    def try_save_page(self, **kwargs: Any) -> dict[str, str]:
        """Attempts to save page content using `save_page`, catching and logging any exceptions.

        Args:
            **kwargs: Arguments to pass to `self.save_page()`. Expected to include
                      `source` (the page content as a string) and optionally
                      `filename_prefix` for naming the saved file.

        Returns:
            dict[str, str]: A dictionary where keys are file types (e.g., "html")
            and values are their corresponding URIs or paths where the content was saved.
            Returns an empty dictionary if the saving operation fails for any reason.

        """
        try:
            return self.save_page(**kwargs)
        except Exception as e:
            logger.warning(f"WebScraperRequests ('{self.agent_id}'): Unable to save a copy of the page. Error: {e!s}", exc_info=True)
            return {}

    def save_page(self, **kwargs: Any) -> dict[str, str]:
        """Saves page content (e.g., HTML source). Subclasses should override for richer saving.

        This base implementation saves the provided "source" kwarg (assumed to be
        HTML text) to a file with an ".html" extension using `bm.save`.
        Subclasses (like `WebScraperSelenium`) override this to save additional
        content like screenshots.

        Args:
            **kwargs: Keyword arguments. Expected to contain "source" (the page
                      content as a string) and optionally "filename_prefix" to help
                      name the saved file.

        Returns:
            dict[str, str]: A dictionary where keys are file types (e.g., "html")
            and values are the URIs or paths to the saved files. Returns an empty
            dictionary if "source" is not provided or if saving fails.

        """
        save_paths: dict[str, str] = {}
        source_content = kwargs.get("source")
        if source_content and isinstance(source_content, str):
            basename = kwargs.get("filename_prefix", f"scraped_page_{shortuuid.uuid()[:8]}")
            saved_uri = bm.save(data=source_content, basename=basename, extension=".html")
            if saved_uri:
                save_paths["html"] = str(saved_uri)
            else:
                logger.warning(f"WebScraperRequests ('{self.agent_id}'): Failed to save HTML content using bm.save.")
        elif not source_content:
            logger.debug(f"WebScraperRequests ('{self.agent_id}'): No 'source' content provided to save_page.")
        else:
            logger.warning(f"WebScraperRequests ('{self.agent_id}'): 'source' content for save_page was not a string (type: {type(source_content)}).")
        return save_paths


class WebScraperSelenium(WebScraperRequests):
    """A web scraper agent that uses Selenium WebDriver for fetching and interacting with web pages.

    This class extends `WebScraperRequests` to leverage Selenium for scraping
    dynamic websites that heavily rely on JavaScript to render content. It manages
    a WebDriver instance for browser automation.

    Key functionalities include:
    -   Initializing and managing a Selenium WebDriver instance.
    -   Overriding `get` and `_process` methods to use the WebDriver for page fetching.
    -   Saving page HTML and screenshots using WebDriver capabilities.
    -   Extracting browser headers, cookies, and user agent.
    -   Providing helper methods for finding elements (`find_all`, `_extract`) and
        checking image visibility.

    Note:
        This class relies on several Selenium-specific imports (`By`, `NoSuchElementException`,
        `WebDriverException`, `StaleElementReferenceException`) which are conditionally
        imported at the top level for type hinting but are primarily expected to be
        available in the execution environment where Selenium is used.
        The `new_webdriver()` function is called to obtain a WebDriver instance but is
        not defined within this module; it must be provided from an external source
        (e.g., a utility module specific to the Selenium setup).

    Attributes:
        driver (WebDriver | Any): The Selenium WebDriver instance. Initialized to `None`
                                  and set up by `new_webdriver()`.

    """

    driver: WebDriver | Any = PrivateAttr(default=None)
    """The Selenium WebDriver instance used for browser automation."""

    def __init__(self, name: str, job: str, **kwargs: Any) -> None:
        """Initializes the WebScraperSelenium agent.

        Calls the superclass initializer (`WebScraperRequests`) and then initializes
        the Selenium WebDriver by calling `self.new_webdriver()`.

        Args:
            name (str): The name of the agent (passed to `Agent` base).
            job (str): The job identifier associated with this agent instance
                       (passed to `Agent` base, though `Agent` itself doesn't directly use it).
            **kwargs: Additional keyword arguments passed to the `Agent` superclass.

        """
        # Call Agent's __init__ via super() chain
        super().__init__(name=name, job=job, **kwargs)  # Assuming name & job are for Agent base

        self.new_webdriver()  # Initialize WebDriver

    def run(self, records: list[dict[str, Any]], recurse: int = 0, **kwargs: Any) -> int:
        """Synchronously processes a list of records (URLs to scrape) with optional recursion.

        Iterates through the provided `records`, calling `self._process` (the Selenium
        version) for each. If `recurse` is greater than 0, it calls `self.extract_next`
        on each processed record to find more URLs and recursively calls `run` for them.
        Implements a random delay between processing records (after the first few,
        controlled by `REQUEST_INTERVAL` and `self.wait_time`).

        Args:
            records (list[dict[str,Any]]): A list of records (dictionaries, each
                typically containing a "url") to scrape.
            recurse (int): The number of recursion levels allowed. If > 0,
                `extract_next` will be used to find more records to scrape.
            **kwargs: Additional keyword arguments passed to `self._process`.

        Returns:
            int: The total number of records processed (including recursive calls).

        Note:
            This is a synchronous method. For use in an asyncio environment
            (like Buttermilk agents often are), this would block the event loop.
            Consider adapting to an asynchronous pattern if this agent is to be
            used in async flows directly.
            `self.results` and `self.wait_time` are used but not explicitly defined
            as attributes in this class or its Pydantic model fields; they are assumed
            to be managed as instance attributes, potentially initialized by subclasses
            or configuration.

        """
        num_records_processed = 0
        records_for_recursion: list[dict[str, Any]] = []

        # Ensure self.results and self.wait_time exist
        if not hasattr(self, "results") or not isinstance(self.results, list):
            self.results: list[dict[str, Any]] = []
        if not hasattr(self, "wait_time"):  # wait_time should be configurable
            self.wait_time = REQUEST_INTERVAL

        for i, record_item in enumerate(records):
            processed_record = self._process(record_item, **kwargs)
            self.results.append(processed_record)  # Store processed record
            num_records_processed += 1

            if recurse > 0:
                next_records_info = self.extract_next(record=processed_record)
                records_for_recursion.extend(next_records_info)

            # Delay logic
            if i > 5 and i < (len(records) - 1):
                sleep_duration = random.randint(self.wait_time, self.wait_time * 2)
                logger.debug(f"WebScraperSelenium ('{self.agent_id}'): Sleeping for {sleep_duration} seconds between scrapes.")
                time.sleep(sleep_duration)

        if recurse > 0 and records_for_recursion:
            random.shuffle(records_for_recursion)
            num_records_processed += self.run(records=records_for_recursion, recurse=recurse - 1, **kwargs)

        return num_records_processed

    def quit(self) -> None:
        """Quits the Selenium WebDriver and performs cleanup.

        Attempts to close the browser window and properly terminate the WebDriver session.
        Then calls the superclass's `quit` method if it exists (though `WebScraperRequests`
        does not define one).
        """
        if self.driver:
            logger.info(f"WebScraperSelenium ('{self.agent_id}'): Quitting Selenium WebDriver.")
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"WebScraperSelenium ('{self.agent_id}'): Error quitting Selenium WebDriver: {e!s}", exc_info=True)
            self.driver = None

        if hasattr(super(), "quit") and callable(super().quit):  # type: ignore
            super().quit()  # type: ignore

    def new_webdriver(self) -> WebDriver | None:
        """Initializes or re-initializes the Selenium WebDriver.

        If an existing driver is present, it attempts to close it. Then, it calls
        an externally defined `new_webdriver()` function (expected to be in the global
        scope or imported elsewhere) to get a new WebDriver instance. The instance
        is assigned to `self.driver`. It also configures the logging level for
        "seleniumwire.handler" to WARNING to reduce noise if SeleniumWire is used.

        Returns:
            WebDriver | None: The newly created Selenium WebDriver instance, or `None`
            if creation fails (e.g., `new_webdriver` is not defined or raises an error).

        Note:
            This method relies on an external `new_webdriver()` function that must be
            available in the scope where this class is used.

        """
        if self.driver:
            try:
                self.driver.close()
                logger.info(f"WebScraperSelenium ('{self.agent_id}'): Closed existing Selenium session.")
                time.sleep(1)
            except Exception as e:
                logger.info(f"WebScraperSelenium ('{self.agent_id}'): Failed to close existing Selenium session: {e!s}")

        logger.info(f"WebScraperSelenium ('{self.agent_id}'): Starting new Selenium session.")
        try:
            # Assumes new_webdriver is available in the global scope or imported.
            if "new_webdriver" not in globals() or not callable(globals()["new_webdriver"]):
                 logger.error("`new_webdriver` function is not defined or not callable globally. Cannot create WebDriver.")
                 self.driver = None
            else:
                 self.driver = globals()["new_webdriver"]()

            if self.driver:
                logging.getLogger("seleniumwire.handler").setLevel(logging.WARNING)  # For SeleniumWire
        except NameError:  # Explicitly catch NameError if new_webdriver is not found
             logger.error("`new_webdriver` function is not defined. Selenium WebDriver cannot be created for agent '{self.agent_id}'.")
             self.driver = None
        except Exception as e:
            logger.error(f"WebScraperSelenium ('{self.agent_id}'): Error creating new Selenium WebDriver: {e!s}", exc_info=True)
            self.driver = None
        return self.driver

    def save_page(self, **kwargs: Any) -> dict[str, str]:
        """Saves the current page's HTML source and a PNG screenshot using Selenium.

        Overrides `WebScraperRequests.save_page`. It uses `self.driver` to get
        the page source and take a screenshot. Files are saved using `bm.save`.

        Args:
            **kwargs: Not directly used for content fetching (uses `self.driver`),
                      but `filename_prefix` can be passed to influence filenames.

        Returns:
            dict[str, str]: A dictionary mapping "html" and "png" to their respective
            saved file URIs/paths. Returns an empty dict if `self.driver` is not set
            or if saving fails.

        """
        if not self.driver or not hasattr(self.driver, "current_url"):  # Check if driver is valid
            logger.warning(f"WebScraperSelenium ('{self.agent_id}'): WebDriver not available or no page loaded, cannot save page.")
            return {}

        save_paths: dict[str, str] = {}
        page_url = self.driver.current_url
        # Create a base filename from URL or a default if URL is problematic
        base_filename = kwargs.get("filename_prefix") or re.sub(r"[^\w\-_\.]", "_", page_url.split("/")[-1] or f"page_{shortuuid.uuid()[:4]}")

        try:
            html_source = self.driver.page_source
            saved_html_uri = bm.save(data=html_source, basename=f"{base_filename}_source", extension=".html")
            if saved_html_uri: save_paths["html"] = str(saved_html_uri)

            screenshot_png_bytes = self.driver.get_screenshot_as_png()
            saved_png_uri = bm.save(data=screenshot_png_bytes, basename=f"{base_filename}_screenshot", extension=".png", mode="wb")
            if saved_png_uri: save_paths["png"] = str(saved_png_uri)

            logger.info(
                f"WebScraperSelenium ('{self.agent_id}'): Fetched {page_url}. "
                f"Saved HTML to: {save_paths.get('html')}, Screenshot to: {save_paths.get('png')}",
            )
        except Exception as e:
            logger.error(f"WebScraperSelenium ('{self.agent_id}'): Error saving page with Selenium for URL {page_url}: {e!s}", exc_info=True)
        return save_paths

    def update_session(self) -> None:
        """Updates the internal `requests.Session` (`self._session`) with headers and cookies
        from the current Selenium WebDriver state.

        This is useful if, after navigating with Selenium, subsequent requests
        need to be made with the `requests` library using the browser's current context
        (e.g., session cookies, user-agent).
        """
        if not self._session:
            self._session = requests.Session()

        if self.driver and self._session and hasattr(self.driver, "current_url"):
            try:
                current_url = self.driver.current_url
                if current_url and current_url.startswith("http"):
                    self._session.headers["referer"] = current_url

                    # Attempt to get headers if using Selenium-Wire (via self.get_headers)
                    # and cookies using standard Selenium's get_cookies.
                    # self.get_headers() is designed to return (headers_dict, cookies_list_or_dict)
                    browser_headers, browser_cookies_list = self.get_headers()
                    if browser_headers: self._session.headers.update(browser_headers)

                    # Update requests session cookies from WebDriver's cookies
                    selenium_cookies = self.driver.get_cookies()  # Standard Selenium method
                    for cookie in selenium_cookies:
                        if "name" in cookie and "value" in cookie:
                            self._session.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"))

                    user_agent = self.get_user_agent()
                    if user_agent: self._session.headers.update({"User-Agent": user_agent})
                    logger.debug(f"WebScraperSelenium ('{self.agent_id}'): Updated requests session from WebDriver state for URL: {current_url}")
            except Exception as e:
                logger.warning(f"WebScraperSelenium ('{self.agent_id}'): Failed to fully update requests session from WebDriver: {e!s}", exc_info=True)

    def _extract(self, item: WebElement, kind: str, value: str | None = None) -> Any:
        """Helper method to extract data (attribute or text) from a Selenium WebElement.

        Args:
            item (WebElement): The Selenium WebElement from which to extract data.
            kind (str): The type of extraction. Must be "attribute" or "text".
            value (str | None): If `kind` is "attribute", this specifies the name
                of the attribute to extract (e.g., "href", "src"). Required for
                attribute extraction.

        Returns:
            Any: The value of the specified attribute if `kind` is "attribute",
                 or the text content of the element if `kind` is "text".
                 Returns None or raises error for invalid kinds.

        Raises:
            NotImplementedError: If the specified `kind` is not supported.

        """
        if kind == "attribute":
            if value:  # Attribute name must be provided
                return item.get_attribute(value)
            logger.warning("_extract called with kind='attribute' but no attribute name (value) provided.")
            return None
        if kind == "text":
            return item.text
        raise NotImplementedError(
            f"Extraction kind '{kind}' with value '{value}' is not supported by _extract.",
        )

    def find_all(self, element: WebDriver | WebElement, xpath: str, kind: str = "attribute", value: str | None = None) -> list[Any]:
        """Finds all web elements matching an XPath expression starting from a given `element`
        and extracts specified data (attribute or text) from each.

        Args:
            element (WebDriver | WebElement): The parent Selenium WebDriver or WebElement
                instance from which to start the search.
            xpath (str): The XPath expression used to find child elements.
            kind (str): The type of data to extract from each found element.
                Supported values are "attribute" or "text". Defaults to "attribute".
            value (str | None): If `kind` is "attribute", this is the name of the
                attribute to extract (e.g., "href", "src"). Required if `kind` is "attribute".
                Defaults to None.

        Returns:
            list[Any]: A list containing the extracted data (attribute values or text content)
                       from all found elements. Returns an empty list if no elements match
                       the XPath or if extraction fails for all found elements.

        Note:
            This method relies on `selenium.webdriver.common.by.By` and
            `selenium.common.exceptions.NoSuchElementException` being available
            in the execution scope.

        """
        results: list[Any] = []
        if not self.driver:  # Ensure driver is available if element is not specified for search
            logger.warning("WebDriver not available in find_all.")
            return results

        try:
            # Ensure By and NoSuchElementException are available
            # These are typically imported at the top of a Selenium-using file
            items = element.find_elements(By.XPATH, value=xpath)
            for item_element in items:
                extracted_value = self._extract(item_element, kind, value)
                if extracted_value is not None:
                    results.append(extracted_value)
        except NoSuchElementException:  # Should not be raised by find_elements (returns empty list)
            logger.debug(
                f"No elements found for XPath '{xpath}' within the given element.",
            )
        except Exception as e:  # Catch other potential errors
            logger.error(f"WebScraperSelenium ('{self.agent_id}'): Error in find_all with XPath '{xpath}': {e!s}", exc_info=True)
        return results

    def checkImageVisible(self, img_element: WebElement, min_size: int = 50) -> bool:
        """Checks if a given Selenium WebElement (assumed to be an image) is visible on the page
        and meets a minimum size requirement.

        Args:
            img_element (WebElement): The Selenium WebElement representing an image tag.
            min_size (int): The minimum natural width the image must have to be
                considered valid. Defaults to 50 pixels.

        Returns:
            bool: `True` if the image element is on the page, displayed (not hidden
                  by CSS), and its `naturalWidth` property is greater than `min_size`.
                  Returns `False` otherwise or if any error occurs during checks.

        """
        try:
            if not self.checkImageOnPage(img_element, min_size): return False  # Check size and basic presence
            if not img_element.is_displayed(): return False  # Check if styled to be visible
        except StaleElementReferenceException:  # Handle if element becomes stale
            logger.debug("StaleElementReferenceException during checkImageVisible.")
            return False
        except Exception as e:
            logger.debug(f"WebScraperSelenium ('{self.agent_id}'): Error checking image visibility: {e!s}")
            return False
        return True

    def checkImageOnPage(self, img_element: WebElement, min_size: int = 50) -> bool:
        """Checks if a Selenium WebElement (image) has a `naturalWidth` greater than
        `min_size` and is not styled with `display: none`.

        This is a helper for `checkImageVisible` focusing on intrinsic size and CSS display.

        Args:
            img_element (WebElement): The Selenium WebElement for the image.
            min_size (int): The minimum natural width required. Defaults to 50.

        Returns:
            bool: `True` if the image meets the size and CSS display criteria, `False` otherwise.

        """
        try:
            natural_width = img_element.get_property("naturalWidth")
            # Ensure natural_width is a number and greater than min_size
            if not (isinstance(natural_width, (int, float)) and natural_width > min_size):
                return False
            # Check if element is hidden by CSS 'display: none'
            if "none" in str(img_element.value_of_css_property("display")).lower():
                return False
        except StaleElementReferenceException:
            logger.debug("StaleElementReferenceException during checkImageOnPage.")
            return False
        except Exception as e:
            logger.debug(f"WebScraperSelenium ('{self.agent_id}'): Error checking image on page (size/display): {e!s}")
            return False
        return True

    def add_cookies(self, cookies: dict[str, Any] | None = None) -> None:
        """Adds cookies to the Selenium WebDriver session for the current domain.

        Overrides `WebScraperRequests.add_cookies`.
        If the `cookies` dictionary is provided, it expects a "first_url" key
        (string URL). The method first navigates to this URL to set the correct
        domain context for the WebDriver. Then, it adds cookies defined under a
        "cookies" key within the input `cookies` dictionary (this should be a
        dictionary of cookie names to string values).

        Args:
            cookies (dict[str, Any] | None): A dictionary structured as:
                `{"first_url": "http://example.com", "cookies": {"name1": "value1", "name2": "value2"}}`.
                If `None` or improperly structured, the method does nothing or logs a warning.

        """
        if not cookies or not self.driver:
            logger.debug(f"WebScraperSelenium ('{self.agent_id}'): No cookies to add or driver not available.")
            return

        first_url = cookies.get("first_url")
        cookie_dict = cookies.get("cookies")

        if not isinstance(first_url, str) or not isinstance(cookie_dict, dict):
            logger.warning(
                f"WebScraperSelenium ('{self.agent_id}'): add_cookies requires 'first_url' (str) and 'cookies' (dict) in the input.",
            )
            return

        try:
            self.driver.get(first_url)  # Navigate to the domain to set cookies correctly
            for name, value in cookie_dict.items():
                # Selenium's add_cookie expects a dict with specific keys.
                self.driver.add_cookie({"name": name, "value": str(value)})
            logger.info(f"WebScraperSelenium ('{self.agent_id}'): Added {len(cookie_dict)} cookies for domain of {first_url}.")
        except Exception as e:
            logger.error(f"WebScraperSelenium ('{self.agent_id}'): Error adding cookies in Selenium for URL {first_url}: {e!s}", exc_info=True)

    def get_user_agent(self) -> str | None:
        """Gets the User-Agent string from the current Selenium WebDriver session.

        Returns:
            str | None: The User-Agent string if available, otherwise `None`
            (e.g., if driver is not initialized or script execution fails).

        """
        if not self.driver: return None
        try:
            return self.driver.execute_script("return navigator.userAgent")
        except Exception as e:
            logger.warning(f"WebScraperSelenium ('{self.agent_id}'): Could not get User-Agent from Selenium: {e!s}")
            return None

    def get_headers(self) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Attempts to get HTTP headers and cookies from the current page using Selenium-Wire.

        This method relies on `self.driver.requests` being available, which is
        a feature of Selenium-Wire, not standard Selenium. If `self.driver.requests`
        is not available (e.g., using standard Selenium), it attempts to return
        only cookies using standard Selenium `driver.get_cookies()`.

        Returns:
            tuple[dict[str, str], list[dict[str, Any]]]: A tuple containing:
                - A dictionary of response headers for the current page URL (if using Selenium-Wire).
                - A list of cookie dictionaries as returned by `driver.get_cookies()`.
                Returns `({}, [])` if headers/cookies cannot be retrieved or driver is unavailable.

        """
        if not self.driver or not hasattr(self.driver, "current_url"): return {}, []

        headers_dict: dict[str, str] = {}
        cookies_list: list[dict[str, Any]] = []

        try:
            # Selenium-Wire specific: try to get headers from captured requests
            if hasattr(self.driver, "requests") and self.driver.requests:
                for request_obj in self.driver.requests:
                    if request_obj.response and request_obj.url == self.driver.current_url:
                        headers_dict = dict(request_obj.response.headers)  # Convert to simple dict
                        break  # Found headers for current URL

            # Standard Selenium: get cookies
            cookies_list = self.driver.get_cookies()

        except AttributeError:
            logger.debug("WebScraperSelenium ('{self.agent_id}'): driver.requests not available (not using Selenium-Wire or no requests captured). Only standard cookies will be retrieved.")
            try:  # Try to get standard cookies even if headers failed
                cookies_list = self.driver.get_cookies()
            except Exception as e_cookie:
                 logger.warning(f"WebScraperSelenium ('{self.agent_id}'): Error getting cookies from standard Selenium: {e_cookie!s}")
        except Exception as e:
            logger.warning(f"WebScraperSelenium ('{self.agent_id}'): Error getting headers/cookies: {e!s}")

        return headers_dict, cookies_list

    @retry(
        retry=retry_if_exception_type(
            (
                TooManyRequests,
                urllib3.exceptions.HTTPError,
                WebDriverException,  # Base for many Selenium errors
                StaleElementReferenceException,  # Specific Selenium error
            ),
        ),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(3),  # Adjusted retry attempts
    )
    def _process(self, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Processes a single record (URL) using Selenium WebDriver.

        Overrides `WebScraperRequests._process`.
        Navigates to the URL in `record["url"]` using the Selenium `get` method,
        then calls `self.extract` (which should be implemented by subclasses to
        work with Selenium elements or `self.driver.page_source`) to parse the
        response. The extracted data updates the `record`. Includes retry logic
        for common WebDriver exceptions.

        Args:
            record (dict[str, Any]): A dictionary containing at least a "url" key
                with the URL to scrape.
            **kwargs: Additional keyword arguments (not directly used by this method
                      but passed down for potential use in `get` or `extract` by subclasses).

        Returns:
            dict[str, Any]: The updated `record` dictionary, containing extracted
            data under keys defined by the `extract` method, or an "error" key
            if processing failed after retries.

        Raises:
            WebDriverException: If WebDriver operations fail after all retry attempts
                (or other non-retried Selenium exceptions).

        """
        url_to_fetch = record.get("url")
        if not url_to_fetch:
            logger.error(f"WebScraperSelenium ('{self.agent_id}')._process: Record missing 'url'. Record: {record}")
            record["error"] = "Missing 'url' in record."
            return record

        try:
            page_source = self.get(url_to_fetch, **kwargs)  # Selenium-specific get, returns page source
            extracted_data = self.extract(page_source)  # Pass page source to extract

            if isinstance(extracted_data, dict):
                record.update(extracted_data)
            else:  # If extract returns non-dict, store it under a specific key
                record["extracted_content_raw"] = extracted_data
                logger.warning(f"WebScraperSelenium ('{self.agent_id}'): extract() method for URL {url_to_fetch} did not return a dict. Raw data stored.")
            return record
        except StaleElementReferenceException as e_stale:  # More specific handling if needed before retry
            logger.warning(f"WebScraperSelenium ('{self.agent_id}'): StaleElementReferenceException for {url_to_fetch}. Retrying. Error: {e_stale!s}")
            record["error"] = f"Stale element: {e_stale!s}"  # Temporary error, may be overwritten by retry
            raise  # Re-raise for tenacity to handle
        except WebDriverException as e_wd:  # Catch other WebDriver exceptions
            error_msg = f"WebDriverException processing URL {url_to_fetch} with Selenium: {e_wd!s}"
            logger.error(error_msg, exc_info=True)
            record["error"] = error_msg
            raise  # Re-raise for tenacity
        except Exception as e:  # Catch other broad exceptions during processing
            error_msg = f"Unexpected error processing URL {url_to_fetch} with Selenium: {e!s}"
            logger.error(error_msg, exc_info=True)
            record["error"] = error_msg
            # Not re-raising here means the loop in `run` would continue.
            # Depending on desired behavior, could re-raise a non-retryable error.
            return record  # Return record with error

    @retry(
        retry=retry_if_exception_type(
            (
                TooManyRequests,
                urllib3.exceptions.HTTPError,  # General HTTP issues
                WebDriverException,  # Base Selenium WebDriver exception
            ),
        ),
        wait=wait_exponential_jitter(initial=2, max=60, jitter=5),  # Start with 2s wait
        stop=stop_after_attempt(3),  # Retry up to 3 times
    )
    def get(self, url: str, **parameters: Any) -> str:
        """Navigates to a URL using Selenium WebDriver and saves page content.

        Overrides `WebScraperRequests.get`.
        Uses `self.driver.get(url)` to load the page. After successful navigation,
        it calls `self.try_save_page` to save HTML and a screenshot, and
        `self.update_session` to synchronize context with the `requests.Session`.
        Includes retry logic for `WebDriverException`. If a `WebDriverException`
        occurs, it attempts to re-initialize the WebDriver once before the next
        retry attempt by tenacity.

        Args:
            url (str): The URL to navigate to.
            **parameters: Additional keyword arguments (currently unused in `driver.get`
                          but available for `try_save_page`).

        Returns:
            str: The page source HTML of the loaded page.

        Raises:
            RuntimeError: If the WebDriver is not initialized or if re-initialization fails.
            WebDriverException: If `driver.get` fails after all retry attempts.

        """
        if not self.driver:
            logger.error(f"WebScraperSelenium ('{self.agent_id}'): Selenium WebDriver not initialized. Cannot fetch URL: {url}")
            raise RuntimeError(f"Selenium WebDriver not initialized in WebScraperSelenium.get() for agent '{self.agent_id}'")

        logger.debug(f"WebScraperSelenium ('{self.agent_id}'): Fetching URL with Selenium: {url}")
        try:
            self.driver.get(url)
            self.try_save_page(**parameters)
            self.update_session()
        except WebDriverException as e:
            logger.warning(
                f"WebScraperSelenium ('{self.agent_id}'): WebDriverException while getting URL: {url}. Error: {e!s}. "
                "Attempting to restart WebDriver before next retry (if any).",
            )
            self.new_webdriver()  # Attempt to restart the driver
            if not self.driver:  # If restart failed
                 raise RuntimeError(f"Failed to restart WebDriver for agent '{self.agent_id}' after error on URL: {url}") from e
            raise  # Re-raise the original WebDriverException for tenacity to handle retry
        return self.driver.page_source

    def fetch_urls(self, urls: list[str] | None = None, recurse: int = 0, parent_id: str | None = None, **kwargs: Any) -> int:
        """Fetches a list of URLs, creating initial `Record` objects for them.

        This method is a convenience wrapper around `fetch_records`. It converts
        a list of URL strings into a list of `Record` objects (each containing
        the URL and optional `parent_id`) and then calls `fetch_records`.

        Args:
            urls (list[str] | None): A list of URL strings to scrape. If None or empty,
                                     no action is taken.
            recurse (int): The depth for recursive scraping. Passed to `fetch_records`.
                           Defaults to 0 (no recursion).
            parent_id (str | None): An optional identifier for a parent entity,
                                    to be associated with all created records.
            **kwargs: Additional keyword arguments passed to `fetch_records`.

        Returns:
            int: The total number of records processed, as returned by `fetch_records`.
                 Returns 0 if `urls` is None or empty.

        """
        if not urls:
            logger.debug(f"WebScraperSelenium ('{self.agent_id}'): fetch_urls called with no URLs.")
            return 0

        # Create Record objects, ensuring metadata is initialized for potential updates
        initial_records = [Record(url=url, parent_id=parent_id, metadata={}) for url in urls]
        return self.fetch_records(records=initial_records, recurse=recurse, **kwargs)

    def fetch_records(self, records: list[Record], recurse: int = 0, **kwargs: Any) -> int:
        """Processes a list of `Record` objects, typically by scraping their URLs.

        This method shuffles the input `records` list (to potentially vary scrape
        order) and then calls `self.run` to process them.
        It assumes `self.run` is adapted to handle a list of `Record` objects
        or that `Record` objects are converted to dictionaries if `self.run` expects that.

        Args:
            records (list[Record]): A list of `Record` objects to process. Each record
                                   should ideally have its `url` attribute set.
            recurse (int): The depth for recursive scraping. Passed to `self.run`.
                           Defaults to 0.
            **kwargs: Additional keyword arguments passed to `self.run`.

        Returns:
            int: The total number of records processed, as returned by `self.run`.

        """
        if not records:
            logger.debug(f"WebScraperSelenium ('{self.agent_id}'): fetch_records called with no records.")
            return 0

        # The current self.run expects a list of dictionaries.
        # Convert Record objects to dicts before passing to self.run.
        records_as_dicts = [r.model_dump(exclude_none=True) for r in records]
        random.shuffle(records_as_dicts)

        return self.run(records=records_as_dicts, recurse=recurse, **kwargs)  # type: ignore
