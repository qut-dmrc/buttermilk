import random
import time
from typing import Any

import googleapiclient.errors
import requests

# import selenium.common.exceptions
import urllib3
from pydantic import PrivateAttr

from buttermilk._core.agent import Agent

########################
# WebScraper(): a modular web scraping class
#
# Usage:
#   Scraper.processors: A list of functions that accept a dictionary representing a single record and return either a
#                       dictionary or a list of dictionaries that will ultimately become the dataset you want to store.
#
#                       At a minimum, you must create a process() method that will extract data from your URLs
#
#   Scraper.fetch_urls(urls, recurse=None, parent_id=None):
#                       urls:       a list of initial URLs to scrape
#                       recurse:    Integer. If > 0, you must overwrite the Scraper.recurse() method that identifies
#                                   further URLs to scrape. You will probably want to decrement recurse by one and
#                                   pass these to fetch_urls again, but you can implement new scraping logic.
#                       parent_id:  an identifer that will stay with each record generated, allowing you to reconstruct
#                                   scraping hierarchies later on.
#
# Output:
#   Scraper.results:      A list of dicts, interpreted by the processors you specify
#
#
########################

REQUEST_INTERVAL = 20  # Delay between http requests. This is interpreted as randint(REQUEST_INTERVAL, REQUEST_INTERVAL*2)


################################
# WebScraperRequests does not use Selenium, only requests.
# The subclass Scraper() includes the Selenium webdriver (local or remote).
################################
class WebScraperRequests(Agent):
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)

    def add_cookies(self, cookies=None):
        raise NotImplementedError

    # A processor operates on a record and returns a result
    def _process(self, record, **kwargs):
        url = record["url"]

        logger.debug(f"Fetching URL: {url}, child of {record.get('parent_id')}")

        try:
            response = self.get(url, **kwargs)
            record.update(self.extract(response))
        except requests.exceptions.HTTPError as e:
            record["error"] = e.strerror
            gc.logger.warning(f"Received HTTP error searching for {url}: {e}.")
        except (RateLimit, FatalError) as e:
            gc.logger.error(f"Fatal error while processing {url}: {e} {e.args=}")

        return record

    # Retry if we have a connection problem or hit the rate limit
    @retry(
        retry=retry_if_exception_type(
            (
                RateLimit,
                TooManyRequests,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.HTTPError,
                selenium.common.exceptions.WebDriverException,
            ),
        ),
        # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        # Retry up to five times before giving up
        stop=stop_after_attempt(5),
    )
    def get(self, url, headers=None, **params):
        gc.logger.debug(f"Fetching URL: {url}")

        if not self.session:
            self.new_session()

        try:
            self.session.headers.update(headers)
        except TypeError:
            gc.logger.debug("Unable to set headers for requests session.")
            # unable to set headers

        response = self.session.get(url)
        self.try_save_page(source=response.text)
        response.raise_for_status()
        return response

    def extract(self, response) -> dict:
        # Get information out of the response and return it in a dict
        return dict(text=response.text)

    def extract_next(self, record) -> list[dict[str, Any]]:
        # Overwrite this for your use case
        # Returns a list of Records()
        if record.get("level", 0) > 0:
            if self.base_url:
                url = self.base_url.format(nextPage=record["nextPage"])
            else:
                url = record["nextPage"]

            next_record = Record(
                parent_id=record.pop("id"),
                url=url,
                level=record.pop("level") - 1,
                **record,
            )
            return [next_record]

        return list()

    def try_save_page(self, **kwargs: any) -> dict:
        try:
            return self.save_page(**kwargs)
        except Exception as e:
            gc.logger.warning(f"Unable to save a copy to GCS. Error: {e} {e.args=}")
            return {}

    def save_page(self, **kwargs: Any) -> dict:
        # Overwrite this method in your class if you want to save screenshots
        # Returns a dict of uris in the format: {'xls': 'uri', 'pdf': 'other_uri', ...}
        save_paths = dict()

        if kwargs.get("source"):
            # Save the HTML for reuse later.
            save_uri = gc.upload_text(kwargs["source"], extension="html")
            save_paths["html"] = save_uri

        return save_paths


class WebScraperSelenium(WebScraperRequests):
    ################################
    ##
    # The code in this class should be specific to use with the Selenium webdriver.
    ##
    ################################
    driver: webdriver.WebDriver

    def __init__(self, name, job, **kwargs) -> None:
        self.driver = None

        super(WebScraperSelenium, self).__init__(name, job, **kwargs)

        # init our webdriver
        self.driver = self.new_webdriver()

    # SYNCHRONOUS CONTROLLER METHOD #################################
    def run(self, records, recurse=0, **kwargs) -> int:
        num_records = 0
        recurse_records = []

        for i, record in enumerate(records):
            record = self._process(record)

            # Add this page's info to our stored results
            self.results.append(record)

            num_records += 1

            if recurse and recurse > 0:
                recurse_records.extend(self.extract_next(record=record))

            if (
                i > 5 and i < len(records) - 1
            ):  # don't wait the first few times -- speeds up testing.
                wait = random.randint(self.wait_time, self.wait_time * 2)
                gc.logger.debug(f"Sleeping for {wait} seconds.")
                time.sleep(self.wait_time)

        # Fetch any child pages referred to on these pages, if applicable.
        if recurse and recurse > 0 and recurse_records:
            random.shuffle(recurse_records)
            num_records += self.run(records=recurse_records, recurse=recurse - 1)

        return num_records

    def quit(self):
        # try to delete temp profiles etc.
        logger.info("Quitting driver, trying to quit nicely.")
        self.driver.quit()

        super(WebScraperRequests, self).quit()

    def new_webdriver(self):
        if self.driver:
            try:
                self.driver.close()
                gc.logger.info("Closed existing session")
                time.sleep(1)
            except Exception as e:
                gc.logger.info(f"Failed to close session: {e} {e.args=}")

        gc.logger.info("Starting new session")

        # This is driver.new_webdriver(), not self.new_webdriver()
        self.driver = new_webdriver()

        # The seleniumwire debugger is very noisy. Turn it down if we used it.
        logging.getLogger("seleniumwire.handler").setLevel(logging.WARNING)
        return self.driver

    @retry(
        retry=retry_if_exception_type(
            (TimeoutError, googleapiclient.errors.Error),
        ),
        # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        # Retry up to five times before giving up
        stop=stop_after_attempt(5),
    )
    def save_page(self, **kwargs):
        save_paths = dict()

        # Save the HTML for reuse later.
        save_uri = gc.upload_text(self.driver.page_source, extension="html")
        save_paths["html"] = save_uri

        screenshot = self.driver.get_screenshot_as_png()
        save_uri = gc.upload_binary(screenshot, extension="png")
        save_paths["png"] = save_uri

        gc.logger.info(
            f"Fetched {self.driver.current_url}, saved a copy to {save_paths['html']}, and uploaded screenshot to: {save_paths['png']}",
        )
        return save_paths

    def update_session(self):
        if not self.session:
            self.session = self.new_session()

        #  this time, also make sure we match the headers and cookies of the webdriver if we can.
        # Copy the request headers from the webdriver if it has loaded a page already
        if self.driver and self.session:
            if self.driver.current_url[:4] == "http":
                self.session.headers["referer"] = self.driver.current_url
                headers, cookies = self.get_headers()
                self.session.headers.update(headers)

                # TODO: also add cookies
                # self.session.cookies = cookies

            # Set user agent to match
            agent = self.get_user_agent()
            self.session.headers.update({"User-Agent": agent})

    def _extract(self, item, kind, value=None):
        if kind == "attribute" and value:
            return item.get_attribute(value)
        if kind == "text":
            return item.text
        raise NotImplementedError(
            f"find function does not support searches for: {kind} with value: {value}",
        )

    def find_all(self, element, xpath, kind="attribute", value=None):
        assert kind is not None, "You must pass a type of search to run."

        results = []
        try:
            items = element.find_elements(by=By.XPATH, value=xpath)

            for item in items:
                values = self._extract(item, kind, value)
                if values:
                    results.append(values)
        except NoSuchElementException:
            logger.debug(
                f'Unable to find xpath "{xpath}" in element: {element.tag_name}:{element.id}',
            )

        return results

    def checkImageVisible(self, img_element, min_size=50):
        try:
            assert self.checkImageOnPage(img_element, min_size)
            assert img_element.is_displayed()
        except AssertionError:
            return False

        return True

    def checkImageOnPage(self, img_element, min_size=50):
        # Trying to understand whether the image is just lower in the screen
        try:
            assert img_element.get_property("naturalWidth") > min_size
            assert "none" not in str.lower(img_element.value_of_css_property("display"))
        except AssertionError:
            return False

        return True

    def add_cookies(self, cookies=None):
        if not cookies:
            return

        # first, we need to load a webpage from the correct domain
        url = cookies.get("first_url")
        self.get(url)

        # Now we can add cookies for the domain
        for k, v in cookies["cookies"].items():
            self.driver.add_cookie(dict(name=k, value=v))

    def get_user_agent(self):
        return self.driver.execute_script("return navigator.userAgent")

    def get_headers(self):
        # Access requests via the `requests` attribute
        try:
            for request in self.driver.requests:
                if request.response:
                    if request.url == self.driver.current_url:
                        headers = request.response.headers
                        cookies = self.driver.get_cookies()
                        return headers, cookies
        except AttributeError:  # most likely we're using selenium, not seleniumwire
            pass

        # either we couldn't get the headers, or there's no URL loaded by the driver yet.
        return dict(), list()

    @retry(
        selenium.common.exceptions.StaleElementReferenceException,
        delay=120,
        tries=1,
    )
    def _process(self, record, **kwargs):
        try:
            self.get(record["url"], **kwargs)

            record.update(self.extract(record, **kwargs))

            return record
        except selenium.common.exceptions.StaleElementReferenceException as e:
            # something wrong with the browser, wait two minutes and try again
            logger.warning("Browser window went stale, trying again in two minutes.")
            err = f"Error processing record {record.get('url')}: {e}"
            record["error"] = err

            self.results.append(record)

            raise

    @retry(
        retry=retry_if_exception_type(
            (
                TooManyRequests,
                urllib3.exceptions.HTTPError,
                selenium.common.exceptions.WebDriverException,
            ),
        ),
        # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        # Retry up to five times before giving up
        stop=stop_after_attempt(5),
    )
    def get(self, url, **params):
        gc.logger.debug(f"Fetching URL: {url}")
        try:
            self.driver.get(url)  # Use webdriver to get the new page
            self.try_save_page(**params)
            self.update_session()  # Ensure our requests session is updated with the new headers
        except selenium.common.exceptions.WebDriverException as e:
            gc.logger.warning(
                f"Unable to get requested URL due to webdriver error. URL: {url}. Error: {e} {e.args=}",
            )

            # kill session and try again?
            self.new_webdriver()
            self.new_session()
            raise
        return self.driver.page_source

    def fetch_urls(self, urls: list[str] = None, recurse=0, parent_id=None, **kwargs):
        records = []
        for url in urls:
            records.append(Record(url=url, parent_id=parent_id))

        return self.fetch_records(records, **kwargs)

    def fetch_records(self, records: list[Record], recurse=0, **kwargs):
        random.shuffle(records)

        return self.run(records, recurse=recurse, **kwargs)
