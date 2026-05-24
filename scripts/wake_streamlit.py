"""
Ping the deployed Streamlit app and wake it if the sleep page is shown.
Used by GitHub Actions to reduce cold-start delays on Streamlit Community Cloud.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://rag-chatbot-ix99syqnr8b65ygj3wa9gm.streamlit.app/",
)
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
APP_READY_SELECTOR = "[data-testid='stAppViewContainer']"


def main() -> int:
    print(f"Checking Streamlit app: {APP_URL}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(3000)

            content = page.content().lower()
            is_sleeping = "gone to sleep" in content or "zzzz" in content

            if is_sleeping:
                print("App is sleeping. Clicking wake button...")
                page.get_by_role("button", name=WAKE_BUTTON_TEXT).click(timeout=15_000)
                page.wait_for_selector(APP_READY_SELECTOR, timeout=300_000)
                print("App woke up successfully.")
            else:
                print("App is already awake.")
                page.wait_for_selector(APP_READY_SELECTOR, timeout=60_000)
                print("App loaded successfully.")

            return 0
        except PlaywrightTimeoutError as exc:
            print(f"Timed out while waking the app: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
