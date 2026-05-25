"""
Ping the deployed Streamlit app and wake it if the sleep page is shown.
Used by GitHub Actions to reduce cold-start delays on Streamlit Community Cloud.
"""

import os
import sys

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://rag-chatbot-ix99syqnr8b65ygj3wa9gm.streamlit.app/",
)
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
AWAKE_MARKERS = ("rag chatbot", "chat with your documents")
SLEEP_MARKERS = ("gone to sleep", "zzzz")


def is_sleeping(html: str) -> bool:
    lower = html.lower()
    return any(marker in lower for marker in SLEEP_MARKERS)


def is_awake(html: str) -> bool:
    lower = html.lower()
    return any(marker in lower for marker in AWAKE_MARKERS)


def main() -> int:
    print(f"Checking Streamlit app: {APP_URL}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)

            content = ""
            for _ in range(12):
                page.wait_for_timeout(5_000)
                content = page.content()
                if is_awake(content) or is_sleeping(content):
                    break

            if is_awake(content):
                print("App is already awake.")
                return 0

            if is_sleeping(content):
                print("App is sleeping. Clicking wake button...")
                page.get_by_role("button", name=WAKE_BUTTON_TEXT).click(timeout=15_000)

                for _ in range(60):
                    page.wait_for_timeout(5_000)
                    if is_awake(page.content()):
                        print("App woke up successfully.")
                        return 0

                print("Failed to wake app within timeout.", file=sys.stderr)
                return 1

            print("App responded but status is unclear. Treating ping as success.")
            return 0
        except Exception as exc:
            print(f"Error while checking app: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
